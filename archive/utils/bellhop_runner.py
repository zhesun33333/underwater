"""
BELLHOP 调用封装
.env 生成 → bellhop.exe 执行 → .arr 解析 → CIR 构建 → 卷积
"""
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import fftconvolve

try:
    from scipy.signal import kaiser
except ImportError:
    from scipy.signal.windows import get_window

    def kaiser(M, beta):
        return get_window(("kaiser", beta), M)


class NoArrivalsError(RuntimeError):
    """BELLHOP completed, but no receiver arrivals were produced."""


def resolve_num_beams(
    setting,
    range_km: float,
    min_beams: int = 1000,
    beams_per_km: float = 100.0,
    max_beams: int = 8000,
) -> int:
    """Resolve fixed, BELLHOP-auto, or bounded distance-adaptive beam count."""
    if isinstance(setting, str):
        mode = setting.strip().lower()
        if mode == "bellhop_auto":
            return 0
        if mode != "adaptive":
            raise ValueError(f"Unsupported num_beams mode: {setting}")
    elif setting is not None and int(setting) > 0:
        return int(setting)

    lower = max(3, int(min_beams))
    upper = max(lower, int(max_beams))
    estimated = int(np.ceil(float(range_km) * float(beams_per_km)))
    return min(upper, max(lower, estimated))


# ---- BELLHOP .env 文件生成 ----

def generate_env(
    env_path: Path,
    title: str,
    freq_hz: float,
    depths_m: np.ndarray,
    sound_speeds_mps: np.ndarray,
    water_depth_m: float,
    source_depth_m: float,
    receiver_depth_m: float,
    range_km: float,
    bottom_type: str = "silt",
    bottom_sound_speed_mps: float = 1520.0,
    bottom_density_gcc: float = 1.4,
    bottom_attenuation_db_per_lambda: float = 3.0,
    num_beams: int = 0,
    ray_box_margin: float = 1.05,
):
    """
    生成 BELLHOP .env 输入文件。

    使用标准的 BELLHOP 语法，包含：
    - 声速剖面 (SSP)
    - 半空间海底
    - 单发射深度 / 单接收深度 / 单水平距离
    - 'A' 选项输出到达文件 (.arr)
    """
    lines = []
    lines.append(f"'{title}'")
    lines.append(f"{freq_hz:.2f}")
    lines.append("1")
    lines.append("'CVF'")

    # 裁剪 SSP 到水底深度以上，避免 BELLHOP 读到超深点报错
    mask = depths_m <= water_depth_m + 1e-6
    if mask.sum() < 2:
        raise ValueError(f"SSP 在水深 {water_depth_m:.1f}m 以上点数不足 ({mask.sum()})，无法运行 BELLHOP")
    z_trim = depths_m[mask]
    c_trim = sound_speeds_mps[mask]
    # 确保最后一个点在水底深度
    if abs(z_trim[-1] - water_depth_m) > 0.01:
        # 线性插值水底声速
        c_bottom = np.interp(water_depth_m, depths_m, sound_speeds_mps)
        z_trim = np.append(z_trim, water_depth_m)
        c_trim = np.append(c_trim, c_bottom)

    # Bellhop 读取顺序：NPts, Sigma, BottomDepth
    lines.append(f"{len(z_trim)}  1.0  {water_depth_m:.2f}")

    # SSP 记录：每行 depth sound_speed /
    for z, c in zip(z_trim, c_trim):
        lines.append(f"{z:.2f}  {c:.2f}  /")

    # 底边界条件：BC + Sigma
    lines.append("'A'  0.0")

    # 底部半空间参数：depth, cp, cs, rho, alphaP, alphaS
    lines.append(f"{water_depth_m:.2f}  {bottom_sound_speed_mps:.2f}  0.0  "
                 f"{bottom_density_gcc:.3f}  {bottom_attenuation_db_per_lambda:.1f}  0.0 /")

    # 源/接收/距离
    lines.append("1")
    lines.append(f"{source_depth_m:.2f} /")
    lines.append("1")
    lines.append(f"{receiver_depth_m:.2f} /")
    lines.append("1")
    lines.append(f"{range_km:.4f} /")

    # 运行类型与束参数
    lines.append("'A'")
    # 0 lets BELLHOP choose a range/frequency/depth-aware beam count.
    lines.append(f"{int(num_beams)}")
    lines.append("-90 90 /")
    margin = max(1.01, float(ray_box_margin))
    z_box_m = max(water_depth_m * margin, water_depth_m + 1.0)
    r_box_km = max(range_km * margin, range_km + 0.1)
    lines.append(f"0.0  {z_box_m:.2f}  {r_box_km:.4f}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---- .arr 文件解析 ----

def parse_arrivals(arr_path: Path, max_arrivals: int = 20) -> List[Dict]:
    """
    解析 BELLHOP 的 .arr 输出。

    Bellhop 生成的 .arr 文件可能是:
    1. ASCII 文本格式（当前我们这次实际运行得到的）
    2. 二进制格式（某些运行方式或版本）

    返回按延迟排序的到达结构列表 (保留幅值最大的 max_arrivals 条):
    [
      {
        "delay_s": float,
        "amplitude_real": float,
        "amplitude_imag": float,
        "amplitude_linear": float,
        "amplitude_db": float,
        "phase_rad": float,
        "source_angle_deg": float,
        "receiver_angle_deg": float,
        "num_top_bnc": int,
        "num_bot_bnc": int,
      },
      ...
    ]
    """
    if not arr_path.exists():
        raise FileNotFoundError(f".arr 文件未生成: {arr_path}")

    raw = arr_path.read_bytes()

    try:
        arrivals = _parse_arr_text(raw)
        return _postprocess_arrivals(arrivals, max_arrivals)
    except Exception:
        pass

    # 文本解析失败，尝试不同 endianness 的二进制解析
    for endian in ('<', '>'):
        try:
            arrivals = _parse_arr_raw(raw, endian)
            if arrivals:
                return _postprocess_arrivals(arrivals, max_arrivals)
        except (struct.error, IndexError, ValueError):
            continue

    raise RuntimeError(f"无法解析 .arr 文件 (不支持的格式): {arr_path}")


def _postprocess_arrivals(arrivals: List[Dict], max_arrivals: int) -> List[Dict]:
    """统一整理到达结构：按强度排序并归一化。"""
    if not arrivals:
        return []

    arrivals.sort(key=lambda a: a["amplitude_linear"], reverse=True)
    arrivals = arrivals[:max_arrivals]
    arrivals.sort(key=lambda a: a["delay_s"])

    if arrivals:
        max_amp = max(a["amplitude_linear"] for a in arrivals)
        if max_amp > 0:
            for a in arrivals:
                a["amplitude_normalized"] = a["amplitude_linear"] / max_amp

    return arrivals


def _parse_arr_text(data: bytes) -> List[Dict]:
    """
    解析 BELLHOP ASCII .arr 文件。

    实际输出格式为 'a' (magnitude + phase in degrees):
      pressure_magnitude  phase_deg  delay_s  src_angle_deg  rcv_angle_deg  top_bnc  bot_bnc

    示例行:
      1.56958940E-05  540.000000  0.708899081  0.00000000  -23.5368538  23.5415077  3  2
    """
    text = data.decode('utf-8', errors='ignore')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    arrivals = []
    for line in lines:
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            magnitude = abs(float(parts[0]))
            phase_deg = float(parts[1])
            delay = float(parts[2])
            delay_imag = float(parts[3])
            src_ang = float(parts[4])
            rcv_ang = float(parts[5])
            top_bnc = int(float(parts[6]))
            bot_bnc = int(float(parts[7]))
        except ValueError:
            continue

        phase_rad = np.radians(phase_deg)
        amp_real = magnitude * np.cos(phase_rad)
        amp_imag = magnitude * np.sin(phase_rad)
        amp_db = 20.0 * np.log10(magnitude) if magnitude > 1e-20 else -400.0

        arrivals.append({
            "delay_s": float(delay),
            "delay_imag_s": float(delay_imag),
            "amplitude_real": float(amp_real),
            "amplitude_imag": float(amp_imag),
            "amplitude_linear": float(magnitude),
            "amplitude_db": float(amp_db),
            "phase_rad": float(phase_rad),
            "source_angle_deg": float(src_ang),
            "receiver_angle_deg": float(rcv_ang),
            "num_top_bnc": int(top_bnc),
            "num_bot_bnc": int(bot_bnc),
        })

    return arrivals


def _parse_arr_raw(data: bytes, endian: str) -> List[Dict]:
    """尝试解析 .arr 二进制数据的内部实现。"""
    offset = 0
    # Header: freq (float), nsd (int), nrd (int), nr (int)
    head_fmt = endian + "fiii"
    head_size = struct.calcsize(head_fmt)
    freq, nsd, nrd, nr = struct.unpack_from(head_fmt, data, offset)
    offset += head_size

    if nsd <= 0 or nrd <= 0 or nr <= 0 or nsd > 1000 or nr > 100000:
        return []

    arrivals = []
    for _isd in range(nsd):
        for _ird in range(nrd):
            for _ir in range(nr):
                # 每条射线: A_real(f), A_imag(f), delay(f), SrcAngle(f), RcvrAngle(f),
                #           NumTopBnc(i), NumBotBnc(i) = 7 fields per arrival
                n_arrivals_raw = data[offset:offset + 4]
                if len(n_arrivals_raw) < 4:
                    return arrivals
                n_arrivals = struct.unpack(endian + "i", n_arrivals_raw)[0]
                offset += 4

                if n_arrivals <= 0 or n_arrivals > 10000:
                    continue

                arr_fmt = endian + "fffffii"
                arr_size = struct.calcsize(arr_fmt)
                for _ in range(n_arrivals):
                    if offset + arr_size > len(data):
                        return arrivals
                    (a_r, a_i, delay, src_ang, rcv_ang, top_bnc, bot_bnc) = \
                        struct.unpack_from(arr_fmt, data, offset)
                    offset += arr_size

                    amp_lin = np.sqrt(a_r ** 2 + a_i ** 2)
                    amp_db = 20.0 * np.log10(amp_lin) if amp_lin > 1e-20 else -400.0
                    phase = np.arctan2(a_i, a_r)

                    arrivals.append({
                        "delay_s": float(delay),
                        "amplitude_real": float(a_r),
                        "amplitude_imag": float(a_i),
                        "amplitude_linear": float(amp_lin),
                        "amplitude_db": float(amp_db),
                        "phase_rad": float(phase),
                        "source_angle_deg": float(src_ang),
                        "receiver_angle_deg": float(rcv_ang),
                        "num_top_bnc": int(top_bnc),
                        "num_bot_bnc": int(bot_bnc),
                    })

    return arrivals


# ---- CIR 构建与卷积 ----

def build_cir(
    arrivals: List[Dict],
    fs_hz: float,
    duration_s: float,
    fractional_delay: bool = True,
    use_normalized: bool = True,
    delay_reference_s: float = 0.0,
) -> np.ndarray:
    """
    从 BELLHOP 到达结构构建离散时间冲激响应。

    参数:
        arrivals: parse_arrivals() 的输出
        fs_hz: CIR 采样率 (通常与音频一致)
        duration_s: CIR 长度 (秒)，通常设为音频长度
        fractional_delay: True 使用带限插值，False 直接舍入到最近采样点
        use_normalized: True 使用单频内归一化幅度（适合单频 PulseCom）；
                        False 使用原始 BELLHOP 幅度（适合多频宽带叠加，保留频点间相对增益）

    返回:
        cir: 1D numpy array, dtype=float64
    """
    n_samples = int(round(duration_s * fs_hz))
    cir = np.zeros(n_samples, dtype=np.float64)

    for a in arrivals:
        delay = a["delay_s"] - delay_reference_s
        if use_normalized:
            amp = a.get("amplitude_normalized", a["amplitude_linear"])
        else:
            amp = a["amplitude_linear"]

        if fractional_delay:
            # Kaiser 窗 sinc 插值实现分数延迟
            _add_fractional_delay(cir, delay, amp, fs_hz)
        else:
            idx = int(round(delay * fs_hz))
            if 0 <= idx < n_samples:
                cir[idx] += amp

    return cir


def _add_fractional_delay(cir: np.ndarray, delay_s: float, amplitude: float, fs_hz: float):
    """
    使用加窗 sinc 插值在冲激响应中添加分数延迟。
    窗长 = 32 samples, Kaiser β=6.0
    """
    n_samples = len(cir)
    center = delay_s * fs_hz
    base_idx = int(np.floor(center))
    frac = center - base_idx

    half_len = 16
    t = (np.arange(-half_len, half_len + 1) + frac)
    sinc_vals = np.sinc(t)
    window = kaiser(2 * half_len + 1, beta=6.0)
    kernel = sinc_vals * window

    for k, w in zip(range(-half_len, half_len + 1), kernel):
        idx = base_idx + k
        if 0 <= idx < n_samples:
            cir[idx] += amplitude * w


def apply_channel(audio: np.ndarray, cir: np.ndarray) -> np.ndarray:
    """
    音频与 CIR 卷积，输出与输入等长。
    """
    if len(cir) == 0 or not np.any(np.abs(cir) > 0):
        raise ValueError("CIR is empty; refusing to return unprocessed audio")

    result = fftconvolve(audio, cir, mode="full")
    return result[:len(audio)]


def apply_frequency_dependent_channel(
    audio: np.ndarray,
    channels: List[Dict],
    fs_hz: float,
) -> np.ndarray:
    """Apply frequency-dependent CIRs by interpolating their FFT responses."""
    if not channels:
        raise ValueError("No frequency channels were provided")

    channels = sorted(channels, key=lambda item: item["freq_hz"])
    cir_len = max(len(item["cir"]) for item in channels)
    n_fft = next_fast_len(len(audio) + cir_len - 1)
    audio_spectrum = rfft(audio, n=n_fft)
    bin_freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs_hz)
    sample_freqs = np.asarray([item["freq_hz"] for item in channels], dtype=float)
    responses = np.vstack([rfft(item["cir"], n=n_fft) for item in channels])

    if len(channels) == 1:
        result = irfft(audio_spectrum * responses[0], n=n_fft)
        return result[:len(audio)]

    upper = np.searchsorted(sample_freqs, bin_freqs, side="right")
    upper = np.clip(upper, 1, len(sample_freqs) - 1)
    lower = upper - 1
    span = sample_freqs[upper] - sample_freqs[lower]
    weight = np.divide(
        bin_freqs - sample_freqs[lower], span,
        out=np.zeros_like(bin_freqs), where=span > 0,
    )
    weight[bin_freqs <= sample_freqs[0]] = 0.0
    weight[bin_freqs >= sample_freqs[-1]] = 1.0
    bins = np.arange(len(bin_freqs))
    response = responses[lower, bins] * (1.0 - weight) + responses[upper, bins] * weight

    result = irfft(audio_spectrum * response, n=n_fft)
    return result[:len(audio)]


# ---- 完整 BELLHOP 运行流程 ----

class BellhopRunner:
    """BELLHOP 流程编排器。"""

    def __init__(self, bellhop_exe: str, temp_dir: str = "temp_bellhop"):
        self.bellhop_exe = Path(bellhop_exe).resolve()
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        if not self.bellhop_exe.exists():
            raise FileNotFoundError(
                f"BELLHOP 可执行文件不存在: {self.bellhop_exe}\n"
                f"请确认路径或安装 BELLHOP 后修改 config.yaml → paths.bellhop_exe"
            )
        if not os.access(self.bellhop_exe, os.X_OK):
            raise PermissionError(
                f"BELLHOP 文件没有执行权限: {self.bellhop_exe}\n"
                f"请执行: chmod +x {self.bellhop_exe}"
            )

    def run(
        self,
        depths_m: np.ndarray,
        sound_speeds_mps: np.ndarray,
        freq_hz: float,
        source_depth_m: float,
        receiver_depth_m: float,
        range_km: float,
        water_depth_m: float,
        bottom: Optional[Dict] = None,
        title: str = "bellhop_run",
        num_beams="adaptive",
        ray_box_margin: float = 1.05,
        max_arrivals: int = 20,
        min_beams: int = 1000,
        beams_per_km: float = 100.0,
        max_beams: int = 8000,
    ) -> List[Dict]:
        """
        运行一次完整的 BELLHOP 计算并返回到达结构。

        步骤：生成 .env → 运行 bellhop.exe → 解析 .arr
        """
        if bottom is None:
            bottom = {
                "type": "silt",
                "sound_speed_mps": 1520.0,
                "density_gcc": 1.4,
                "attenuation_db_per_lambda": 3.0,
            }

        # 使用时间戳避免文件名冲突
        ts = int(time.time() * 1_000_000) % 100_000_000
        run_id = f"bh_{ts}"
        env_path = self.temp_dir / f"{run_id}.env"

        effective_num_beams = resolve_num_beams(
            num_beams, range_km, min_beams, beams_per_km, max_beams,
        )
        generate_env(
            env_path=env_path,
            title=title,
            freq_hz=freq_hz,
            depths_m=depths_m,
            sound_speeds_mps=sound_speeds_mps,
            water_depth_m=water_depth_m,
            source_depth_m=source_depth_m,
            receiver_depth_m=receiver_depth_m,
            range_km=range_km,
            bottom_type=bottom["type"],
            bottom_sound_speed_mps=bottom["sound_speed_mps"],
            bottom_density_gcc=bottom["density_gcc"],
            bottom_attenuation_db_per_lambda=bottom["attenuation_db_per_lambda"],
            num_beams=effective_num_beams,
            ray_box_margin=ray_box_margin,
        )

        try:
            result = subprocess.run(
                [str(self.bellhop_exe), env_path.stem],
                cwd=str(self.temp_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                stderr = result.stderr[:500] if result.stderr else ""
                raise RuntimeError(
                    f"BELLHOP 运行失败 (exit={result.returncode}): {stderr}"
                )
            arr_path = self.temp_dir / f"{run_id}.arr"
            if not arr_path.exists():
                run_files = [f.name for f in self.temp_dir.glob(f"{run_id}.*")]
                raise FileNotFoundError(
                    f"BELLHOP 未生成本次 .arr 文件: {arr_path.name}; "
                    f"本次临时文件: {run_files}"
                )
            return parse_arrivals(arr_path, max_arrivals=max_arrivals)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("BELLHOP 运行超时 (>60s)") from e
        finally:
            for f in self.temp_dir.glob(f"{run_id}.*"):
                try:
                    f.unlink()
                except OSError:
                    pass

    def run_and_build_cir(
        self,
        depths_m: np.ndarray,
        sound_speeds_mps: np.ndarray,
        freq_hz: float,
        source_depth_m: float,
        receiver_depth_m: float,
        range_km: float,
        water_depth_m: float,
        audio_fs_hz: float,
        audio_duration_s: float,
        bottom: Optional[Dict] = None,
        title: str = "bellhop_run",
        max_arrivals: int = 20,
        use_normalized: bool = True,
        num_beams="adaptive",
        ray_box_margin: float = 1.05,
        min_beams: int = 1000,
        beams_per_km: float = 100.0,
        max_beams: int = 8000,
    ) -> Dict:
        """
        运行 BELLHOP → 构建 CIR (一步到位)。
        """
        arrivals = self.run(
            depths_m=depths_m,
            sound_speeds_mps=sound_speeds_mps,
            freq_hz=freq_hz,
            source_depth_m=source_depth_m,
            receiver_depth_m=receiver_depth_m,
            range_km=range_km,
            water_depth_m=water_depth_m,
            bottom=bottom,
            title=title,
            num_beams=num_beams,
            ray_box_margin=ray_box_margin,
            max_arrivals=max_arrivals,
            min_beams=min_beams,
            beams_per_km=beams_per_km,
            max_beams=max_beams,
        )
        if not arrivals:
            raise NoArrivalsError("BELLHOP returned 0 arrivals")
        delay_reference_s = min(a["delay_s"] for a in arrivals)
        cir = build_cir(
            arrivals, audio_fs_hz, audio_duration_s,
            use_normalized=use_normalized,
            delay_reference_s=delay_reference_s,
        )
        return {
            "cir": cir,
            "arrivals": arrivals,
            "delay_reference_s": delay_reference_s,
        }

    def compute_broadband_cir(
        self,
        depths_m: np.ndarray,
        sound_speeds_mps: np.ndarray,
        freqs_hz: List[float],
        source_depth_m: float,
        receiver_depth_m: float,
        range_km: float,
        water_depth_m: float,
        audio_fs_hz: float,
        audio_duration_s: float,
        bottom: Optional[Dict] = None,
        max_arrivals: int = 20,
        num_beams="adaptive",
        ray_box_margin: float = 1.05,
        min_beams: int = 1000,
        beams_per_km: float = 100.0,
        max_beams: int = 8000,
    ) -> Dict:
        """
        对多个频率运行 BELLHOP，返回共享时延参考的频点 CIR。

        各频点保留原始 BELLHOP 幅度；调用方在频域插值响应，不再把不同
        频点的 CIR 直接相加或做信道内归一化。
        """
        frequency_results = []
        completed_runs = 0
        for freq in freqs_hz:
            try:
                arrivals = self.run(
                    depths_m=depths_m, sound_speeds_mps=sound_speeds_mps,
                    freq_hz=freq, source_depth_m=source_depth_m,
                    receiver_depth_m=receiver_depth_m, range_km=range_km,
                    water_depth_m=water_depth_m, bottom=bottom,
                    title=f"bb_{freq:.0f}Hz", num_beams=num_beams,
                    ray_box_margin=ray_box_margin, max_arrivals=max_arrivals,
                    min_beams=min_beams, beams_per_km=beams_per_km,
                    max_beams=max_beams,
                )
                completed_runs += 1
                if arrivals:
                    frequency_results.append({"freq_hz": float(freq), "arrivals": arrivals})
                else:
                    print(f"  [WARN] BELLHOP @ {freq:.1f}Hz returned 0 arrivals")
            except Exception as e:
                print(f"  [WARN] BELLHOP @ {freq:.1f}Hz 失败: {e}")

        if not frequency_results:
            if completed_runs:
                raise NoArrivalsError("所有频点均未产生有效到达")
            raise RuntimeError("所有频点的 BELLHOP 计算均失败")

        delay_reference_s = min(
            arrival["delay_s"]
            for item in frequency_results
            for arrival in item["arrivals"]
        )
        channels = []
        for item in frequency_results:
            cir = build_cir(
                item["arrivals"], audio_fs_hz, audio_duration_s,
                use_normalized=False, delay_reference_s=delay_reference_s,
            )
            channels.append({**item, "cir": cir})

        return {
            "channels": channels,
            "delay_reference_s": delay_reference_s,
            "successful_freqs_hz": [item["freq_hz"] for item in channels],
        }
