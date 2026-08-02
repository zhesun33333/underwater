"""
BELLHOP 调用封装
.env 生成 → bellhop.exe 执行 → .arr 解析 → CIR 构建 → 卷积
"""
import hashlib
import os
import struct
import subprocess
import uuid
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


class InvalidEnvironmentError(RuntimeError):
    """BELLHOP rejected a generated environment file."""


class BellhopExecutionError(RuntimeError):
    """BELLHOP itself could not run or produced an unusable output format."""


def make_channel_rng(random_seed: int, *identity_parts) -> np.random.Generator:
    """Create a stable RNG for one sample/channel, independent of resume order."""
    identity = "\0".join([str(random_seed), *(str(part) for part in identity_parts)])
    digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, byteorder="little"))


def _finite_scalar(name: str, value: float, *, positive: bool = False) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def _prepare_ssp_profile(
    depths_m: np.ndarray,
    sound_speeds_mps: np.ndarray,
    water_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate an SSP and terminate it at exactly the formatted bottom depth."""
    water_depth = _finite_scalar("water_depth_m", water_depth_m, positive=True)
    depths = np.asarray(depths_m, dtype=np.float64).reshape(-1)
    speeds = np.asarray(sound_speeds_mps, dtype=np.float64).reshape(-1)
    if depths.size != speeds.size:
        raise ValueError(
            f"SSP depth/speed length mismatch: {depths.size} != {speeds.size}"
        )

    finite = np.isfinite(depths) & np.isfinite(speeds)
    depths = depths[finite]
    speeds = speeds[finite]
    if depths.size < 2:
        raise ValueError("SSP must contain at least two finite points")
    if np.any(speeds <= 0.0):
        raise ValueError("SSP sound speeds must be positive")

    order = np.argsort(depths, kind="stable")
    depths = depths[order]
    speeds = speeds[order]
    unique = np.concatenate(([True], np.diff(depths) > 0.0))
    depths = depths[unique]
    speeds = speeds[unique]
    if depths.size < 2 or water_depth > depths[-1] + 1e-6:
        raise ValueError(
            f"SSP max depth {depths[-1]:.6f}m does not cover water depth "
            f"{water_depth:.6f}m"
        )

    bottom_speed = float(np.interp(water_depth, depths, speeds))
    above_bottom = depths < water_depth
    trimmed_depths = np.append(depths[above_bottom], water_depth)
    trimmed_speeds = np.append(speeds[above_bottom], bottom_speed)
    if trimmed_depths.size < 2 or np.any(np.diff(trimmed_depths) <= 0.0):
        raise ValueError("SSP must have at least two strictly increasing depths")
    return trimmed_depths, trimmed_speeds, water_depth


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
    freq_hz = _finite_scalar("freq_hz", freq_hz, positive=True)
    source_depth_m = _finite_scalar("source_depth_m", source_depth_m, positive=True)
    receiver_depth_m = _finite_scalar("receiver_depth_m", receiver_depth_m, positive=True)
    range_km = _finite_scalar("range_km", range_km, positive=True)
    bottom_sound_speed_mps = _finite_scalar(
        "bottom_sound_speed_mps", bottom_sound_speed_mps, positive=True,
    )
    bottom_density_gcc = _finite_scalar(
        "bottom_density_gcc", bottom_density_gcc, positive=True,
    )
    bottom_attenuation_db_per_lambda = _finite_scalar(
        "bottom_attenuation_db_per_lambda", bottom_attenuation_db_per_lambda,
    )
    z_trim, c_trim, water_depth_m = _prepare_ssp_profile(
        depths_m, sound_speeds_mps, water_depth_m,
    )
    if source_depth_m >= water_depth_m or receiver_depth_m >= water_depth_m:
        raise ValueError(
            "source and receiver depths must be strictly shallower than the bottom"
        )

    lines = []
    lines.append(f"'{title}'")
    lines.append(f"{freq_hz:.8E}")
    lines.append("1")
    lines.append("'CVF'")

    # Bellhop 读取顺序：NPts, Sigma, BottomDepth
    lines.append(f"{len(z_trim)}  1.00000000E+00  {water_depth_m:.8E}")

    # SSP 记录：每行 depth sound_speed /
    for z, c in zip(z_trim, c_trim):
        lines.append(f"{z:.8E}  {c:.8E}  /")

    # 底边界条件：BC + Sigma
    lines.append("'A'  0.0")

    # 底部半空间参数：depth, cp, cs, rho, alphaP, alphaS
    lines.append(f"{water_depth_m:.8E}  {bottom_sound_speed_mps:.8E}  0.0  "
                 f"{bottom_density_gcc:.8E}  {bottom_attenuation_db_per_lambda:.8E}  0.0 /")

    # 源/接收/距离
    lines.append("1")
    lines.append(f"{source_depth_m:.8E} /")
    lines.append("1")
    lines.append(f"{receiver_depth_m:.8E} /")
    lines.append("1")
    lines.append(f"{range_km:.8E} /")

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

    # ASCII files may legitimately contain zero arrivals. Only use the text
    # parser when the payload itself is text; otherwise try the binary layouts.
    try:
        raw.decode("ascii")
        is_text = b"\x00" not in raw
    except UnicodeDecodeError:
        is_text = False
    if is_text:
        arrivals = _parse_arr_text(raw)
        return _postprocess_arrivals(arrivals, max_arrivals)

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
    max_arrivals = int(max_arrivals)
    if max_arrivals <= 0:
        raise ValueError(f"max_arrivals must be positive, got {max_arrivals}")
    arrivals = [
        arrival for arrival in arrivals
        if np.isfinite(arrival.get("delay_s", np.nan))
        and arrival.get("delay_s", -1.0) >= 0.0
        and np.isfinite(arrival.get("amplitude_linear", np.nan))
        and arrival.get("amplitude_linear", 0.0) > 0.0
    ]
    if not arrivals:
        return []

    first_arrival = min(arrivals, key=lambda a: a["delay_s"])
    arrivals.sort(key=lambda a: a["amplitude_linear"], reverse=True)
    arrivals = arrivals[:max_arrivals]
    if not any(arrival is first_arrival for arrival in arrivals):
        arrivals[-1] = first_arrival
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
    fs_hz = _finite_scalar("fs_hz", fs_hz, positive=True)
    duration_s = _finite_scalar("duration_s", duration_s, positive=True)
    delay_reference_s = _finite_scalar("delay_reference_s", delay_reference_s)
    n_samples = max(1, int(round(duration_s * fs_hz)))
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
    # At output index base_idx + k, the ideal sampled impulse is sinc(k-frac).
    t = np.arange(-half_len, half_len + 1) - frac
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
    audio = np.asarray(audio, dtype=np.float64)
    cir = np.asarray(cir, dtype=np.float64)
    if audio.ndim != 1 or audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("Audio must be a non-empty finite 1D array")
    if cir.ndim != 1 or not np.all(np.isfinite(cir)):
        raise ValueError("CIR must be a finite 1D array")
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
    audio = np.asarray(audio, dtype=np.float64)
    fs_hz = _finite_scalar("fs_hz", fs_hz, positive=True)
    if audio.ndim != 1 or audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("Audio must be a non-empty finite 1D array")
    if not channels:
        raise ValueError("No frequency channels were provided")

    channels = sorted(channels, key=lambda item: item["freq_hz"])
    for item in channels:
        _finite_scalar("channel frequency", item["freq_hz"], positive=True)
        cir = np.asarray(item["cir"], dtype=np.float64)
        if cir.ndim != 1 or cir.size == 0 or not np.all(np.isfinite(cir)):
            raise ValueError("Every frequency channel must contain a finite 1D CIR")
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
        bottom = {
            "type": "silt",
            "sound_speed_mps": 1520.0,
            "density_gcc": 1.4,
            "attenuation_db_per_lambda": 3.0,
            **(bottom or {}),
        }

        # UUID keeps concurrent processes from sharing BELLHOP scratch files.
        run_id = f"bh_{uuid.uuid4().hex}"
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
                details = (result.stderr or result.stdout or "")[:1000]
                if "Bad real number" in details or "End of file" in details:
                    raise InvalidEnvironmentError(
                        "BELLHOP rejected the generated .env "
                        f"(freq={freq_hz:.1f}Hz, water={water_depth_m:.6f}m, "
                        f"range={range_km:.6f}km, ssp_points={len(depths_m)}): {details}"
                    )
                raise BellhopExecutionError(
                    f"BELLHOP 运行失败 (exit={result.returncode}): {details}"
                )
            arr_path = self.temp_dir / f"{run_id}.arr"
            if not arr_path.exists():
                run_files = [f.name for f in self.temp_dir.glob(f"{run_id}.*")]
                raise BellhopExecutionError(
                    f"BELLHOP 未生成本次 .arr 文件: {arr_path.name}; "
                    f"本次临时文件: {run_files}"
                )
            try:
                return parse_arrivals(arr_path, max_arrivals=max_arrivals)
            except Exception as e:
                raise BellhopExecutionError(f"BELLHOP .arr 解析失败: {e}") from e
        except OSError as e:
            raise BellhopExecutionError(f"无法启动 BELLHOP: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise BellhopExecutionError("BELLHOP 运行超时 (>60s)") from e
        finally:
            for f in self.temp_dir.glob(f"{run_id}.*"):
                try:
                    f.unlink()
                except OSError:
                    pass

    def run_with_beam_retry(
        self,
        *,
        min_beams: int = 1000,
        beams_per_km: float = 100.0,
        max_beams: int = 8000,
        **run_kwargs,
    ) -> Dict:
        """Run once, then recheck a zero-arrival result with more beams."""
        range_km = float(run_kwargs["range_km"])
        beam_setting = run_kwargs.get("num_beams", "adaptive")
        initial_beams = resolve_num_beams(
            beam_setting, range_km, min_beams, beams_per_km, max_beams,
        )
        arrivals = self.run(
            **run_kwargs,
            min_beams=min_beams,
            beams_per_km=beams_per_km,
            max_beams=max_beams,
        )
        effective_beams = initial_beams
        retried = False

        # A user-selected fixed/automatic count is respected. Adaptive mode gets
        # one denser check so sparse ray sampling is not mislabeled as no path.
        if (not arrivals and isinstance(beam_setting, str)
                and beam_setting.strip().lower() == "adaptive"
                and initial_beams < max_beams):
            effective_beams = min(max_beams, max(4000, initial_beams * 2))
            retried = True
            arrivals = self.run(
                **{**run_kwargs, "num_beams": effective_beams},
                min_beams=min_beams,
                beams_per_km=beams_per_km,
                max_beams=max_beams,
            )

        return {
            "arrivals": arrivals,
            "num_beams_effective": effective_beams,
            "beam_retry": retried,
        }

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
        run_result = self.run_with_beam_retry(
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
        arrivals = run_result["arrivals"]
        if not arrivals:
            raise NoArrivalsError("BELLHOP returned 0 arrivals after beam check")
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
            "num_beams_effective": run_result["num_beams_effective"],
            "beam_retry": run_result["beam_retry"],
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
                run_result = self.run_with_beam_retry(
                    depths_m=depths_m, sound_speeds_mps=sound_speeds_mps,
                    freq_hz=freq, source_depth_m=source_depth_m,
                    receiver_depth_m=receiver_depth_m, range_km=range_km,
                    water_depth_m=water_depth_m, bottom=bottom,
                    title=f"bb_{freq:.0f}Hz", num_beams=num_beams,
                    ray_box_margin=ray_box_margin, max_arrivals=max_arrivals,
                    min_beams=min_beams, beams_per_km=beams_per_km,
                    max_beams=max_beams,
                )
                arrivals = run_result["arrivals"]
                completed_runs += 1
                if arrivals:
                    frequency_results.append({
                        "freq_hz": float(freq),
                        "arrivals": arrivals,
                        "num_beams_effective": run_result["num_beams_effective"],
                        "beam_retry": run_result["beam_retry"],
                    })
                else:
                    print(f"  [WARN] BELLHOP @ {freq:.1f}Hz returned 0 arrivals after beam check")
            except InvalidEnvironmentError as e:
                print(f"  [WARN] BELLHOP environment invalid @ {freq:.1f}Hz: {e}")
                raise
            except BellhopExecutionError:
                raise
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
            "num_beams_effective": max(
                item["num_beams_effective"] for item in channels
            ),
            "beam_retry_frequencies_hz": [
                item["freq_hz"] for item in channels if item["beam_retry"]
            ],
        }
