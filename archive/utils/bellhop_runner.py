"""
BELLHOP 调用封装
.env 生成 → bellhop.exe 执行 → .arr 解析 → CIR 构建 → 卷积
"""
import struct
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import fftconvolve

try:
    from scipy.signal import kaiser
except ImportError:
    from scipy.signal.windows import get_window

    def kaiser(M, beta):
        return get_window(("kaiser", beta), M)


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

    # Bellhop 读取顺序：NPts, Sigma, BottomDepth
    lines.append(f"{len(depths_m)}  1.0  {water_depth_m:.1f}")

    # SSP 记录：每行 depth sound_speed /
    for z, c in zip(depths_m, sound_speeds_mps):
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
    nbeams = max(21, int(range_km * 5))
    lines.append("'A'")
    lines.append(f"{nbeams}")
    lines.append("-90 90 /")
    lines.append("0.0  1000.0  10.0")

    env_path.write_text("\n".join(lines), encoding="utf-8")


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
            src_ang = float(parts[3])
            rcv_ang = float(parts[4])
            top_bnc = int(float(parts[5]))
            bot_bnc = int(float(parts[6]))
        except ValueError:
            continue

        phase_rad = np.radians(phase_deg)
        amp_real = magnitude * np.cos(phase_rad)
        amp_imag = magnitude * np.sin(phase_rad)
        amp_db = 20.0 * np.log10(magnitude) if magnitude > 1e-20 else -400.0

        arrivals.append({
            "delay_s": float(delay),
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
        delay = a["delay_s"]
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
    if len(cir) == 0 or np.all(cir == 0):
        return audio.copy()

    result = fftconvolve(audio, cir, mode="full")
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
        )

        # 运行 BELLHOP (在 temp_dir 下执行)
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
        except subprocess.TimeoutExpired:
            raise RuntimeError("BELLHOP 运行超时 (>60s)")

        # 查找输出文件
        arr_path = self.temp_dir / f"{run_id}.arr"
        if not arr_path.exists():
            # BELLHOP 可能输出到不同路径
            candidates = list(self.temp_dir.glob("*.arr"))
            if candidates:
                arr_path = max(candidates, key=lambda p: p.stat().st_mtime)
            else:
                # 检查是否有其他输出文件
                all_files = list(self.temp_dir.glob("*"))
                raise FileNotFoundError(
                    f"BELLHOP 未生成 .arr 文件。临时目录内容: "
                    f"{[f.name for f in all_files]}"
                )

        arrivals = parse_arrivals(arr_path)

        # 清理临时文件
        for f in self.temp_dir.glob(f"{run_id}.*"):
            try:
                f.unlink()
            except OSError:
                pass

        return arrivals

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
    ) -> np.ndarray:
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
        )
        # 限制到达数量
        arrivals = arrivals[:max_arrivals]

        return build_cir(arrivals, audio_fs_hz, audio_duration_s, use_normalized=use_normalized)

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
    ) -> np.ndarray:
        """
        对多个频率运行 BELLHOP，合并为宽带 CIR (用于舰船噪声等宽带信号)。

        各频点 CIR 使用原始 BELLHOP 幅度（不做单频内归一化），保留频点间
        的相对增益差异。等权重叠加后进行单次全局归一化。
        """
        cirs = []
        for freq in freqs_hz:
            try:
                cir = self.run_and_build_cir(
                    depths_m=depths_m,
                    sound_speeds_mps=sound_speeds_mps,
                    freq_hz=freq,
                    source_depth_m=source_depth_m,
                    receiver_depth_m=receiver_depth_m,
                    range_km=range_km,
                    water_depth_m=water_depth_m,
                    audio_fs_hz=audio_fs_hz,
                    audio_duration_s=audio_duration_s,
                    bottom=bottom,
                    title=f"bb_{freq:.0f}Hz",
                    use_normalized=False,
                )
                cirs.append(cir)
            except Exception as e:
                print(f"  [WARN] BELLHOP @ {freq:.1f}Hz 失败: {e}")

        if not cirs:
            raise RuntimeError("所有频点的 BELLHOP 计算均失败")

        # 等权重叠加（各频点 CIR 使用原始 BELLHOP 幅度，频点间增益差异已保留）
        combined = np.sum(cirs, axis=0)
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined /= max_val
        return combined
