"""
舰船辐射噪声数据合成管线 — 第一步：BELLHOP 信道卷积

与 PulseCom 管线的主要区别:
  1. 目录结构: <class_name>/json/*.json + <class_name>/wav/*.wav
     而非 PulseCom 的 jsonc/<TYPE>/*.jsonc + wav_by_type/<TYPE>/*.wav
  2. 无单一中心频率: 舰船噪声为宽带信号，使用多点频率计算宽带 CIR
  3. water_depth_m 常为空列表，需更激进地回退
  4. JSON 为标准格式，非 JSONC

用法:
  python pipeline_ship_step1_channel.py [--config config_ship.yaml]
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.json_parser import load_jsonc, get_geometry
from utils.ssp_sampler import SSPSampler
from utils.bellhop_runner import (
    BellhopExecutionError, BellhopRunner, InvalidEnvironmentError,
    NoArrivalsError, apply_frequency_dependent_channel, make_channel_rng,
)

warnings.filterwarnings("ignore")

CHANNEL_MODEL_VERSION = "2.2"


# ---- 配置加载 ----

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_env_vars(cfg)
    return cfg


def _resolve_env_vars(obj):
    import os
    import re
    if isinstance(obj, str):
        def replacer(m):
            return os.environ.get(m.group(1), m.group(0))
        return re.compile(r'\$\{(\w+)\}').sub(replacer, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


# ---- 舰船数据专用: 目录扫描 ----

def find_ship_json_files(dataset_root: Path, classes: List[str], json_subdir: str, json_ext: str) -> List[Tuple[Path, str]]:
    """
    扫描舰船数据集的所有 JSON 文件。

    舰船目录结构:
      05_ship_radiated_noise/
        cargo_ship/
          json/*.json
          wav/*.wav
        cruise_ship/
          json/*.json
          wav/*.wav
        ...

    返回 [(json_path, class_name), ...]
    """
    all_files = []
    for class_name in classes:
        json_dir = dataset_root / class_name / json_subdir
        if not json_dir.exists():
            print(f"  [WARN] JSON 目录不存在: {json_dir}")
            continue
        files = sorted(json_dir.glob(f"*{json_ext}"))
        for jf in files:
            if not jf.name.startswith("manifest"):
                all_files.append((jf, class_name))
    return all_files


def find_ship_wav(dataset_root: Path, class_name: str, wav_subdir: str, sample_id: str) -> Optional[Path]:
    """在舰船数据集中查找 WAV 文件。"""
    wav_path = dataset_root / class_name / wav_subdir / f"{sample_id}.wav"
    if wav_path.exists():
        return wav_path
    # 回退: 递归搜索
    candidates = list((dataset_root / class_name).rglob(f"{sample_id}.wav"))
    return candidates[0] if candidates else None


# ---- 音频 I/O ----

def read_audio(wav_path: Path) -> Tuple[int, np.ndarray]:
    sr, data = wavfile.read(wav_path)
    if data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.signedinteger):
        scale = float(max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max))
        data = data.astype(np.float64) / scale
    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float64)
    else:
        raise ValueError(f"不支持的 WAV 数据类型: {data.dtype}")
    if data.ndim > 1:
        data = data[:, 0]
    if data.size == 0:
        raise ValueError(f"WAV 为空: {wav_path}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"WAV 包含 NaN/Inf: {wav_path}")
    if not np.any(np.abs(data) > 0.0):
        raise ValueError(f"WAV 全为零: {wav_path}")
    return int(sr), data


def write_audio(wav_path: Path, sr: int, data: np.ndarray, bits: int = 32):
    if not np.all(np.isfinite(data)):
        raise ValueError("拒绝写入包含 NaN/Inf 的音频")
    data = np.clip(data, -1.0, 1.0)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if bits == 32:
        int_data = (data * 2147483647.0).astype(np.int32)
    elif bits == 16:
        int_data = (data * 32767.0).astype(np.int16)
    else:
        raise ValueError(f"不支持的位深: {bits}")
    wavfile.write(wav_path, sr, int_data)


# ---- 源端线谱 SNR 读取 ----

def compute_post_channel_snr_ship(meta: dict) -> float:
    """
    返回源信号元数据中的线谱/连续谱 SNR。

    为兼容既有数据格式，结果仍写入历史字段名 snr_db；它实际是
    source/pre-channel line-spectrum SNR，并未从频率依赖信道处理后的
    WAV 中重新估计，因此不能解释为 received/post-channel SNR。
    """
    orig_snr = meta.get("snr_after_mix_db")
    try:
        orig_snr = float(orig_snr)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(orig_snr):
        return 0.0
    return round(orig_snr, 2)


# ---- 核心处理 ----

def process_single_ship(
    json_path: Path,
    class_name: str,
    dataset_root: Path,
    dataset_cfg: dict,
    channel_cfg: dict,
    ssp_sampler: SSPSampler,
    bellhop: BellhopRunner,
    output_root: Path,
    random_seed: int,
    skip_existing: bool = True,
) -> dict:
    """处理单条舰船辐射噪声: 多频 BELLHOP → 宽带 CIR → 卷积 → 保存。"""
    try:
        meta = load_jsonc(json_path)
        if not isinstance(meta, dict):
            raise TypeError("JSON 顶层必须是对象")
    except Exception as e:
        print(f"  [SKIP] JSON 解析失败: {json_path} — {e}")
        return {"status": "failed", "reason": "json_parse"}

    sample_id = meta.get("id", json_path.stem)
    wav_subdir = dataset_cfg["wav_subdir"]

    # 查找 WAV
    wav_path = find_ship_wav(dataset_root, class_name, wav_subdir, sample_id)
    if wav_path is None:
        print(f"  [SKIP] WAV 不存在: {sample_id}")
        return {"status": "failed", "reason": "wav_missing"}

    # 提取参数
    geo = get_geometry(meta)
    n_channels = int(channel_cfg.get("num_channels_per_audio", 2))
    if n_channels <= 0:
        return {"id": sample_id, "status": "failed", "reason": "invalid_channel_count"}
    use_jitter = channel_cfg.get("use_json_geometry", True)
    jitter = float(channel_cfg.get("geometry_jitter_pct", 0.1)) if use_jitter else None
    if jitter is not None and (not np.isfinite(jitter) or not 0.0 <= jitter < 1.0):
        return {"id": sample_id, "status": "failed", "reason": "invalid_geometry_jitter"}

    results = []
    for ch_idx in range(n_channels):
        channel_rng = make_channel_rng(
            random_seed, "ship", class_name, sample_id, ch_idx,
        )
        try:
            result = _apply_one_ship_channel(
                meta=meta,
                wav_path=wav_path,
                sample_id=sample_id,
                class_name=class_name,
                ch_idx=ch_idx,
                dataset_cfg=dataset_cfg,
                channel_cfg=channel_cfg,
                ssp_sampler=ssp_sampler,
                bellhop=bellhop,
                output_root=output_root,
                geo=geo,
                rng=channel_rng,
                jitter=jitter,
                skip_existing=skip_existing,
            )
            results.append(result)
        except NoArrivalsError as e:
            print(f"  [SKIP] 0 到达: {sample_id}_ch{ch_idx} — {e}")
            results.append({"status": "no_arrivals", "reason": str(e)})
        except (BellhopExecutionError, InvalidEnvironmentError):
            raise
        except Exception as e:
            print(f"  [ERR] {sample_id}_ch{ch_idx}: {e}")
            results.append({"status": "failed", "reason": str(e)})

    statuses = [result["status"] for result in results]
    usable = sum(status in ("success", "existing") for status in statuses)
    if statuses and all(status == "existing" for status in statuses):
        status = "skipped"
    elif usable == len(statuses):
        status = "success"
    elif usable > 0:
        status = "partial"
    elif statuses and all(status == "no_arrivals" for status in statuses):
        status = "no_arrivals"
    else:
        status = "failed"
    return {"id": sample_id, "status": status, "channel_statuses": statuses}


def _apply_one_ship_channel(
    meta: dict,
    wav_path: Path,
    sample_id: str,
    class_name: str,
    ch_idx: int,
    dataset_cfg: dict,
    channel_cfg: dict,
    ssp_sampler: SSPSampler,
    bellhop: BellhopRunner,
    output_root: Path,
    geo: dict,
    rng: np.random.Generator,
    jitter: Optional[float],
    skip_existing: bool,
) -> Optional[dict]:
    """对单条舰船噪声应用一个信道。"""

    new_id = f"{sample_id}_ch{ch_idx}"

    # 输出路径: 保持 <class_name>/json/ 和 <class_name>/wav/ 结构
    output_wav_dir = output_root / dataset_cfg["name"] / class_name / dataset_cfg["wav_subdir"]
    output_json_dir = output_root / dataset_cfg["name"] / class_name / dataset_cfg["json_subdir"]
    output_wav_path = output_wav_dir / f"{new_id}.wav"
    output_json_path = output_json_dir / f"{new_id}{dataset_cfg['json_ext']}"

    configured_freqs = channel_cfg.get("frequency", {}).get(
        "ship_freqs_hz", [50, 100, 200, 400, 800, 1600],
    )
    try:
        configured_freqs = sorted({float(freq) for freq in configured_freqs})
    except (TypeError, ValueError):
        raise ValueError(f"ship_freqs_hz 配置无效: {configured_freqs!r}")

    if skip_existing and output_wav_path.exists() and output_json_path.exists():
        try:
            existing_meta = load_jsonc(output_json_path)
            existing_freqs = (existing_meta.get("bellhop_env") or {}).get("freqs_hz")
            frequencies_match = (
                existing_freqs is not None
                and len(existing_freqs) == len(configured_freqs)
                and np.allclose(existing_freqs, configured_freqs, rtol=0.0, atol=0.01)
            )
            if (existing_meta.get("channel_model_version") == CHANNEL_MODEL_VERSION
                    and frequencies_match):
                return {"id": new_id, "status": "existing"}
        except Exception:
            pass

    # --- 几何参数 ---
    tx_depth = geo.get("tx_depth_m")
    rx_depth = geo.get("rx_depth_m")
    range_m = geo.get("range_m")
    water_depth = geo.get("water_depth_m")

    # 抖动
    if jitter and jitter > 0:
        if tx_depth is not None:
            tx_depth *= (1.0 + rng.uniform(-jitter, jitter))
        if rx_depth is not None:
            rx_depth *= (1.0 + rng.uniform(-jitter, jitter))
        if range_m is not None:
            range_m *= (1.0 + rng.uniform(-jitter, jitter))

    range_km = (range_m / 1000.0 if range_m is not None
                else rng.uniform(*channel_cfg.get("range_km_range", [10.0, 80.0])))

    # --- SSP 采样 ---
    min_ssp_depth = max(
        20.0, (tx_depth or 20) + 10, (rx_depth or 20) + 10,
        water_depth or 0.0,
    )
    z, c, ssp_info = ssp_sampler.sample_matching_depth(min_depth_m=min_ssp_depth, rng=rng)

    # 水深: ship 数据的 water_depth_m 常为 []，使用 SSP 最大深度
    if water_depth is None:
        water_depth = max(z)
        wd_range = channel_cfg.get("water_depth_m_range")
        if wd_range:
            water_depth = min(water_depth, rng.uniform(*wd_range))

    if water_depth <= 2.0:
        return {"id": new_id, "status": "failed", "reason": "invalid_water_depth"}

    # 检查 SSP 是否覆盖到水底: max(z) < water_depth 意味着 SSP 太浅，
    # BELLHOP 会自行外推深层声速，结果不可靠，直接跳过
    ssp_max_depth = float(max(z))
    if ssp_max_depth < water_depth:
        print(f"  [SKIP] SSP 剖面太浅 (max_ssp={ssp_max_depth:.1f}m) 无法覆盖水深 "
              f"({water_depth:.1f}m): {sample_id}_ch{ch_idx}")
        return {"id": new_id, "status": "failed", "reason": "ssp_too_shallow"}

    tx_depth = tx_depth if tx_depth else rng.uniform(5.0, water_depth * 0.5)
    rx_depth = rx_depth if rx_depth else rng.uniform(5.0, water_depth * 0.5)

    tx_depth = max(1.0, min(tx_depth, water_depth - 1.0))
    rx_depth = max(1.0, min(rx_depth, water_depth - 1.0))
    range_km = max(0.01, range_km)
    beam_setting = channel_cfg.get("num_beams", "adaptive")

    # --- 读取音频 ---
    fs, audio = read_audio(wav_path)
    audio_duration_s = len(audio) / fs
    declared_duration = meta.get("audio_duration_s")
    try:
        duration_mismatch = abs(float(declared_duration) - audio_duration_s) > (1.0 / fs)
    except (TypeError, ValueError):
        duration_mismatch = declared_duration is not None
    if duration_mismatch:
        print(f"  [WARN] 时长不匹配，采用 WAV: wav={audio_duration_s:.6f}s, "
              f"meta={declared_duration}")
    declared_fs = meta.get("model_fs_hz")
    try:
        declared_fs = int(float(declared_fs))
    except (TypeError, ValueError, OverflowError):
        declared_fs = None
    if declared_fs is not None and fs != declared_fs:
        print(f"  [WARN] 采样率不匹配，采用 WAV: wav={fs}, meta={declared_fs}")

    # --- 宽带 BELLHOP ---
    bottom = channel_cfg.get("bottom", {})
    ship_freqs = configured_freqs
    if (not ship_freqs or any(not np.isfinite(freq) or freq <= 0.0
                              or freq > fs / 2.0 for freq in ship_freqs)):
        raise ValueError(f"ship_freqs_hz 必须位于 (0, {fs / 2.0}] Hz: {ship_freqs!r}")

    broadband = bellhop.compute_broadband_cir(
        depths_m=z, sound_speeds_mps=c,
        freqs_hz=ship_freqs,
        source_depth_m=tx_depth, receiver_depth_m=rx_depth,
        range_km=range_km, water_depth_m=water_depth,
        audio_fs_hz=fs, audio_duration_s=audio_duration_s,
        bottom=bottom,
        max_arrivals=int(channel_cfg.get("max_arrivals", 20)),
        num_beams=beam_setting,
        ray_box_margin=float(channel_cfg.get("ray_box_margin", 1.05)),
        min_beams=int(channel_cfg.get("min_beams", 1000)),
        beams_per_km=float(channel_cfg.get("beams_per_km", 100.0)),
        max_beams=int(channel_cfg.get("max_beams", 8000)),
    )

    # --- 卷积 ---
    convolved = apply_frequency_dependent_channel(audio, broadband["channels"], fs)

    # --- 归一化前宽带 RMS 增益 ---
    # 兼容既有数据格式，结果仍写入历史字段名 tl_db；实际定义为
    # 20*log10(RMS(channel_output)/RMS(source))。数值越大（越接近 0）
    # 表示增益越高、衰减越弱，不是通常取正值的 transmission loss。
    rms_orig = np.sqrt(max(1e-20, np.mean(audio ** 2)))
    rms_conv = np.sqrt(max(1e-20, np.mean(convolved ** 2)))
    tl_est = 20.0 * np.log10(rms_conv / rms_orig)

    # --- 输出峰值归一化 ---
    # CIR 多频叠加后 peak 归一化无法保证卷积输出音量一致；
    # 此处对卷积结果做 peak 归一化，确保所有样本输出电平统一，
    # 同时保留信道的频率选择性（频谱形状不变、SNR 不变）。
    peak = float(np.max(np.abs(convolved)))
    target_peak = float(channel_cfg.get("output_norm_peak", 0.95))
    if not np.isfinite(target_peak) or not 0.0 < target_peak <= 1.0:
        raise ValueError(f"output_norm_peak 必须在 (0, 1]，实际为 {target_peak!r}")
    if peak > 1e-10:
        convolved *= (target_peak / peak)
        gain_db = round(20.0 * float(np.log10(target_peak / peak)), 2)
    else:
        gain_db = 0.0

    # --- 源端线谱 SNR（不做信道后重估） ---
    snr_post = compute_post_channel_snr_ship(meta)

    # --- 更新元数据 ---
    new_meta = dict(meta)
    new_meta["channel_model_version"] = CHANNEL_MODEL_VERSION
    new_meta["id"] = new_id
    new_meta["wav_path"] = str(output_wav_path.relative_to(output_root))
    new_meta["audio_duration_s"] = audio_duration_s

    new_meta["bellhop_env"] = {
        "ssp": {
            "depths_m": z.tolist(),
            "sound_speeds_mps": c.tolist(),
            "profile_index": ssp_info["profile_index"],
            "source_file": ssp_info.get("nc_file", str(ssp_sampler.nc_path)),
        },
        "water_depth_m": round(water_depth, 2),
        "source_depth_m": round(tx_depth, 2),
        "receiver_depth_m": round(rx_depth, 2),
        "range_km": round(range_km, 4),
        "range_m": round(range_km * 1000, 2),
        "freqs_hz": ship_freqs,
        "mode": "broadband",
        "num_beams_setting": beam_setting,
        "num_beams_effective": broadband["num_beams_effective"],
        "beam_retry_frequencies_hz": broadband["beam_retry_frequencies_hz"],
        "max_arrivals": int(channel_cfg.get("max_arrivals", 20)),
        "ray_box_margin": float(channel_cfg.get("ray_box_margin", 1.05)),
        "ray_box_range_km": round(max(
            range_km * max(1.01, float(channel_cfg.get("ray_box_margin", 1.05))),
            range_km + 0.1,
        ), 4),
        "ray_box_depth_m": round(max(
            water_depth * max(1.01, float(channel_cfg.get("ray_box_margin", 1.05))),
            water_depth + 1.0,
        ), 2),
        "bottom": bottom,
    }

    new_meta["bellhop_output"] = {
        "mode": "broadband",
        "num_freqs_requested": len(ship_freqs),
        "num_freqs_succeeded": len(broadband["channels"]),
        "freqs_hz": ship_freqs,
        "successful_freqs_hz": broadband["successful_freqs_hz"],
        "arrivals_per_frequency": {
            str(int(item["freq_hz"])): len(item["arrivals"])
            for item in broadband["channels"]
        },
        "absolute_first_arrival_delay_s": round(broadband["delay_reference_s"], 6),
        "cir_delay_reference": "global_first_arrival",
        "broadband_application": "frequency_response_interpolation",
        "tl_db": round(float(tl_est), 2),  # legacy name: pre-normalization broadband RMS gain dB
        "snr_db": snr_post,  # legacy name: source/pre-channel line-spectrum SNR dB
        "output_normalization_gain_db": gain_db,
        "cir_duration_s": audio_duration_s,
        "cir_fs_hz": fs,
    }

    # 删除原始数据中不适用于信道版本的冗余字段
    new_meta.pop("field_descriptions", None)
    new_meta.pop("conversion_notes", None)
    new_meta.pop("original_metadata", None)

    # --- 保存 ---
    write_audio(output_wav_path, fs, convolved)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "id": new_id,
        "status": "success",
        "wav": str(output_wav_path),
        "json": str(output_json_path),
        "snr_db": snr_post,
        "ssp_index": ssp_info["profile_index"],
        "freqs": ship_freqs,
    }


# ---- 进度统计 ----

class Progress:
    def __init__(self, total: int):
        self.total = total
        self.processed = 0
        self.succeeded = 0
        self.skipped = 0
        self.partial = 0
        self.no_arrivals = 0
        self.failed = 0
        self.start_time = time.time()
        self.report_every = 1 if total <= 100 else 50

    def update(self, status: str):
        self.processed += 1
        if status == "skipped":
            self.skipped += 1
        elif status == "success":
            self.succeeded += 1
        elif status == "partial":
            self.partial += 1
        elif status == "no_arrivals":
            self.no_arrivals += 1
        else:
            self.failed += 1
        if self.processed % self.report_every == 0 or self.processed == self.total:
            elapsed = time.time() - self.start_time
            rate = self.processed / elapsed if elapsed > 0 else 0
            eta = (self.total - self.processed) / rate if rate > 0 else 0
            print(f"  [{self.processed}/{self.total}] "
                  f"ok={self.succeeded} existing={self.skipped} partial={self.partial} "
                  f"no_arr={self.no_arrivals} err={self.failed} "
                  f"| {rate:.1f}/s | ETA {eta:.0f}s")


# ---- 入口 ----

def main():
    parser = argparse.ArgumentParser(description="Ship Step 1: BELLHOP broadband channel convolution")
    parser.add_argument("--config", default="config_ship.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    dataset_cfg = cfg["dataset"]
    channel_cfg = cfg["channel"]
    limits = cfg.get("limits", {})

    # 初始化
    print("=" * 60)
    print("舰船辐射噪声 — Step 1: BELLHOP 宽带信道卷积")
    print(f"  SSP NetCDF: {paths['ssp_nc']}")
    print(f"  BELLHOP:    {paths['bellhop_exe']}")
    print(f"  射线策略:   {channel_cfg.get('num_beams', 'adaptive')} "
          f"(min={channel_cfg.get('min_beams', 1000)}, "
          f"per_km={channel_cfg.get('beams_per_km', 100)}, "
          f"max={channel_cfg.get('max_beams', 8000)})")
    print(f"  宽带频点:   {channel_cfg['frequency']['ship_freqs_hz']} Hz")

    ssp_sampler = SSPSampler(paths["ssp_nc"])
    print(f"  已加载 {ssp_sampler.n_profiles} 根 SSP 剖面")

    bellhop = BellhopRunner(
        bellhop_exe=paths["bellhop_exe"],
        temp_dir=paths.get("temp_dir", "temp_bellhop"),
    )

    random_seed = int(limits.get("random_seed", 42))

    raw_root = Path(paths["raw_data"])
    output_root = Path(paths["processed_audio"])
    output_root.mkdir(parents=True, exist_ok=True)

    # 收集所有舰船 JSON 文件
    ds_root = raw_root / dataset_cfg["name"]
    if not ds_root.exists():
        print(f"[FATAL] 数据集目录不存在: {ds_root}")
        return

    classes = dataset_cfg["classes"]
    all_tasks = find_ship_json_files(ds_root, classes, dataset_cfg["json_subdir"], dataset_cfg["json_ext"])
    print(f"  数据集: {dataset_cfg['name']}")
    print(f"  舰船类别: {len(classes)} 类 — {classes}")
    print(f"  JSON 文件总数: {len(all_tasks)}")

    # 应用限制
    max_samples = limits.get("max_samples")
    if max_samples:
        all_tasks = all_tasks[:max_samples]
        print(f"  限制处理: {max_samples} 条")

    print(f"\n开始处理 {len(all_tasks)} 条音频...\n")

    progress = Progress(len(all_tasks))
    skip_existing = limits.get("skip_existing", True)

    for (json_path, class_name) in all_tasks:
        if len(all_tasks) <= 100:
            print(f"  [RUN] {class_name}/{json_path.name}", flush=True)
        result = process_single_ship(
            json_path=json_path,
            class_name=class_name,
            dataset_root=ds_root,
            dataset_cfg=dataset_cfg,
            channel_cfg=channel_cfg,
            ssp_sampler=ssp_sampler,
            bellhop=bellhop,
            output_root=output_root,
            random_seed=random_seed,
            skip_existing=skip_existing,
        )
        progress.update(result["status"])

    # 汇总
    elapsed = time.time() - progress.start_time
    print(f"\n{'=' * 60}")
    print(f"舰船 Step 1 完成")
    print(f"  总数: {progress.total}")
    print(f"  成功: {progress.succeeded}")
    print(f"  跳过: {progress.skipped}")
    print(f"  部分成功: {progress.partial}")
    print(f"  无到达: {progress.no_arrivals}")
    print(f"  失败: {progress.failed}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  输出: {output_root.resolve()}")
    print(f"{'=' * 60}")
    if progress.succeeded + progress.partial + progress.skipped == 0:
        raise SystemExit("舰船 Step 1 没有产生或复用任何有效样本")


if __name__ == "__main__":
    main()
