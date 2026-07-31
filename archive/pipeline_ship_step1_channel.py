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
from utils.bellhop_runner import BellhopRunner, apply_channel

warnings.filterwarnings("ignore")


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
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)
    elif data.dtype == np.float64:
        pass
    else:
        data = data.astype(np.float64)
        mx = np.max(np.abs(data))
        if mx > 0:
            data /= mx
    if data.ndim > 1:
        data = data[:, 0]
    return sr, data


def write_audio(wav_path: Path, sr: int, data: np.ndarray, bits: int = 32):
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if bits == 32:
        int_data = (data * 2147483647.0).astype(np.int32)
    elif bits == 16:
        int_data = (data * 32767.0).astype(np.int16)
    else:
        raise ValueError(f"不支持的位深: {bits}")
    wavfile.write(wav_path, sr, int_data)


# ---- 后处理 SNR 计算 ----

def compute_post_channel_snr_ship(meta: dict) -> float:
    """
    舰船噪声: 连续宽带信号，信号和背景噪声经过同一 CIR。

    LTI 系统下宽带 SNR 近似不变:
      SNR_post ≈ SNR_orig = snr_after_mix_db

    # SNR 不变，衰减程度看 actual TL (宽带 TL 仅做参考)
    若 TL 过大（如 < -60 dB），说明信号绝对电平已极低。
    """
    orig_snr = meta.get("snr_after_mix_db")
    if orig_snr is None or (isinstance(orig_snr, float) and np.isnan(orig_snr)):
        return 0.0
    return round(float(orig_snr), 2)


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
    rng: np.random.Generator,
    skip_existing: bool = True,
) -> Optional[str]:
    """处理单条舰船辐射噪声: 多频 BELLHOP → 宽带 CIR → 卷积 → 保存。"""
    try:
        meta = load_jsonc(json_path)
    except Exception as e:
        print(f"  [SKIP] JSON 解析失败: {json_path} — {e}")
        return None

    sample_id = meta.get("id", json_path.stem)
    wav_subdir = dataset_cfg["wav_subdir"]

    # 查找 WAV
    wav_path = find_ship_wav(dataset_root, class_name, wav_subdir, sample_id)
    if wav_path is None:
        print(f"  [SKIP] WAV 不存在: {sample_id}")
        return None

    # 提取参数
    geo = get_geometry(meta)
    audio_duration_s = float(meta.get("audio_duration_s", 30))

    n_channels = channel_cfg.get("num_channels_per_audio", 2)
    use_jitter = channel_cfg.get("use_json_geometry", True)
    jitter = channel_cfg.get("geometry_jitter_pct", 0.1) if use_jitter else None

    results = []
    for ch_idx in range(n_channels):
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
                audio_duration_s=audio_duration_s,
                rng=rng,
                jitter=jitter,
                skip_existing=skip_existing,
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"  [ERR] {sample_id}_ch{ch_idx}: {e}")
            continue

    return sample_id if results else None


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
    audio_duration_s: float,
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

    if skip_existing and output_wav_path.exists() and output_json_path.exists():
        return {"id": new_id, "status": "skipped"}

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

    range_km = range_m / 1000.0 if range_m else rng.uniform(*channel_cfg.get("range_km_range", [0.2, 10.0]))

    # --- SSP 采样 ---
    min_ssp_depth = max(20.0, (tx_depth or 20) + 10, (rx_depth or 20) + 10)
    z, c, ssp_info = ssp_sampler.sample_matching_depth(min_depth_m=min_ssp_depth, rng=rng)

    # 水深: ship 数据的 water_depth_m 常为 []，使用 SSP 最大深度
    if water_depth is None:
        water_depth = max(z)
        wd_range = channel_cfg.get("water_depth_m_range")
        if wd_range:
            water_depth = min(water_depth, rng.uniform(*wd_range))

    # 检查 SSP 是否覆盖到水底: max(z) < water_depth 意味着 SSP 太浅，
    # BELLHOP 会自行外推深层声速，结果不可靠，直接跳过
    ssp_max_depth = float(max(z))
    if ssp_max_depth < water_depth - 1.0:
        print(f"  [SKIP] SSP 剖面太浅 (max_ssp={ssp_max_depth:.1f}m) 无法覆盖水深 "
              f"({water_depth:.1f}m): {sample_id}_ch{ch_idx}")
        return None

    tx_depth = tx_depth if tx_depth else rng.uniform(5.0, water_depth * 0.5)
    rx_depth = rx_depth if rx_depth else rng.uniform(5.0, water_depth * 0.5)

    tx_depth = max(1.0, min(tx_depth, water_depth - 1.0))
    rx_depth = max(1.0, min(rx_depth, water_depth - 1.0))
    range_km = max(0.01, range_km)

    # --- 读取音频 ---
    fs, audio = read_audio(wav_path)
    declared_fs = int(meta.get("model_fs_hz", 16000))
    if fs != declared_fs:
        print(f"  [WARN] 采样率不匹配: wav={fs}, meta={declared_fs}")

    # --- 宽带 BELLHOP ---
    bottom = channel_cfg.get("bottom", {})
    ship_freqs = channel_cfg.get("frequency", {}).get("ship_freqs_hz", [50, 100, 200, 400, 800, 1600])

    cir = bellhop.compute_broadband_cir(
        depths_m=z, sound_speeds_mps=c,
        freqs_hz=ship_freqs,
        source_depth_m=tx_depth, receiver_depth_m=rx_depth,
        range_km=range_km, water_depth_m=water_depth,
        audio_fs_hz=fs, audio_duration_s=audio_duration_s,
        bottom=bottom,
    )

    # --- 卷积 ---
    convolved = apply_channel(audio, cir)

    # --- RMS 传播损失 (归一化前计算，反映信道真实能量增益) ---
    rms_orig = np.sqrt(max(1e-20, np.mean(audio ** 2)))
    rms_conv = np.sqrt(max(1e-20, np.mean(convolved ** 2)))
    tl_est = 20.0 * np.log10(rms_conv / rms_orig)

    # --- 输出峰值归一化 ---
    # CIR 多频叠加后 peak 归一化无法保证卷积输出音量一致；
    # 此处对卷积结果做 peak 归一化，确保所有样本输出电平统一，
    # 同时保留信道的频率选择性（频谱形状不变、SNR 不变）。
    peak = float(np.max(np.abs(convolved)))
    target_peak = float(channel_cfg.get("output_norm_peak", 0.95))
    if peak > 1e-10:
        convolved *= (target_peak / peak)
        gain_db = round(20.0 * float(np.log10(target_peak / peak)), 2)
    else:
        gain_db = 0.0

    # --- 后处理 SNR ---
    snr_post = compute_post_channel_snr_ship(meta)

    # --- 更新元数据 ---
    new_meta = dict(meta)
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
        "bottom": bottom,
    }

    new_meta["bellhop_output"] = {
        "mode": "broadband",
        "num_freqs": len(ship_freqs),
        "freqs_hz": ship_freqs,
        "snr_db": snr_post,
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
        self.failed = 0
        self.start_time = time.time()

    def update(self, success: bool, skip: bool = False):
        self.processed += 1
        if skip:
            self.skipped += 1
        elif success:
            self.succeeded += 1
        else:
            self.failed += 1
        if self.processed % 50 == 0 or self.processed == self.total:
            elapsed = time.time() - self.start_time
            rate = self.processed / elapsed if elapsed > 0 else 0
            eta = (self.total - self.processed) / rate if rate > 0 else 0
            print(f"  [{self.processed}/{self.total}] "
                  f"ok={self.succeeded} skip={self.skipped} fail={self.failed} "
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
    print(f"  宽带频点:   {channel_cfg['frequency']['ship_freqs_hz']} Hz")

    ssp_sampler = SSPSampler(paths["ssp_nc"])
    print(f"  已加载 {ssp_sampler.n_profiles} 根 SSP 剖面")

    bellhop = BellhopRunner(
        bellhop_exe=paths["bellhop_exe"],
        temp_dir=paths.get("temp_dir", "temp_bellhop"),
    )

    rng = np.random.default_rng(limits.get("random_seed", 42))

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
        result = process_single_ship(
            json_path=json_path,
            class_name=class_name,
            dataset_root=ds_root,
            dataset_cfg=dataset_cfg,
            channel_cfg=channel_cfg,
            ssp_sampler=ssp_sampler,
            bellhop=bellhop,
            output_root=output_root,
            rng=rng,
            skip_existing=skip_existing,
        )
        progress.update(success=result is not None)

    # 汇总
    elapsed = time.time() - progress.start_time
    print(f"\n{'=' * 60}")
    print(f"舰船 Step 1 完成")
    print(f"  总数: {progress.total}")
    print(f"  成功: {progress.succeeded}")
    print(f"  跳过: {progress.skipped}")
    print(f"  失败: {progress.failed}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  输出: {output_root.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
