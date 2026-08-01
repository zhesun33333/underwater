"""
水声大模型数据合成管线 — 第一步：BELLHOP 信道卷积 (PulseCom 专用)

功能:
  1. 遍历 raw_data/PulseCom 所有音频
  2. 读取 JSONC 元数据，提取几何参数与信号参数
  3. 随机采样 SSP 剖面
  4. 调用 BELLHOP 生成单频信道冲激响应 (CIR)
  5. 音频卷积 → 写入 processed_audio/
  6. 更新 JSON 元数据 (追加 bellhop_env / bellhop_output)

用法:
  python pipeline_step1_channel.py [--config config.yaml]
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.io import wavfile

# 将当前目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.json_parser import (
    load_jsonc, get_wav_path, get_id,
    get_center_freq, get_geometry,
)
from utils.ssp_sampler import SSPSampler
from utils.bellhop_runner import BellhopRunner, build_cir, apply_channel

warnings.filterwarnings("ignore")


# ---- 配置加载 ----

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 展开环境变量
    cfg = _resolve_env_vars(cfg)
    return cfg


def _resolve_env_vars(obj):
    """递归解析配置中 ${VAR} 格式的环境变量。"""
    import os
    import re
    if isinstance(obj, str):
        pattern = re.compile(r'\$\{(\w+)\}')
        def replacer(m):
            return os.environ.get(m.group(1), m.group(0))
        return pattern.sub(replacer, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


# ---- 工具函数 ----

def find_json_files(dataset_root: Path, json_subdir: str, json_ext: str) -> List[Path]:
    """递归查找所有 JSON/JSONC 文件。"""
    json_dir = dataset_root / json_subdir
    if not json_dir.exists():
        print(f"  [WARN] 目录不存在: {json_dir}")
        return []
    files = sorted(json_dir.rglob(f"*{json_ext}"))
    # 过滤掉 manifest 等非样本文件
    files = [f for f in files if not f.name.startswith("manifest")]
    return files


def read_audio(wav_path: Path) -> Tuple[int, np.ndarray]:
    """读取 WAV，返回 (sample_rate, samples)。samples 转为 float64 [-1, 1]。"""
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
        data = data[:, 0]  # 取第一声道
    return sr, data


def write_audio(wav_path: Path, sr: int, data: np.ndarray, bits: int = 32):
    """写入 WAV 文件。data 为 float64 [-1, 1]。"""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if bits == 32:
        int_data = (data * 2147483647.0).astype(np.int32)
    elif bits == 16:
        int_data = (data * 32767.0).astype(np.int16)
    else:
        raise ValueError(f"不支持的位深: {bits}")
    wavfile.write(wav_path, sr, int_data)


# ---- BELLHOP 结果摘要 ----

def summarize_arrivals(arrivals: List[Dict]) -> dict:
    """从到达结构中提取关键摘要。"""
    if not arrivals:
        return {"num_arrivals": 0}
    delays = [a["delay_s"] for a in arrivals]
    amps_db = [a.get("amplitude_db", -400) for a in arrivals]
    return {
        "num_arrivals": len(arrivals),
        "max_delay_s": round(float(max(delays)), 6),
        "min_delay_s": round(float(min(delays)), 6),
        "mean_delay_s": round(float(np.mean(delays)), 6),
        "max_amplitude_db": round(float(max(amps_db)), 2),
        "has_shadow_zone": len(arrivals) <= 2,
    }


# ---- 主流程 ----

def process_single_audio(
    json_path: Path,
    dataset_root: Path,
    dataset_cfg: dict,
    channel_cfg: dict,
    ssp_sampler: SSPSampler,
    bellhop: BellhopRunner,
    output_root: Path,
    rng: np.random.Generator,
    skip_existing: bool = True,
) -> Optional[str]:
    """
    处理单条音频: BELLHOP 信道卷积 + 保存新文件。

    返回新 sample_id，失败返回 None。
    """
    # 1. 加载元数据
    try:
        meta = load_jsonc(json_path)
    except Exception as e:
        print(f"  [SKIP] JSON 解析失败: {json_path} — {e}")
        return None

    sample_id = get_id(meta)
    wav_rel = get_wav_path(meta)

    # 2. 查找 WAV
    wav_path = dataset_root / wav_rel
    if not wav_path.exists():
        print(f"  [SKIP] WAV 不存在: {wav_path}")
        return None

    # 3. 提取参数
    center_freq = get_center_freq(meta)
    geo = get_geometry(meta)
    audio_duration_s = float(meta.get("audio_duration_s", 30))

    # 4. 对每条音频生成 N 个不同信道的版本
    n_channels = channel_cfg.get("num_channels_per_audio", 2)
    if channel_cfg.get("use_json_geometry", True):
        jitter = channel_cfg.get("geometry_jitter_pct", 0.1)
    else:
        jitter = None

    results = []
    for ch_idx in range(n_channels):
        try:
            result = _apply_one_channel(
                meta=meta,
                wav_path=wav_path,
                sample_id=sample_id,
                ch_idx=ch_idx,
                dataset_root=dataset_root,
                dataset_cfg=dataset_cfg,
                channel_cfg=channel_cfg,
                ssp_sampler=ssp_sampler,
                bellhop=bellhop,
                output_root=output_root,
                center_freq=center_freq,
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


def _apply_one_channel(
    meta: dict,
    wav_path: Path,
    sample_id: str,
    ch_idx: int,
    dataset_root: Path,
    dataset_cfg: dict,
    channel_cfg: dict,
    ssp_sampler: SSPSampler,
    bellhop: BellhopRunner,
    output_root: Path,
    center_freq: Optional[float],
    geo: dict,
    audio_duration_s: float,
    rng: np.random.Generator,
    jitter: Optional[float],
    skip_existing: bool,
) -> Optional[dict]:
    """应用单个信道并保存。"""

    new_id = f"{sample_id}_ch{ch_idx}"

    # PulseCom 输出路径: wav_by_type/<type>/ + jsonc/<type>/
    signal_type = meta.get("signal_type", "UNKNOWN")
    rel_dir = Path(dataset_cfg["wav_dir"]) / signal_type
    output_wav_dir = output_root / dataset_cfg["name"] / rel_dir
    output_json_dir = output_root / dataset_cfg["name"] / dataset_cfg["json_dir"] / signal_type

    output_wav_path = output_wav_dir / f"{new_id}.wav"
    output_json_path = output_json_dir / f"{new_id}{dataset_cfg['json_ext']}"

    if skip_existing and output_wav_path.exists() and output_json_path.exists():
        return {"id": new_id, "status": "skipped"}

    # --- 几何参数 ---
    tx_depth = geo.get("tx_depth_m")
    rx_depth = geo.get("rx_depth_m")
    range_m = geo.get("range_m")
    water_depth = geo.get("water_depth_m")

    # 抖动 (在 JSON 原值上加减随机比例)
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

    if water_depth is None:
        water_depth = max(z)  # 用 SSP 最大深度
        if channel_cfg.get("water_depth_m_range"):
            water_depth = min(water_depth, rng.uniform(*channel_cfg["water_depth_m_range"]))

    # 检查 SSP 是否覆盖到水底: max(z) < water_depth 意味着 SSP 太浅，
    # BELLHOP 会自行外推深层声速，结果不可靠，直接跳过
    ssp_max_depth = float(max(z))
    if ssp_max_depth < water_depth - 1.0:
        print(f"  [SKIP] SSP 剖面太浅 (max_ssp={ssp_max_depth:.1f}m) 无法覆盖水深 "
              f"({water_depth:.1f}m): {sample_id}_ch{ch_idx}")
        return None

    tx_depth = tx_depth if tx_depth else rng.uniform(5.0, water_depth * 0.5)
    rx_depth = rx_depth if rx_depth else rng.uniform(5.0, water_depth * 0.5)

    # 确保深度合理
    tx_depth = max(1.0, min(tx_depth, water_depth - 1.0))
    rx_depth = max(1.0, min(rx_depth, water_depth - 1.0))
    range_km = max(0.01, range_km)

    # --- 读取音频 ---
    fs, audio = read_audio(wav_path)
    if fs != int(meta.get("model_fs_hz", 16000)):
        print(f"  [WARN] 采样率不匹配: wav={fs}, meta={meta.get('model_fs_hz')}")

    # --- BELLHOP + CIR (PulseCom 始终有中心频率) ---
    bottom = channel_cfg.get("bottom", {})

    if not center_freq or center_freq <= 0:
        print(f"  [SKIP] 无有效中心频率: center_freq={center_freq}")
        return None

    arrivals = bellhop.run(
        depths_m=z, sound_speeds_mps=c,
        freq_hz=center_freq,
        source_depth_m=tx_depth, receiver_depth_m=rx_depth,
        range_km=range_km, water_depth_m=water_depth,
        bottom=bottom,
        title=f"{sample_id}_ch{ch_idx}",
    )
    if not arrivals:
        print(f"  [SKIP] 0 到达 (声影区): {sample_id}_ch{ch_idx}  "
              f"range={range_km:.2f}km freq={center_freq:.0f}Hz")
        return None

    cir = build_cir(arrivals, fs, audio_duration_s, use_normalized=False)

    # --- 卷积 ---
    convolved = apply_channel(audio, cir)

    # --- RMS 传播损失 (归一化前计算，反映信道真实能量增益) ---
    rms_orig = np.sqrt(max(1e-20, np.mean(audio ** 2)))
    rms_conv = np.sqrt(max(1e-20, np.mean(convolved ** 2)))
    tl_est = 20.0 * np.log10(rms_conv / rms_orig)

    # --- 输出峰值归一化 ---
    peak = float(np.max(np.abs(convolved)))
    target_peak = float(channel_cfg.get("output_norm_peak", 0.95))
    if peak > 1e-10:
        convolved *= (target_peak / peak)
        gain_db = round(20.0 * float(np.log10(target_peak / peak)), 2)
    else:
        gain_db = 0.0

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
        "freq_hz": round(center_freq, 2),
        "bottom": bottom,
    }

    new_meta["bellhop_output"] = {
        "arrivals": arrivals,
        "summary": summarize_arrivals(arrivals),
        "tl_db": round(tl_est, 2),
        "output_normalization_gain_db": gain_db,
        "cir_duration_s": audio_duration_s,
        "cir_fs_hz": fs,
    }

    # 删除原始数据中不适用于信道版本的大段文档字段
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
        "tl_db": round(tl_est, 2),
        "n_arrivals": len(arrivals) if arrivals else 0,
        "ssp_index": ssp_info["profile_index"],
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
        if self.processed % 100 == 0 or self.processed == self.total:
            elapsed = time.time() - self.start_time
            rate = self.processed / elapsed if elapsed > 0 else 0
            eta = (self.total - self.processed) / rate if rate > 0 else 0
            print(f"  [{self.processed}/{self.total}] "
                  f"ok={self.succeeded} skip={self.skipped} fail={self.failed} "
                  f"| {rate:.1f}/s | ETA {eta:.0f}s")


# ---- 入口 ----

def main():
    parser = argparse.ArgumentParser(description="Step 1: BELLHOP channel convolution")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    channel_cfg = cfg["channel"]
    limits = cfg.get("limits", {})

    # 初始化 SSP 采样器和 BELLHOP
    print("=" * 60)
    print("初始化...")
    print(f"  SSP NetCDF: {paths['ssp_nc']}")
    print(f"  BELLHOP exe: {paths['bellhop_exe']}")

    ssp_sampler = SSPSampler(paths["ssp_nc"])
    print(f"  已加载 {ssp_sampler.n_profiles} 根 SSP 剖面")

    bellhop = BellhopRunner(
        bellhop_exe=paths["bellhop_exe"],
        temp_dir=paths.get("temp_dir", "temp_bellhop"),
    )

    rng = np.random.default_rng(limits.get("random_seed", 42))

    # 设置路径
    raw_root = Path(paths["raw_data"])
    output_root = Path(paths["processed_audio"])
    output_root.mkdir(parents=True, exist_ok=True)

    # 收集 PulseCom JSON 文件
    ds_cfg = cfg["datasets"]["pulsecom"]
    ds_root = raw_root / ds_cfg["name"]
    if not ds_root.exists():
        print(f"[FATAL] PulseCom 数据集目录不存在: {ds_root}")
        return

    json_files = find_json_files(ds_root, ds_cfg["json_dir"], ds_cfg["json_ext"])
    print(f"  {ds_cfg['name']}: 找到 {len(json_files)} 个 JSON 文件")

    all_tasks = [(jf, ds_root, ds_cfg) for jf in json_files]

    # 应用限制
    max_samples = limits.get("max_samples")
    if max_samples:
        all_tasks = all_tasks[:max_samples]

    print(f"\n总共 {len(all_tasks)} 条待处理\n")

    # 主循环
    progress = Progress(len(all_tasks))
    skip_existing = limits.get("skip_existing", True)

    for json_path, ds_root, ds_cfg in all_tasks:
        result = process_single_audio(
            json_path=json_path,
            dataset_root=ds_root,
            dataset_cfg=ds_cfg,
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
    print(f"Step 1 完成")
    print(f"  总数: {progress.total}")
    print(f"  成功: {progress.succeeded}")
    print(f"  跳过: {progress.skipped}")
    print(f"  失败: {progress.failed}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  输出: {output_root.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
