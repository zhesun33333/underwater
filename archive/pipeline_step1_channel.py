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
from utils.bellhop_runner import (
    BellhopExecutionError, BellhopRunner, InvalidEnvironmentError,
    apply_channel, build_cir, make_channel_rng,
)

warnings.filterwarnings("ignore")

CHANNEL_MODEL_VERSION = "2.2"


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
        data = data[:, 0]  # 取第一声道
    if data.size == 0:
        raise ValueError(f"WAV 为空: {wav_path}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"WAV 包含 NaN/Inf: {wav_path}")
    if not np.any(np.abs(data) > 0.0):
        raise ValueError(f"WAV 全为零: {wav_path}")
    return int(sr), data


def write_audio(wav_path: Path, sr: int, data: np.ndarray, bits: int = 32):
    """写入 WAV 文件。data 为 float64 [-1, 1]。"""
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
        "sparse_arrivals": len(arrivals) <= 2,
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
    random_seed: int,
    skip_existing: bool = True,
) -> dict:
    """
    处理单条音频: BELLHOP 信道卷积 + 保存新文件。

    返回新 sample_id，失败返回 None。
    """
    # 1. 加载元数据
    try:
        meta = load_jsonc(json_path)
        if not isinstance(meta, dict):
            raise TypeError("JSON 顶层必须是对象")
    except Exception as e:
        print(f"  [SKIP] JSON 解析失败: {json_path} — {e}")
        return {"status": "failed", "reason": "json_parse"}

    sample_id = get_id(meta)
    if not sample_id or sample_id == "unknown":
        sample_id = json_path.stem
    wav_rel = get_wav_path(meta)

    # 2. 查找 WAV
    wav_path = dataset_root / wav_rel
    if not wav_path.exists():
        print(f"  [SKIP] WAV 不存在: {wav_path}")
        return {"status": "failed", "reason": "wav_missing"}

    # 3. 提取参数
    center_freq = get_center_freq(meta)
    geo = get_geometry(meta)
    # 4. 对每条音频生成 N 个不同信道的版本
    n_channels = int(channel_cfg.get("num_channels_per_audio", 2))
    if n_channels <= 0:
        return {"id": sample_id, "status": "failed", "reason": "invalid_channel_count"}
    if channel_cfg.get("use_json_geometry", True):
        jitter = float(channel_cfg.get("geometry_jitter_pct", 0.1))
        if not np.isfinite(jitter) or not 0.0 <= jitter < 1.0:
            return {"id": sample_id, "status": "failed", "reason": "invalid_geometry_jitter"}
    else:
        jitter = None

    results = []
    for ch_idx in range(n_channels):
        channel_rng = make_channel_rng(
            random_seed, "pulsecom", meta.get("signal_type", "UNKNOWN"),
            sample_id, ch_idx,
        )
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
                rng=channel_rng,
                jitter=jitter,
                skip_existing=skip_existing,
            )
            results.append(result)
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
        try:
            existing_meta = load_jsonc(output_json_path)
            existing_freq = (existing_meta.get("bellhop_env") or {}).get("freq_hz")
            frequency_matches = (
                center_freq is not None
                and existing_freq is not None
                and np.isclose(float(existing_freq), float(center_freq), rtol=0.0, atol=0.01)
            )
            if (existing_meta.get("channel_model_version") == CHANNEL_MODEL_VERSION
                    and frequency_matches):
                return {"id": new_id, "status": "existing"}
        except Exception:
            pass

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

    range_km = (range_m / 1000.0 if range_m is not None
                else rng.uniform(*channel_cfg.get("range_km_range", [10.0, 80.0])))

    # --- SSP 采样 ---
    min_ssp_depth = max(
        20.0, (tx_depth or 20) + 10, (rx_depth or 20) + 10,
        water_depth or 0.0,
    )
    z, c, ssp_info = ssp_sampler.sample_matching_depth(min_depth_m=min_ssp_depth, rng=rng)

    if water_depth is None:
        water_depth = max(z)  # 用 SSP 最大深度
        if channel_cfg.get("water_depth_m_range"):
            water_depth = min(water_depth, rng.uniform(*channel_cfg["water_depth_m_range"]))

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

    # 确保深度合理
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

    # --- BELLHOP + CIR (PulseCom 始终有中心频率) ---
    bottom = channel_cfg.get("bottom", {})

    if not center_freq or center_freq <= 0:
        print(f"  [SKIP] 无有效中心频率: center_freq={center_freq}")
        return {"id": new_id, "status": "failed", "reason": "invalid_center_frequency"}
    if center_freq > fs / 2.0:
        print(f"  [SKIP] 代表频率超过 WAV Nyquist: freq={center_freq:.2f}Hz, fs={fs}Hz")
        return {"id": new_id, "status": "failed", "reason": "frequency_above_nyquist"}

    run_result = bellhop.run_with_beam_retry(
        depths_m=z, sound_speeds_mps=c,
        freq_hz=center_freq,
        source_depth_m=tx_depth, receiver_depth_m=rx_depth,
        range_km=range_km, water_depth_m=water_depth,
        bottom=bottom,
        title=f"{sample_id}_ch{ch_idx}",
        num_beams=beam_setting,
        ray_box_margin=float(channel_cfg.get("ray_box_margin", 1.05)),
        max_arrivals=int(channel_cfg.get("max_arrivals", 20)),
        min_beams=int(channel_cfg.get("min_beams", 1000)),
        beams_per_km=float(channel_cfg.get("beams_per_km", 100.0)),
        max_beams=int(channel_cfg.get("max_beams", 8000)),
    )
    arrivals = run_result["arrivals"]
    effective_num_beams = run_result["num_beams_effective"]
    if not arrivals:
        print(f"  [SKIP] 射线加密复核后仍为 0 到达: {sample_id}_ch{ch_idx}  "
              f"range={range_km:.2f}km freq={center_freq:.0f}Hz")
        return {"id": new_id, "status": "no_arrivals"}

    first_arrival_delay_s = min(a["delay_s"] for a in arrivals)
    cir = build_cir(
        arrivals, fs, audio_duration_s, use_normalized=False,
        delay_reference_s=first_arrival_delay_s,
    )

    # --- 卷积 ---
    convolved = apply_channel(audio, cir)

    # --- 归一化前 CIR 能量增益 ---
    # 兼容既有数据格式，结果仍写入历史字段名 tl_db，但它实际定义为
    #   10*log10(sum(h[n]^2))，即 channel energy gain（通常为负值），
    # 而不是通常取正值的 transmission loss。数值越大（越接近 0）表示
    # 信道增益越高、衰减越弱；该值在输出 peak 归一化之前计算。
    cir_energy = float(np.sum(cir ** 2))
    tl_est = 10.0 * np.log10(max(cir_energy, 1e-40))

    # --- 输出峰值归一化 ---
    peak = float(np.max(np.abs(convolved)))
    target_peak = float(channel_cfg.get("output_norm_peak", 0.95))
    if not np.isfinite(target_peak) or not 0.0 < target_peak <= 1.0:
        raise ValueError(f"output_norm_peak 必须在 (0, 1]，实际为 {target_peak!r}")
    if peak > 1e-10:
        convolved *= (target_peak / peak)
        gain_db = round(20.0 * float(np.log10(target_peak / peak)), 2)
    else:
        gain_db = 0.0

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
        "freq_hz": round(center_freq, 2),
        "num_beams_setting": beam_setting,
        "num_beams_effective": effective_num_beams,
        "beam_retry": run_result["beam_retry"],
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
        "arrivals": arrivals,
        "summary": summarize_arrivals(arrivals),
        "tl_db": round(tl_est, 2),  # legacy name: pre-normalization channel energy gain dB
        "absolute_first_arrival_delay_s": round(first_arrival_delay_s, 6),
        "cir_delay_reference": "first_arrival",
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
        "status": "success",
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
        self.partial = 0
        self.no_arrivals = 0
        self.failed = 0
        self.start_time = time.time()
        self.report_every = 1 if total <= 100 else 100

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
    print(f"  射线策略: {channel_cfg.get('num_beams', 'adaptive')} "
          f"(min={channel_cfg.get('min_beams', 1000)}, "
          f"per_km={channel_cfg.get('beams_per_km', 100)}, "
          f"max={channel_cfg.get('max_beams', 8000)})")

    ssp_sampler = SSPSampler(paths["ssp_nc"])
    print(f"  已加载 {ssp_sampler.n_profiles} 根 SSP 剖面")

    bellhop = BellhopRunner(
        bellhop_exe=paths["bellhop_exe"],
        temp_dir=paths.get("temp_dir", "temp_bellhop"),
    )

    random_seed = int(limits.get("random_seed", 42))

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
        if len(all_tasks) <= 100:
            print(f"  [RUN] {json_path.name}", flush=True)
        result = process_single_audio(
            json_path=json_path,
            dataset_root=ds_root,
            dataset_cfg=ds_cfg,
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
    print(f"Step 1 完成")
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
        raise SystemExit("Step 1 没有产生或复用任何有效样本")


if __name__ == "__main__":
    main()
