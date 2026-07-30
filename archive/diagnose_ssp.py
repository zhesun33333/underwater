"""
SSP 复杂度诊断脚本
排查为什么 480k 样本的 SSP 复杂度全是 0.0038
"""
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import xarray as xr

NC_PATH = "acoustic_model_100profiles.nc"
PROCESSED_ROOT = Path("processed_audio")
SAMPLE_JSON_FILES = 20  # 随机抽查多少个 processed JSON

# ============================================================
# 1. NetCDF 剖面差异检查
# ============================================================
def check_netcdf():
    print("=" * 60)
    print("1. NetCDF 剖面差异检查")
    print("=" * 60)
    ds = xr.open_dataset(NC_PATH)

    if "sound_speed" in ds:
        var = "sound_speed"
    elif "ssp_profiles" in ds:
        var = "ssp_profiles"
    else:
        print("ERROR: 找不到 sound_speed 或 ssp_profiles")
        return

    da = ds[var]
    n_profiles = da.sizes["range"]
    depths = ds["depth"].values.astype(float)
    print(f"  变量: {var}, 剖面数: {n_profiles}, 深度点数: {len(depths)}")
    print(f"  深度范围: {depths[0]:.1f} ~ {depths[-1]:.1f} m")

    # 计算每个剖面的复杂度
    def complexity(speeds):
        valid = np.isfinite(speeds)
        z = depths[valid]
        c = speeds[valid]
        if len(z) <= 1:
            return None
        grads = np.diff(c) / np.diff(z)
        return float(np.var(grads))

    complexities = []
    for i in range(n_profiles):
        prof = da.isel(range=i).values.astype(float)
        comp = complexity(prof)
        if comp is not None:
            complexities.append(comp)

    print(f"\n  复杂度统计 (raw, 未四舍五入):")
    print(f"    min  = {min(complexities):.10f}")
    print(f"    max  = {max(complexities):.10f}")
    print(f"    mean = {np.mean(complexities):.10f}")
    print(f"    std  = {np.std(complexities):.10f}")
    print(f"    唯一值数量: {len(set(round(c, 10) for c in complexities))}")

    # 检查剖面之间的实际差异
    print(f"\n  各剖面表面声速差异:")
    surface_speeds = []
    for i in range(min(n_profiles, 10)):
        prof = da.isel(range=i).values.astype(float)
        surface_speeds.append(prof[0])
    print(f"    前 10 个剖面 surface speed: {[f'{s:.1f}' for s in surface_speeds]}")
    if len(set(round(s, 1) for s in surface_speeds)) == 1:
        print(f"    WARNING: 前 10 个剖面表面声速完全相同!")

    # 剖面间差异热力检查
    print(f"\n  相邻深度点平均梯度 (前 5 个剖面):")
    for i in range(min(n_profiles, 5)):
        prof = da.isel(range=i).values.astype(float)
        valid = np.isfinite(prof)
        z = depths[valid]
        c = prof[valid]
        grads = np.diff(c) / np.diff(z)
        print(f"    剖面 {i}: mean_grad={np.mean(grads):.6f}, var={np.var(grads):.8f}")

    ds.close()
    return complexities


# ============================================================
# 2. Processed JSON 剖面采样分布
# ============================================================
def check_processed_jsons():
    print("\n" + "=" * 60)
    print("2. Processed JSON 剖面采样分布")
    print("=" * 60)

    # 收集所有 processed JSON
    json_files = []
    for ext in (".jsonc", ".json"):
        json_files.extend(PROCESSED_ROOT.rglob(f"*{ext}"))

    if not json_files:
        print("  ERROR: processed_audio 下未找到 JSON 文件")
        return

    print(f"  总 JSON 文件数: {len(json_files)}")

    # 随机抽查
    rng = np.random.default_rng(42)
    sample_files = rng.choice(json_files, min(SAMPLE_JSON_FILES, len(json_files)), replace=False)

    profile_indices = []
    raw_complexities = []

    def compute_complexity(depths, speeds):
        if not depths or not speeds or len(depths) <= 1:
            return None
        grads = [(speeds[i+1] - speeds[i]) / (depths[i+1] - depths[i])
                 for i in range(len(depths) - 1)]
        if len(grads) == 0:
            return None
        mean_g = sum(grads) / len(grads)
        return sum((g - mean_g) ** 2 for g in grads) / len(grads)

    for jf in sample_files:
        try:
            meta = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        ssp = meta.get("bellhop_env", {}).get("ssp", {})
        idx = ssp.get("profile_index")
        depths = ssp.get("depths_m")
        speeds = ssp.get("sound_speeds_mps")

        if idx is not None:
            profile_indices.append(idx)

        if depths and speeds:
            comp = compute_complexity(depths, speeds)
            if comp is not None:
                raw_complexities.append(comp)

    print(f"\n  抽查 {len(sample_files)} 个文件:")
    if profile_indices:
        counter = Counter(profile_indices)
        print(f"    命中剖面数: {len(counter)} (共 {len(profile_indices)} 个)")
        print(f"    剖面索引分布: {dict(sorted(counter.items()))}")
        if len(counter) == 1:
            print(f"    CRITICAL: 只命中了剖面 {list(counter.keys())[0]}！采样器可能坏了！")

    if raw_complexities:
        print(f"\n    复杂度原始值 (高精度, 共 {len(raw_complexities)} 个):")
        for i, c in enumerate(raw_complexities):
            print(f"      [{i}] {c:.12f}")
        unique = set(round(c, 12) for c in raw_complexities)
        print(f"    唯一值数量: {len(unique)}")
        if len(unique) == 1:
            print(f"    CRITICAL: 抽查的所有样本复杂度完全相等!")

    return profile_indices


# ============================================================
# 3. 采样器复现
# ============================================================
def check_sampler_behavior():
    print("\n" + "=" * 60)
    print("3. SSPSampler 行为复现")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.ssp_sampler import SSPSampler

    sampler = SSPSampler(NC_PATH)
    print(f"  n_profiles = {sampler.n_profiles}")

    # 模拟 pipeline 实际调用方式
    rng = np.random.default_rng(42)

    # 测试：多次调用 sample_matching_depth，看返回的 profile_index 分布
    indices = []
    for i in range(1000):
        # 模拟典型的 min_ssp_depth (PulseCom: ~30m, Ship: ~20m)
        min_depth = rng.uniform(20.0, 50.0)
        _, _, info = sampler.sample_matching_depth(min_depth_m=min_depth, rng=rng)
        indices.append(info["profile_index"])

    counter = Counter(indices)
    print(f"  1000 次采样后命中的唯一剖面数: {len(counter)}")
    print(f"  各剖面命中次数 (前 10 和后 10):")
    sorted_items = sorted(counter.items())
    for idx, cnt in sorted_items[:10]:
        print(f"    剖面 {idx:3d}: {cnt:4d} 次")
    if len(sorted_items) > 20:
        print(f"    ...")
    for idx, cnt in sorted_items[-10:]:
        print(f"    剖面 {idx:3d}: {cnt:4d} 次")

    if len(counter) == 1:
        idx = list(counter.keys())[0]
        print(f"\n  BUG CONFIRMED: 采样器始终返回剖面 {idx}！")
        print(f"  检查 sample_matching_depth 中的 min_depth_m 过滤逻辑...")

        # 进一步排查: 检查各剖面的 max_depth
        print(f"\n  各剖面 max_depth:")
        for i in range(sampler.n_profiles):
            z, c, info = sampler.sample(index=i)
            print(f"    剖面 {i}: max_depth={info['max_depth_m']:.1f}m")

    sampler.close()


# ============================================================
if __name__ == "__main__":
    check_netcdf()
    check_processed_jsons()
    check_sampler_behavior()
