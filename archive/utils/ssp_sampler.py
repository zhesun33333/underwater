"""
SSP 剖面随机采样器
从 NetCDF 文件中读取声速剖面，支持随机采样和按索引选择。
"""
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class SSPSampler:
    """声速剖面采样器，封装 xarray NetCDF 读取。"""

    def __init__(self, nc_path: str):
        self.nc_path = Path(nc_path)
        if not self.nc_path.exists():
            raise FileNotFoundError(f"SSP NetCDF 文件不存在: {self.nc_path}")

        import xarray as xr
        self.ds = xr.open_dataset(self.nc_path)

        # 自动寻找声速变量
        if "sound_speed" in self.ds:
            self.variable = "sound_speed"
        elif "ssp_profiles" in self.ds:
            self.variable = "ssp_profiles"
        else:
            raise KeyError("NetCDF 中未找到 sound_speed 或 ssp_profiles 变量")

        self.da = self.ds[self.variable]
        self.n_profiles = self.da.sizes["range"]
        self._depths = self.ds["depth"].values.astype(float)

    def sample(self, index: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        随机或指定抽取一根 SSP 剖面。

        Args:
            index: 剖面索引，None 则随机
            rng: numpy 随机数生成器

        Returns:
            z: 深度数组 (m), 升序
            c: 声速数组 (m/s)
            info: 剖面元信息
        """
        if rng is None:
            rng = np.random.default_rng()

        if index is None:
            index = rng.integers(0, self.n_profiles)
        else:
            index = int(index) % self.n_profiles

        da_1d = self.da.isel(range=index).transpose("depth")
        z = self._depths.copy()
        c = da_1d.values.astype(float)

        # 去除 NaN
        valid = np.isfinite(z) & np.isfinite(c)
        z = z[valid]
        c = c[valid]

        if z.size < 2:
            raise ValueError(f"SSP profile {index} has fewer than two finite points")
        if np.any(c <= 0.0):
            raise ValueError(f"SSP profile {index} contains non-positive sound speed")

        # 按深度升序
        order = np.argsort(z)
        z = z[order]
        c = c[order]

        unique = np.concatenate(([True], np.diff(z) > 0.0))
        z = z[unique]
        c = c[unique]
        if z.size < 2:
            raise ValueError(f"SSP profile {index} has fewer than two unique depths")

        info = {
            "profile_index": int(index),
            "n_profiles_total": self.n_profiles,
            "min_depth_m": float(np.min(z)),
            "max_depth_m": float(np.max(z)),
            "n_depth_points": len(z),
            "min_speed_mps": float(np.min(c)),
            "max_speed_mps": float(np.max(c)),
            "surface_speed_mps": float(c[0]) if len(c) > 0 else None,
        }
        return z, c, info

    def sample_matching_depth(self, min_depth_m: float = 50.0, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        采样一根至少覆盖 min_depth_m 深度的 SSP。
        最多尝试 50 次，失败则返回最深的一根。
        """
        if rng is None:
            rng = np.random.default_rng()

        best_z, best_c, best_info = self.sample(index=0, rng=rng)
        best_max_depth = best_info["max_depth_m"]

        for _ in range(50):
            z, c, info = self.sample(index=None, rng=rng)
            if info["max_depth_m"] >= min_depth_m:
                return z, c, info
            if info["max_depth_m"] > best_max_depth:
                best_z, best_c, best_info = z, c, info
                best_max_depth = info["max_depth_m"]

        return best_z, best_c, best_info

    def close(self):
        """关闭 NetCDF 数据集。"""
        if hasattr(self, 'ds'):
            self.ds.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
