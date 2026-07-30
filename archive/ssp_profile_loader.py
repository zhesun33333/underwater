"""
ssp_profile_loader_pycharm_clean.py

PyCharm 直接运行版：
1. 从 acoustic_model_100profiles.nc 中读取一根完整 SSP；
2. 默认读取中心剖面 profile_index=50；
3. 绘制 0–100 m 声速剖面；
4. 输出 selected_ssp_profile.png。

说明：
    图中不显示经纬度和距离信息；
    控制台也不打印经纬度和距离信息。
"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# 用户配置：PyCharm 直接运行时主要改这里
# ============================================================

# 当前 Python 文件所在目录
BASE_DIR = Path(__file__).resolve().parent

# NetCDF 声速剖面文件
# 如果 acoustic_model_100profiles.nc 和本 py 文件在同一个目录，就保持这个写法
NC_FILE = Path(r"F:\paper\SSP_inverse\EOF\ssp-eof\acoustic_model_100profiles.nc")

# 选择第几根 SSP
# 100 根剖面时，profile_index=50 大致是中间剖面
PROFILE_INDEX = 50

# 如果想按距离选择，例如 RANGE_KM = 7.5，则把 PROFILE_INDEX 改成 None
RANGE_KM = None

# 检查是否至少覆盖到 100 m
REQUIRED_DEPTH = 100.0

# 输出图片
OUTPUT_PNG = BASE_DIR / "selected_ssp_profile.png"

# 是否弹出图像窗口
SHOW_FIGURE = True


def load_sound_speed_profile(
    nc_file,
    profile_index=None,
    range_km=None,
    variable=None,
    required_depth=None,
    remove_nan=True,
):
    """
    从 NetCDF 文件中读取一根完整的垂向声速剖面。

    Parameters
    ----------
    nc_file : str or Path
        NetCDF 声速剖面文件路径。
    profile_index : int or None
        按剖面序号选择一根 SSP。例如 profile_index=50。
    range_km : float or None
        按距离选择最近的一根 SSP。例如 range_km=7.5。
    variable : str or None
        NetCDF 中的变量名。若为 None，则自动寻找 sound_speed 或 ssp_profiles。
    required_depth : float or None
        如果给定，则检查该剖面是否至少覆盖到 required_depth。
    remove_nan : bool
        是否删除 NaN 深度点。

    Returns
    -------
    z : np.ndarray
        深度数组，单位 m。
    c : np.ndarray
        声速数组，单位 m/s。
    info : dict
        所选剖面的基本信息。
    """

    nc_file = Path(nc_file)

    if not nc_file.exists():
        raise FileNotFoundError(
            f"NetCDF 文件不存在: {nc_file}\n"
            f"请确认 acoustic_model_100profiles.nc 是否和本 Python 文件在同一个目录，"
            f"或者将 NC_FILE 改成完整绝对路径。"
        )

    ds = xr.open_dataset(nc_file)

    # --------------------------------------------------------
    # 1. 自动选择声速变量
    # --------------------------------------------------------
    if variable is None:
        if "sound_speed" in ds:
            variable = "sound_speed"
        elif "ssp_profiles" in ds:
            variable = "ssp_profiles"
        else:
            raise KeyError(
                "无法在 NetCDF 文件中找到声速变量。"
                "期望变量名为 'sound_speed' 或 'ssp_profiles'。"
            )

    if variable not in ds:
        raise KeyError(f"变量 '{variable}' 不存在于文件 {nc_file}")

    da = ds[variable]

    if "depth" not in da.dims:
        raise ValueError(f"变量 '{variable}' 必须包含 depth 维度。")

    if "range" not in da.dims:
        raise ValueError(f"变量 '{variable}' 必须包含 range 维度。")

    # --------------------------------------------------------
    # 2. 选择一根剖面
    # --------------------------------------------------------
    if profile_index is not None and range_km is not None:
        raise ValueError("profile_index 和 range_km 只能二选一，不能同时指定。")

    if profile_index is not None:
        profile_index = int(profile_index)

        if profile_index < 0 or profile_index >= da.sizes["range"]:
            raise IndexError(
                f"profile_index={profile_index} 超出范围。"
                f"有效范围为 0 到 {da.sizes['range'] - 1}。"
            )

        da_1d = da.isel(range=profile_index)
        selected_index = profile_index

    elif range_km is not None:
        da_1d = da.sel(range=float(range_km), method="nearest")
        selected_range = float(da_1d["range"].values)
        selected_index = int(np.argmin(np.abs(ds["range"].values - selected_range)))

    else:
        # 默认选择中心剖面
        selected_index = da.sizes["range"] // 2
        da_1d = da.isel(range=selected_index)

    # --------------------------------------------------------
    # 3. 整理为 z 和 c
    # --------------------------------------------------------
    da_1d = da_1d.transpose("depth")

    z = ds["depth"].values.astype(float)
    c = da_1d.values.astype(float)

    if remove_nan:
        valid = np.isfinite(z) & np.isfinite(c)
        z = z[valid]
        c = c[valid]

    if len(z) < 2:
        raise ValueError("所选 SSP 的有效深度点少于 2 个，无法绘图或使用。")

    # 按深度升序排列
    order = np.argsort(z)
    z = z[order]
    c = c[order]

    max_valid_depth = float(np.nanmax(z))

    if required_depth is not None and max_valid_depth < float(required_depth):
        raise ValueError(
            f"所选 SSP 仅覆盖到 {max_valid_depth:.2f} m，"
            f"但要求至少覆盖到 {float(required_depth):.2f} m。"
        )

    info = {
        "nc_file": str(nc_file),
        "variable": variable,
        "selected_profile_index": int(selected_index),
        "min_depth_m": float(np.nanmin(z)),
        "max_depth_m": max_valid_depth,
        "n_depth": int(len(z)),
        "min_sound_speed_mps": float(np.nanmin(c)),
        "max_sound_speed_mps": float(np.nanmax(c)),
    }

    return z, c, info


def plot_sound_speed_profile(
    z,
    c,
    output_png="selected_ssp_profile.png",
    show=False,
):
    """
    绘制一根声速剖面。

    图中不显示经纬度和距离信息。
    """

    output_png = Path(output_png)

    fig, ax = plt.subplots(figsize=(5.5, 7.0))

    ax.plot(c, z, linewidth=2.0)

    ax.invert_yaxis()
    ax.set_xlabel("Sound Speed (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Sound Speed Profile")
    ax.grid(True, linestyle=":", alpha=0.7)

    fig.savefig(output_png, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

    return str(output_png)


def main():
    """
    PyCharm 直接运行入口。
    """

    z, c, info = load_sound_speed_profile(
        nc_file=NC_FILE,
        profile_index=PROFILE_INDEX,
        range_km=RANGE_KM,
        required_depth=REQUIRED_DEPTH,
    )

    output_png = plot_sound_speed_profile(
        z=z,
        c=c,
        output_png=OUTPUT_PNG,
        show=SHOW_FIGURE,
    )

    print("=" * 60)
    print("SSP loaded successfully.")
    print(f"NetCDF file          : {info['nc_file']}")
    print(f"Variable            : {info['variable']}")
    print(f"Profile index       : {info['selected_profile_index']}")
    print(f"Depth range         : {info['min_depth_m']:.2f} - {info['max_depth_m']:.2f} m")
    print(f"Number of z points  : {info['n_depth']}")
    print(f"Sound speed range   : {info['min_sound_speed_mps']:.3f} - {info['max_sound_speed_mps']:.3f} m/s")
    print(f"Output figure       : {output_png}")
    print("=" * 60)


if __name__ == "__main__":
    main()
