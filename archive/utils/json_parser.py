"""
JSONC/JSON 解析 + 三级标签提取
处理 PulseCom (.jsonc) 和 Ship (.json) 两种格式
"""
import json
import math
from pathlib import Path
from typing import Dict, Optional


def _strip_jsonc_comments(text: str) -> str:
    """去除 JSONC 中的 // 和 /* */ 注释，注意不破坏字符串内的内容。"""
    result = []
    i = 0
    n = len(text)
    in_string = False
    string_char = None

    while i < n:
        ch = text[i]
        if not in_string and (ch == '"' or ch == "'"):
            in_string = True
            string_char = ch
            result.append(ch)
            i += 1
            continue
        if in_string:
            result.append(ch)
            if ch == '\\' and i + 1 < n:
                i += 1
                result.append(text[i])
            elif ch == string_char:
                in_string = False
            i += 1
            continue

        if ch == '/' and i + 1 < n:
            nxt = text[i + 1]
            if nxt == '/':
                i += 2
                while i < n and text[i] != '\n':
                    i += 1
                continue
            elif nxt == '*':
                i += 2
                while i < n - 1 and not (text[i] == '*' and text[i + 1] == '/'):
                    i += 1
                i += 2
                continue
        result.append(ch)
        i += 1

    return ''.join(result)


def load_jsonc(path: Path) -> dict:
    """加载 JSONC 或 JSON 文件，返回 dict。"""
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    cleaned = _strip_jsonc_comments(raw)
    return json.loads(cleaned)


def extract_labels(meta: dict) -> Dict[str, str]:
    """从元数据 dict 提取三级分类标签。

    PulseCom: L1/L2 由 signal_category 映射, L3 = signal_type
    Ship:     L1/L2 由 signal_category 映射, L3 = sub_type

    L1 返回形容词形式 (active/passive) 以便在 prompt 模板中灵活使用。
    """
    signal_category = meta.get("signal_category", "unknown")
    signal_type = meta.get("signal_type", "unknown")
    sub_type = meta.get("sub_type")

    if signal_category in ("pulse", "communication"):
        l1 = "actively transmitted"
        l2 = "detection pulse" if signal_category == "pulse" else "communication signal"
    elif signal_category == "radiated_noise":
        l1 = "source-radiated"
        l2 = "ship-radiated noise"
    else:
        l1 = "unknown"
        l2 = "unknown"

    l3 = sub_type or signal_type

    l3_display_map = {
        "CW": "CW (Continuous Wave)",
        "LFM": "LFM (Linear Frequency Modulation)",
        "HFM": "HFM (Hyperbolic Frequency Modulation)",
        "2FSK": "2FSK (Binary Frequency Shift Keying)",
        "4FSK": "4FSK (Quaternary Frequency Shift Keying)",
        "BPSK": "BPSK (Binary Phase Shift Keying)",
        "QPSK": "QPSK (Quadrature Phase Shift Keying)",
        "OFDM": "OFDM (Orthogonal Frequency Division Multiplexing)",
        "cargo": "Cargo vessel",
        "cruise": "Cruise ship",
        "fishing": "Fishing vessel",
        "warship": "Naval vessel",
        "underwater_target": "Underwater vehicle",
    }
    l3_display = l3_display_map.get(l3, l3)

    return {"L1": l1, "L2": l2, "L3": l3, "L3_display": l3_display}


def _normalize_number(val, *, positive: bool = False, nonnegative: bool = False) -> Optional[float]:
    """Convert a scalar-like value to a finite float or return None."""
    if val is None:
        return None
    if isinstance(val, (int, float, str)) and not isinstance(val, bool):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        if positive and v <= 0.0:
            return None
        if nonnegative and v < 0.0:
            return None
        return v
    return None


def _normalize_water_depth(val) -> Optional[float]:
    """归一化 water_depth_m: None/null/[]/{}/NaN/Inf/非正数 → None。"""
    return _normalize_number(val, positive=True)


def get_wav_path(meta: dict) -> str:
    return meta.get("wav_path", "")


def get_id(meta: dict) -> str:
    return meta.get("id", "unknown")


def get_center_freq(meta: dict) -> Optional[float]:
    """提取用于 BELLHOP 运行的代表频率。"""
    sp = meta.get("signal_params", {}) or {}
    if not isinstance(sp, dict):
        return None

    # Pulse/chirp metadata uses center_freq_hz, while digital modulation
    # metadata uses carrier_freq_hz. Both are preferable to band estimates.
    for key in ("center_freq_hz", "carrier_freq_hz"):
        value = sp.get(key)
        if value is None:
            continue
        try:
            frequency = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(frequency) and frequency > 0.0:
            return frequency

    # Prefer the occupied signal band over the broader nominal subband.
    for low_key, high_key in (
        ("band_low_hz", "band_high_hz"),
        ("subband_low_hz", "subband_high_hz"),
    ):
        low = sp.get(low_key)
        high = sp.get(high_key)
        if low is None or high is None:
            continue
        try:
            low = float(low)
            high = float(high)
        except (TypeError, ValueError):
            continue
        if math.isfinite(low) and math.isfinite(high) and 0.0 <= low < high:
            return (low + high) / 2.0
    return None


def get_geometry(meta: dict) -> Dict[str, Optional[float]]:
    """提取几何参数 (tx_depth, rx_depth, range_m, water_depth_m)。"""
    tx = _normalize_number(meta.get("source_depth_m"), nonnegative=True)
    rx = _normalize_number(meta.get("receiver_depth_m"), nonnegative=True)
    rg = _normalize_number(meta.get("range_m"), positive=True)
    wd = _normalize_water_depth(meta.get("water_depth_m"))

    geo = meta.get("geometry", {}) or {}
    if not isinstance(geo, dict):
        geo = {}
    if tx is None:
        tx = _normalize_number(geo.get("tx_depth_m"), nonnegative=True)
    if rx is None:
        rx = _normalize_number(geo.get("rx_depth_m"), nonnegative=True)
    if rg is None:
        rg = _normalize_number(geo.get("range_m"), positive=True)
    if wd is None:
        wd = _normalize_water_depth(geo.get("water_depth_m"))

    return {
        "tx_depth_m": tx,
        "rx_depth_m": rx,
        "range_m": rg,
        "water_depth_m": wd,
    }
