"""
JSONC/JSON 解析 + 三级标签提取
处理 PulseCom (.jsonc) 和 Ship (.json) 两种格式
"""
import json
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
    """
    signal_category = meta.get("signal_category", "unknown")
    signal_type = meta.get("signal_type", "unknown")
    sub_type = meta.get("sub_type")

    if signal_category in ("pulse", "communication"):
        l1 = "主动信号"
        l2 = "探测脉冲类" if signal_category == "pulse" else "通信类"
    elif signal_category == "radiated_noise":
        l1 = "被动信号"
        l2 = "舰船辐射噪声"
    else:
        l1 = "未知"
        l2 = "未知"

    l3 = sub_type or signal_type

    l3_cn_map = {
        "CW": "CW（连续波）",
        "LFM": "LFM（线性调频）",
        "HFM": "HFM（双曲调频）",
        "2FSK": "2FSK（二进制频移键控）",
        "4FSK": "4FSK（四进制频移键控）",
        "BPSK": "BPSK（二进制相移键控）",
        "QPSK": "QPSK（四进制相移键控）",
        "OFDM": "OFDM（正交频分复用）",
        "cargo": "货船",
        "cruise": "邮轮",
        "fishing": "渔船",
        "warship": "军舰",
        "underwater_target": "水下目标",
    }
    l3_cn = l3_cn_map.get(l3, l3)

    return {"L1": l1, "L2": l2, "L3": l3, "L3_CN": l3_cn}


def _normalize_water_depth(val) -> Optional[float]:
    """归一化 water_depth_m: None/null/[]/{} → None, 数字 → float"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, list) and len(val) == 0:
        return None
    if isinstance(val, dict) and len(val) == 0:
        return None
    return None


def get_wav_path(meta: dict) -> str:
    return meta.get("wav_path", "")


def get_id(meta: dict) -> str:
    return meta.get("id", "unknown")


def get_center_freq(meta: dict) -> Optional[float]:
    """提取用于 BELLHOP 运行的代表频率。"""
    sp = meta.get("signal_params", {}) or {}
    cf = sp.get("center_freq_hz")
    if cf is not None:
        return float(cf)
    lo = sp.get("subband_low_hz") or sp.get("band_low_hz")
    hi = sp.get("subband_high_hz") or sp.get("band_high_hz")
    if lo and hi:
        return (float(lo) + float(hi)) / 2.0
    return None


def get_geometry(meta: dict) -> Dict[str, Optional[float]]:
    """提取几何参数 (tx_depth, rx_depth, range_m, water_depth_m)。"""
    tx = meta.get("source_depth_m")
    rx = meta.get("receiver_depth_m")
    rg = meta.get("range_m")
    wd = _normalize_water_depth(meta.get("water_depth_m"))

    geo = meta.get("geometry", {}) or {}
    if tx is None:
        tx = geo.get("tx_depth_m")
    if rx is None:
        rx = geo.get("rx_depth_m")
    if rg is None:
        rg = geo.get("range_m")
    if wd is None:
        wd = _normalize_water_depth(geo.get("water_depth_m"))

    return {
        "tx_depth_m": float(tx) if tx is not None else None,
        "rx_depth_m": float(rx) if rx is not None else None,
        "range_m": float(rg) if rg is not None else None,
        "water_depth_m": wd,
    }
