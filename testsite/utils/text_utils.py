"""中文文本归一化与单位解析工具。"""

import re


def normalize_chinese_text(text: str) -> str:
    """归一化中文文本：全角转半角、去除多余空白、统一标点。"""
    if not text:
        return ""
    # 全角数字/字母转半角
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:  # 全角空格
            result.append(" ")
        else:
            result.append(ch)
    text = "".join(result)
    # 去除多余空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_number_with_unit(text: str, default_unit: str = "Hz") -> float:
    """从文本中提取数值，支持常见单位后缀。

    示例:
        "8500 Hz" -> 8500.0
        "8.5k" -> 8500.0
        "8.5 kHz" -> 8500.0
        "0.2 s" -> 0.2
        "200 ms" -> 0.2
        "15 节" -> 15.0
        "3叶" -> 3.0
    """
    if not text:
        return None

    text = text.strip().lower().replace(" ", "")

    # 匹配数值（支持小数和科学计数法）
    num_match = re.match(r"([+-]?\d+\.?\d*(?:[ee][+-]?\d+)?)", text)
    if not num_match:
        return None

    value = float(num_match.group(1))
    suffix = text[num_match.end():]

    # 单位换算 (最长前缀匹配, 避免 "m" 抢先匹配到 "mhz")
    multipliers = {
        "khz": 1000, "kilo": 1000, "k": 1000,
        "mhz": 1e6,
        "ms": 0.001, "m": 0.001,
        "ghz": 1e9, "g": 1e9,
    }
    best_mult = 1.0
    best_len = 0
    for unit, mult in multipliers.items():
        if suffix == unit:
            best_mult = mult
            break
        elif suffix.startswith(unit) and len(unit) > best_len:
            best_mult = mult
            best_len = len(unit)
    value *= best_mult

    return value


def match_keyword_in_text(text: str, keywords: list) -> str:
    """在文本中匹配关键词，返回命中的第一个关键词对应的标准标签。

    keywords 格式: [("CW脉冲", "CW"), ("单频脉冲", "CW"), ...]
    第一个元素是匹配词，第二个是标准标签。
    """
    if not text:
        return None
    text_lower = text.lower()
    for pattern, label in keywords:
        if isinstance(pattern, str):
            if pattern.lower() in text_lower:
                return label
        else:  # compiled regex
            if pattern.search(text):
                return label
    return None


def extract_delimited_field(text: str, field_name: str) -> str:
    """从定界符格式的文本中提取字段值。

    支持格式:
        【字段名】: 值
        【字段名】:值
        【字段名】= 值
        字段名: 值（中英文冒号均支持）
    """
    # 格式1: 【字段名】: 值
    patterns = [
        rf"【{re.escape(field_name)}】\s*[:：=]?\s*(.+?)(?:\n|【|$)",
        rf"{re.escape(field_name)}\s*[:：=]\s*(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            # 移除尾部标点
            val = re.sub(r"[，。,\.\s]+$", "", val)
            return val
    return None


