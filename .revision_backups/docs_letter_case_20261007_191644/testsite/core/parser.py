"""Strict letter-only classification parsing.

Model responses must contain exactly one displayed uppercase option letter.
Aliases are used only to decode trusted option text, never model prose.
Invalid responses produce unknown labels and parse_status=invalid_format.
"""

import re
from typing import Dict, List, Tuple

from .scorer import HierPrediction


# ============================================================
# 选项解析 — 将模型输出的短选项标识符还原为完整文本
# ============================================================

def resolve_short_answer(output: str, prompt: str) -> str:
    """Resolve one uppercase option letter; reject all other response forms.

    Only surrounding whitespace is ignored. An empty result means invalid
    format or an option that is absent from the supplied question.
    """
    answer = output.strip()
    if not re.fullmatch(r"[A-Z]", answer):
        return ""
    options = dict(re.findall(r"^\s*([A-Z])\. +([^\r\n]+)", prompt, re.MULTILINE))
    return options.get(answer, "")


class OutputParser:
    """层级分类输出解析器。"""

    def __init__(self, config: dict):
        tax = config["taxonomy"]
        self.l1_map = self._build_alias_map(tax["L1"]["classes"])
        self.l2_map = self._build_alias_map(tax["L2"]["classes"])
        self.l3_map = self._build_alias_map(tax["L3"]["classes"])

    @staticmethod
    def _build_alias_map(classes: List[dict]) -> List[Tuple[str, str]]:
        """构建 (alias, key) 列表, 按 alias 长度降序 (长匹配优先)。"""
        pairs = []
        for c in classes:
            for alias in c["aliases"]:
                pairs.append((alias, c["key"]))
        pairs.sort(key=lambda x: len(x[0]), reverse=True)
        return pairs

    # ============================================================
    # 主解析入口
    # ============================================================

    def parse(self, sample_id: str, model_output: str,
              prompt: str = "") -> HierPrediction:
        """从模型输出中提取 L1/L2/L3。

        Args:
            sample_id: 样本标识
            model_output: 模型原始输出
            prompt: 发送给模型的 prompt (用于解析选项字母)
        """
        # 选项解析: 如果模型只输出 "A" 这样的短标识符, 还原为完整文本
        original_output = model_output
        text = resolve_short_answer(model_output, prompt)

        l1, l1_tier = self._extract(text, self.l1_map)
        l2, l2_tier = self._extract(text, self.l2_map)
        l3, l3_tier = self._extract(text, self.l3_map)

        parse_tier = max(l1_tier, l2_tier, l3_tier)

        return HierPrediction(
            sample_id=sample_id,
            L1=l1, L2=l2, L3=l3,
            parse_tier=parse_tier,
            raw_output=original_output,
            parse_status="valid_option" if text else "invalid_format",
        )

    def parse_turn1(self, sample_id: str, output: str,
                    prompt: str = "") -> HierPrediction:
        """Turn1 输出 — 仅提取 L1 (避免浪费 L2/L3 提取)。"""
        original_output = output
        text = resolve_short_answer(output, prompt)
        l1, l1_tier = self._extract(text, self.l1_map)
        return HierPrediction(
            sample_id=sample_id,
            L1=l1, L2="unknown", L3="unknown",
            parse_tier=l1_tier,
            raw_output=original_output,
            parse_status="valid_option" if text else "invalid_format",
        )

    # ============================================================
    # 核心匹配
    # ============================================================

    def _extract(
        self, text: str, alias_map: List[Tuple[str, str]]
    ) -> Tuple[str, int]:
        """两轮回退匹配 → (class_key, parse_tier)。"""
        text_lower = text.lower()

        # Tier 1: alias 直接匹配
        for alias, key in alias_map:
            if alias.lower() in text_lower:
                return key, 1

        # Tier 2: 去除非字母数字后宽松匹配
        text_clean = re.sub(r'[^\w]', '', text_lower)
        for alias, key in alias_map:
            alias_clean = re.sub(r'[^\w]', '', alias.lower())
            if alias_clean and alias_clean in text_clean:
                return key, 2

        return "unknown", 3


# ============================================================
# Ground Truth 提取 (从 JSON 元数据)
# ============================================================

def extract_l1_l2_l3_from_meta(meta: dict) -> Dict[str, str]:
    """
    从 JSON 元数据中提取 L1/L2/L3 ground truth。

    PulseCom:
      signal_category = "pulse" → L1="active", L2="pulse"
      signal_category = "communication" → L1="active", L2="communication"
      signal_type = "CW"/"LFM"/... → L3
    Ship:
      signal_category = "radiated_noise" → L1="passive", L2="ship_noise"
      sub_type = "cargo"/"cruise"/... → L3
    """
    cat = meta.get("signal_category", "")

    if cat in ("pulse", "communication"):
        l1 = "active"
        l2 = "pulse" if cat == "pulse" else "communication"
    elif cat == "radiated_noise":
        l1 = "passive"
        l2 = "ship_noise"
    else:
        return {"L1": "unknown", "L2": "unknown", "L3": "unknown"}

    # L3: 优先 sub_type, 其次 signal_type, 最后 signal_params.original_class_name
    l3 = (meta.get("sub_type") or meta.get("signal_type")
          or (meta.get("signal_params") or {}).get("original_class_name", "unknown"))
    l3 = str(l3).lower().strip()

    # ship 的 class name → L3 key 映射
    ship_l3_map = {
        "cargo_ship": "cargo", "cruise_ship": "cruise",
        "fishing_boat": "fishing", "warship": "warship",
        "underwater_target": "underwater_target",
    }
    l3 = ship_l3_map.get(l3, l3)

    # PulseCom L3 规范化 (已经是 CW/LFM/2FSK 等, 直接大写)
    if cat in ("pulse", "communication"):
        l3 = l3.upper()

    return {"L1": l1, "L2": l2, "L3": l3}
