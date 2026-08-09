"""
层级分类指标计算引擎。

指标:
  - L1/L2/L3 Accuracy, Per-class P/R/F1, Macro F1, Confusion Matrix
  - L2-given-L1, L3-given-L2, Hierarchical Joint Accuracy
  - 推理质量: Domain Term Match Rate
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ============================================================
# 预测数据结构
# ============================================================

@dataclass
class HierPrediction:
    """单条样本的层级分类预测。"""
    sample_id: str
    L1: str                      # "active" | "passive" | "unknown"
    L2: str                      # "pulse" | "communication" | "ship_noise" | "unknown"
    L3: str                      # "CW" | "LFM" | ... | "unknown"
    parse_tier: int              # 1=直接命中, 2=关键词回溯, 3=失败
    raw_output: str = ""         # 模型原始输出文本


# ============================================================
# 指标数据结构
# ============================================================

@dataclass
class LevelMetrics:
    """单个分类层级的指标。"""
    level: str                   # "L1" | "L2" | "L3"
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    macro_f1: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)
    confusion_labels: List[str] = field(default_factory=list)


@dataclass
class HierarchicalMetrics:
    """完整的层级分类评估结果。"""
    total_samples: int = 0
    l1: Optional[LevelMetrics] = None
    l2: Optional[LevelMetrics] = None
    l3: Optional[LevelMetrics] = None
    l2_given_l1: float = 0.0
    l3_given_l2: float = 0.0
    joint_accuracy: float = 0.0
    parse_tier_dist: Dict[int, int] = field(default_factory=dict)


@dataclass
class ReasoningMetrics:
    """推理质量指标。"""
    num_samples: int
    cascade_skipped: int = 0               # 因T1错误跳过T3的样本数
    # 术语-类别对齐
    alignment_rate: float = 0.0            # should_appear≥1 且 should_not=0 的比例
    contradiction_rate: float = 0.0        # should_not≥1 (说了别类的话)
    # 错误分类体系
    concept_confusion_rate: float = 0.0    # 概念混淆: 用别类的特征描述本类
    vague_rate: float = 0.0                # 空洞泛化: 无区分性术语
    term_stacking_rate: float = 0.0        # 术语堆砌: 正负术语同时出现


# ============================================================
# 主计算器
# ============================================================

class Scorer:
    """层级分类指标计算器。"""

    def __init__(self, config: dict):
        tax = config["taxonomy"]

        # 构建 class key → name 的映射
        self.class_names = {}
        for level in ["L1", "L2", "L3"]:
            for c in tax[level]["classes"]:
                self.class_names[c["key"]] = c["name"]

        # L1/L2/L3 的 class keys 列表 (按 config 顺序)
        self.l1_keys = [c["key"] for c in tax["L1"]["classes"]]
        self.l2_keys = [c["key"] for c in tax["L2"]["classes"]]
        self.l3_keys = [c["key"] for c in tax["L3"]["classes"]]



    def compute_level(
        self, level: str, class_keys: List[str],
        y_pred: List[str], y_true: List[str],
    ) -> LevelMetrics:
        """计算单个分类层级的指标。"""
        n = len(y_pred)
        lm = LevelMetrics(level=level, total=n, correct=0)

        # Accuracy
        lm.correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
        lm.accuracy = lm.correct / n if n > 0 else 0.0

        # Per-class
        for cls in class_keys:
            tp = sum(1 for p, t in zip(y_pred, y_true) if p == cls and t == cls)
            fp = sum(1 for p, t in zip(y_pred, y_true) if p == cls and t != cls)
            fn = sum(1 for p, t in zip(y_pred, y_true) if p != cls and t == cls)
            support = sum(1 for t in y_true if t == cls)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            name = self.class_names.get(cls, cls)
            lm.per_class[name] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": support,
            }

        # Macro F1 (只算有 support 的类)
        f1s = [m["f1"] for m in lm.per_class.values() if m["support"] > 0]
        lm.macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        # Confusion matrix (extra row/col for special labels if present)
        all_keys = list(class_keys)
        special_keys = []
        for label in ("unknown", "cascade_error"):
            if any(p == label for p in y_pred) or any(t == label for t in y_true):
                special_keys.append(label)
                all_keys.append(label)
        key_idx = {k: i for i, k in enumerate(all_keys)}
        cm = [[0] * len(all_keys) for _ in range(len(all_keys))]
        for p, t in zip(y_pred, y_true):
            pi = key_idx.get(p, -1)
            ti = key_idx.get(t, -1)
            if ti >= 0 and pi >= 0:
                cm[ti][pi] += 1
        lm.confusion_matrix = cm
        lm.confusion_labels = [self.class_names.get(k, k) for k in all_keys]

        return lm

    # ============================================================
    # 第四层: 推理质量
    # ============================================================

    # ================================================================
    # L3 术语-类别对齐矩阵
    # 从 shared_terminology.py 统一导入 (single source of truth)
    # ================================================================
    from shared_terminology import L3_SHOULD as _L3_SHOULD, get_should_not as _get_should_not
    _get_should_not = staticmethod(_get_should_not)

    _ALL_L3_TERMS = set()
    for _s in _L3_SHOULD.values():
        _ALL_L3_TERMS.update(_s)

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        """Match a phrase case-insensitively without matching inside words."""
        pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
        return re.search(pattern, text.lower()) is not None

    def compute_reasoning_quality(
        self,
        reasoning_texts: List[str],
        predicted_l3: List[str] = None,
    ) -> ReasoningMetrics:
        """评估推理文本质量: 术语-类别对齐 + 错误分类。"""
        n = len(reasoning_texts)
        if n == 0:
            return ReasoningMetrics(num_samples=0)

        if predicted_l3 is None:
            predicted_l3 = [""] * n

        aligned = 0
        contradicted = 0
        confused = 0
        vague = 0
        stacked = 0

        for text, l3 in zip(reasoning_texts, predicted_l3):
            should_set = self._L3_SHOULD.get(l3, set())
            should_not_set = self._get_should_not(l3)

            should_hits = [t for t in should_set if self._contains_term(text, t)]
            should_not_hits = [t for t in should_not_set if self._contains_term(text, t)]

            has_positive = len(should_hits) >= 1
            has_negative = len(should_not_hits) >= 1

            if has_positive and not has_negative:
                aligned += 1
            if has_negative:
                contradicted += 1

            if has_positive and has_negative:
                stacked += 1
            elif has_negative:
                confused += 1
            elif not has_positive and not has_negative:
                vague += 1

        return ReasoningMetrics(
            num_samples=n,
            alignment_rate=aligned / n,
            contradiction_rate=contradicted / n,
            concept_confusion_rate=confused / n,
            vague_rate=vague / n,
            term_stacking_rate=stacked / n,
        )

    # ============================================================
    # Bootstrap 置信区间
    # ============================================================

    @staticmethod
    def bootstrap_ci(results: list, metric_fn, n_iter: int = 1000,
                     ci_level: float = 0.95) -> tuple:
        """Bootstrap 百分位置信区间。

        Args:
            results: TurnResult 列表
            metric_fn: callable(results) → float
            n_iter: bootstrap 迭代次数
            ci_level: 置信水平 (默认 0.95)

        Returns:
            (lower, upper) 或 (None, None) 如果样本不足
        """
        import random as _random
        n = len(results)
        if n < 5:
            return (None, None)

        rng = _random.Random(42)
        values = []
        for _ in range(n_iter):
            sample = [results[rng.randint(0, n - 1)] for _ in range(n)]
            try:
                values.append(metric_fn(sample))
            except (ZeroDivisionError, ValueError):
                continue

        if len(values) < n_iter * 0.9:
            return (None, None)

        values.sort()
        tail = (1.0 - ci_level) / 2.0
        lo = values[int(len(values) * tail)]
        hi = values[int(len(values) * (1.0 - tail))]
        return (lo, hi)
