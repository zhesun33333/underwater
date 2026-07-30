"""
层级分类指标计算引擎。

指标:
  - L1/L2/L3 Accuracy, Per-class P/R/F1, Macro F1, Confusion Matrix
  - L2-given-L1, L3-given-L2, Hierarchical Joint Accuracy
  - 推理质量: Domain Term Match Rate
"""

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
    source_stratified: Dict[str, dict] = field(default_factory=dict)  # PulseCom_TL / Ship_SNR 分源分层
    per_class_snr: Dict[str, dict] = field(default_factory=dict)      # 逐类 SNR 退化


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
    #
    # 每类只定义自己的区分性术语 (should)。
    # should_not 自动推导为: 全局术语集合 _ALL_L3_TERMS 中不属于本类的所有术语。
    # 设计原则: 术语必须具有 L3 类别区分力, 不包含 L1/L2 层面的通用描述。
    # ================================================================
    _L3_SHOULD = {
        # ── 探测脉冲类 ──
        # 与 pipeline_step2_qa.py / pipeline_ship_step2_qa.py 的 _L3_TERMS 保持同步
        "CW":  {"单频", "连续波", "音调不变", "单音调", "频率集中"},
        "LFM": {"线性调频", "线性扫频", "音调线性", "频率滑移",
                "频带展宽", "由低变高", "由高变低"},
        "HFM": {"双曲调频", "双曲扫频", "非线性", "先急后缓",
                "先快后慢"},
        # ── 通信类 ──
        "2FSK": {"两个音调", "二元调制", "频移键控", "频率跳变",
                 "音调切换"},
        "4FSK": {"四个音调", "四进制", "频移键控", "频率跳变",
                 "多音调交替"},
        "BPSK": {"相位翻转", "相移键控", "二进制", "两个状态",
                 "顿挫感", "音量稳定"},
        "QPSK": {"相移键控", "四个状态", "四进制",
                 "多状态切换", "响度恒定"},
        "OFDM": {"子载波", "多载波", "正交", "频分复用",
                 "并行传输", "密集子载波"},
        # ── 舰船辐射噪声 ──
        "cargo":   {"谐波结构清晰", "轴频", "轴频节律", "低速大型",
                    "大型商船", "规律节律", "谐音丰富"},
        "cruise":  {"机械噪声密集", "密集机械", "高速", "多机组",
                    "宽带噪声", "中高频段", "客船特征"},
        "fishing": {"小型", "结构简单", "稀疏谐音", "柴油机",
                    "音量较小", "低次谐音", "谐音有限"},
        "warship": {"强劲", "大功率", "军用舰船", "多轴推进",
                    "多组谐波", "复杂密集", "能量集中低频"},
        "underwater_target": {"水下目标", "平滑均匀", "音量低",
                              "宽带为主", "谐音稀少", "缺乏节律"},
    }

    # 全局 L3 区分性术语集合 (自动聚合)
    _ALL_L3_TERMS = set()
    for _s in _L3_SHOULD.values():
        _ALL_L3_TERMS.update(_s)

    def _get_should_not(self, l3: str) -> set:
        """自动推导: should_not = 全局集合 - 本类术语。"""
        own = self._L3_SHOULD.get(l3, set())
        return self._ALL_L3_TERMS - own

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

            should_hits = [t for t in should_set if t in text]
            should_not_hits = [t for t in should_not_set if t in text]

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
    # 按数据来源分层: 三分位 (tertile) 动态切分
    # ============================================================

    @staticmethod
    def compute_source_stratified(results: list) -> dict:
        """按数据来源分层评估。

        PulseCom 按 TL 三分位切, Ship 按 SNR 三分位切。
        不再使用固定阈值 —— 根据测试集实际分布动态划分，保证每个 bin 样本量大致相等。
        """
        pulsecom = [r for r in results if r.sample.gt["L1"] == "active"]
        ship = [r for r in results if r.sample.gt["L1"] == "passive"]

        def _tertile_bins(group, metric_key, unit):
            """将 group 按 metric 值升序切为 3 个等大 bin。"""
            # 收集所有有效质量值并排序
            vals_with_idx = []
            for i, r in enumerate(group):
                v = r.sample.metadata.get(metric_key)
                if v is not None:
                    vals_with_idx.append((v, i))
            if len(vals_with_idx) < 3:
                return []

            vals_with_idx.sort(key=lambda x: x[0])
            values = [v for v, _ in vals_with_idx]
            n = len(values)
            t1_val = values[n // 3]
            t2_val = values[2 * n // 3]

            # 三个区间的实际值范围
            lo_vals = [v for v in values if v <= t1_val]
            mid_vals = [v for v in values if t1_val < v <= t2_val]
            hi_vals = [v for v in values if v > t2_val]

            bins = []
            for label, fn, sub_vals in [
                (f"Low  ({lo_vals[0]:.1f} ~ {lo_vals[-1]:.1f} {unit})",
                 lambda v: v is not None and v <= t1_val, lo_vals),
                (f"Mid  ({mid_vals[0]:.1f} ~ {mid_vals[-1]:.1f} {unit})",
                 lambda v: v is not None and t1_val < v <= t2_val, mid_vals),
                (f"High ({hi_vals[0]:.1f} ~ {hi_vals[-1]:.1f} {unit})",
                 lambda v: v is not None and v > t2_val, hi_vals),
            ]:
                matched = [r for r in group
                           if fn(r.sample.metadata.get(metric_key, -999))]
                n_bin = len(matched)
                if n_bin == 0:
                    continue
                l3_ok = sum(1 for r in matched if r.turn2_pred.L3 == r.sample.gt["L3"])
                cascade = sum(1 for r in matched if r.cascade_skipped)
                bins.append({
                    "label": label, "count": n_bin,
                    "l3_acc": l3_ok / n_bin, "cascade_rate": cascade / n_bin,
                })
            return bins

        pc_bins = _tertile_bins(pulsecom, "snr_db", "dB")
        ship_bins = _tertile_bins(ship, "snr_db", "dB")

        return {
            "PulseCom_TL": {
                "name": "PulseCom  |  Transmission Loss TL  |  higher = cleaner",
                "count": len(pulsecom),
                "bins": pc_bins,
            },
            "Ship_SNR": {
                "name": "Ship  |  Line-spectrum SNR  |  higher = clearer",
                "count": len(ship),
                "bins": ship_bins,
            },
        }

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

    @staticmethod
    def compute_per_class_snr(results: list, class_keys: List[str]) -> dict:
        """逐类别 x SNR 区间交叉分析: 哪些类在低 SNR 下崩溃最快。

        Returns:
            {"per_bin": {"≥15dB": {"CW": {"count": N, "acc": 0.9}, ...}, ...},
             "degradation": {"CW": 0.17, "OFDM": 0.60, ...}}
        """
        bins = [
            ("≥15dB",  lambda s: s is not None and s >= 15),
            ("5-15dB", lambda s: s is not None and 5 <= s < 15),
            ("-5-5dB", lambda s: s is not None and -5 <= s < 5),
            ("≤-5dB", lambda s: s is not None and s < -5),
        ]

        per_bin: Dict[str, dict] = {}
        for label, fn in bins:
            group = [r for r in results if fn(r.sample.metadata.get("snr_db"))]
            bin_data = {}
            for cls in class_keys:
                cls_group = [r for r in group if r.sample.gt["L3"] == cls]
                n_cls = len(cls_group)
                if n_cls == 0:
                    continue
                correct = sum(1 for r in cls_group
                              if r.turn2_pred.L3 == cls)
                bin_data[cls] = {"count": n_cls, "acc": correct / n_cls}
            per_bin[label] = bin_data

        degradation = {}
        hi_bin = per_bin.get("≥15dB", {})
        lo_bin = per_bin.get("≤-5dB", {})
        for cls in class_keys:
            hi = hi_bin.get(cls, {}).get("acc")
            lo = lo_bin.get(cls, {}).get("acc")
            if hi is not None and lo is not None:
                degradation[cls] = round(hi - lo, 4)

        return {"per_bin": per_bin, "degradation": degradation}
