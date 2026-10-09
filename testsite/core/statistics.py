"""Confidence intervals for paired original-versus-silent comparisons.

本模块**不在** ``core/protocol.py`` 的 ``IMPLEMENTATION_FILES`` 清单里，
因此新增或修改它不会改变实现哈希，历史 protocol 分片继续有效。

口径（与 ``core/scorer.py`` 完全一致，改动 scorer 时必须同步这里）:
  * L3 Accuracy = 预测等于真值的样本数 / 总样本数。
    ``cascade_error`` / ``unknown`` 是**错误预测**，计入分母，不剔除。
  * Macro F1 = 对 ``support > 0`` 的类求 F1 的均值（无 support 的类不参与平均）。

设计要点:
  * 准确率区间用 Wilson score interval：闭式解，不依赖随机数，可复现，
    也用来交叉验证 bootstrap 的实现。
  * 差值区间与 Macro F1 区间用 **cluster bootstrap**：重抽样单位是 ``source_id``
    （即剥掉 ``_chN`` 后的声源），因为同一声源的不同声道高度相关，
    按样本做 i.i.d. 重抽样会低估方差、把区间算窄。
"""
import math
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

CONFIDENCE_LEVEL = 0.95
DEFAULT_REPLICATES = 10000
DEFAULT_SEED = 20261008
_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}

# 与 testsite/core/dataset_record.py 的 source_id 规则保持一致
_CHANNEL_SUFFIX = re.compile(r"_ch\d+$", re.IGNORECASE)


def source_id(sample_id: str) -> str:
    """剥掉声道后缀得到声源 id（ch0/ch1 归为同一簇）。"""
    return _CHANNEL_SUFFIX.sub("", str(sample_id))


def wilson_interval(successes: int, total: int, confidence: float = CONFIDENCE_LEVEL) -> Tuple[float, float]:
    """二项比例的 Wilson score 区间。闭式解，与随机种子无关。"""
    if total <= 0:
        raise ValueError("wilson_interval requires a positive sample count")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    z = _Z.get(round(confidence, 4))
    if z is None:  # pragma: no cover - 只在传入非常规置信水平时触发
        z = math.sqrt(2.0) * _erf_inv(2.0 * confidence - 1.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _erf_inv(x: float) -> float:  # pragma: no cover - 仅非常规置信水平使用
    """二分法求 erf 的反函数，用于计算任意置信水平的 z 值。"""
    lo, hi = -3.0, 3.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if math.erf(mid) < x:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def encode(labels: Sequence[str], class_keys: Sequence[str]) -> np.ndarray:
    """把标签编码成下标；不在类别表内的值（如 cascade_error）编码为 -1。"""
    index = {key: i for i, key in enumerate(class_keys)}
    return np.array([index.get(x, -1) for x in labels], dtype=np.int64)


def _joint_counts(pred_codes: np.ndarray, true_codes: np.ndarray, n_classes: int,
                  weights: np.ndarray = None) -> np.ndarray:
    """返回 (n_classes+1)×(n_classes+1) 联合计数矩阵，行=真值，列=预测。

    最后一列收纳"预测不在类别表内"的情况，因此 cascade_error 会被正确记为漏检。
    """
    stride = n_classes + 1
    pred_eff = np.where(pred_codes < 0, n_classes, pred_codes)
    flat = np.asarray(true_codes, dtype=np.int64) * stride + np.asarray(pred_eff, dtype=np.int64)
    counts = np.bincount(flat, weights=weights, minlength=stride * stride)
    return counts.reshape(stride, stride).astype(np.float64)


def accuracy_and_macro_f1(counts: np.ndarray, n_classes: int) -> Tuple[float, float]:
    """从联合计数矩阵同时算出 accuracy 与 macro F1，口径复刻 core/scorer.py。"""
    total = counts.sum()
    if total <= 0:
        return 0.0, 0.0
    tp = np.diag(counts)[:n_classes]
    predicted = counts[:, :n_classes].sum(axis=0)   # tp + fp
    support = counts[:n_classes, :].sum(axis=1)     # tp + fn
    accuracy = float(tp.sum() / total)
    denominator = predicted + support
    f1 = np.divide(2.0 * tp, denominator, out=np.zeros(n_classes), where=denominator > 0)
    # core/scorer.py 先把每个类别的 F1 取 4 位小数、再对 support>0 的类求均值。
    # 这里必须复刻该取整，否则区间中心会与已公布的点估计差约 1e-7。
    f1 = np.round(f1, 4)
    present = support > 0
    macro = float(f1[present].mean()) if present.any() else 0.0
    return accuracy, macro


def cluster_membership(sample_ids: Sequence[str]) -> Tuple[np.ndarray, int]:
    """返回 (每条样本的簇下标, 簇个数)。簇 = source_id。"""
    keys = [source_id(s) for s in sample_ids]
    unique = {}
    codes = np.empty(len(keys), dtype=np.int64)
    for i, key in enumerate(keys):
        codes[i] = unique.setdefault(key, len(unique))
    return codes, len(unique)


def cluster_bootstrap(pred_a: np.ndarray, pred_b: np.ndarray, true_codes: np.ndarray,
                      n_classes: int, cluster_codes: np.ndarray, n_clusters: int,
                      replicates: int = DEFAULT_REPLICATES, seed: int = DEFAULT_SEED,
                      confidence: float = CONFIDENCE_LEVEL) -> Dict[str, Dict[str, float]]:
    """按 source_id 整簇重抽样，给出两个条件及其差值的百分位区间。

    返回 accuracy 与 macro_f1 两组，每组含 original / silent / delta 的
    点估计与 [lo, hi] 区间。delta = original − silent（可为负）。
    """
    if pred_a.shape != pred_b.shape or pred_a.shape != true_codes.shape:
        raise ValueError("prediction and ground-truth arrays must have equal length")
    if len(pred_a) == 0:
        raise ValueError("no paired samples to resample")
    if n_clusters <= 0:
        raise ValueError("cluster count must be positive")
    if replicates < 2:
        raise ValueError("replicates must be at least 2")

    stride = n_classes + 1
    flat_a = np.asarray(true_codes, dtype=np.int64) * stride + np.where(pred_a < 0, n_classes, pred_a)
    flat_b = np.asarray(true_codes, dtype=np.int64) * stride + np.where(pred_b < 0, n_classes, pred_b)

    order = np.argsort(cluster_codes, kind="stable")
    sorted_a, sorted_b = flat_a[order], flat_b[order]
    sizes = np.bincount(cluster_codes, minlength=n_clusters)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)

    rng = np.random.default_rng(seed)
    acc_a = np.empty(replicates)
    acc_b = np.empty(replicates)
    f1_a = np.empty(replicates)
    f1_b = np.empty(replicates)
    for i in range(replicates):
        chosen = rng.integers(0, n_clusters, n_clusters)
        picked_sizes = sizes[chosen]
        total = int(picked_sizes.sum())
        offsets = np.repeat(starts[chosen], picked_sizes)
        within = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(picked_sizes) - picked_sizes, picked_sizes)
        sel = offsets + within
        counts_a = np.bincount(sorted_a[sel], minlength=stride * stride).reshape(stride, stride).astype(np.float64)
        counts_b = np.bincount(sorted_b[sel], minlength=stride * stride).reshape(stride, stride).astype(np.float64)
        acc_a[i], f1_a[i] = accuracy_and_macro_f1(counts_a, n_classes)
        acc_b[i], f1_b[i] = accuracy_and_macro_f1(counts_b, n_classes)

    lo_q, hi_q = 100 * (1 - confidence) / 2, 100 * (1 + confidence) / 2

    point_acc_a = _point_accuracy(flat_a, n_classes, stride)
    point_acc_b = _point_accuracy(flat_b, n_classes, stride)
    point_f1_a = _point_f1(flat_a, n_classes, stride)
    point_f1_b = _point_f1(flat_b, n_classes, stride)

    def block(point_a, point_b, boot_a, boot_b):
        delta_boot = boot_a - boot_b
        return {
            "original": {"estimate": float(point_a), "ci95": [float(np.percentile(boot_a, lo_q)),
                                                              float(np.percentile(boot_a, hi_q))]},
            "silent": {"estimate": float(point_b), "ci95": [float(np.percentile(boot_b, lo_q)),
                                                            float(np.percentile(boot_b, hi_q))]},
            "delta": {"estimate": float(point_a - point_b),
                      "ci95": [float(np.percentile(delta_boot, lo_q)),
                               float(np.percentile(delta_boot, hi_q))]},
        }

    return {"accuracy": block(point_acc_a, point_acc_b, acc_a, acc_b),
            "macro_f1": block(point_f1_a, point_f1_b, f1_a, f1_b)}


def _point_accuracy(flat: np.ndarray, n_classes: int, stride: int) -> float:
    counts = np.bincount(flat, minlength=stride * stride).reshape(stride, stride).astype(np.float64)
    return accuracy_and_macro_f1(counts, n_classes)[0]


def _point_f1(flat: np.ndarray, n_classes: int, stride: int) -> float:
    counts = np.bincount(flat, minlength=stride * stride).reshape(stride, stride).astype(np.float64)
    return accuracy_and_macro_f1(counts, n_classes)[1]


def discordant_counts(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict[str, int]:
    """配对二分类的不一致对计数，用于 McNemar 式交叉验证。"""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    return {"original_only": int(np.sum(a & ~b)), "silent_only": int(np.sum(~a & b)),
            "both_correct": int(np.sum(a & b)), "neither_correct": int(np.sum(~a & ~b))}


def paired_difference_interval(counts: Dict[str, int], confidence: float = CONFIDENCE_LEVEL) -> Tuple[float, float]:
    """配对比例差值的解析区间（McNemar 风格），用于交叉验证 bootstrap。

    d = (b − c) / n，SE = sqrt(b + c − (b − c)² / n) / n
    """
    b, c = counts["original_only"], counts["silent_only"]
    n = sum(counts.values())
    if n <= 0:
        raise ValueError("no paired samples")
    z = _Z.get(round(confidence, 4), _Z[CONFIDENCE_LEVEL])
    diff = (b - c) / n
    variance = (b + c - (b - c) ** 2 / n) / (n * n)
    margin = z * math.sqrt(max(0.0, variance))
    return diff - margin, diff + margin


def summarise(paired_rows: List[dict], class_keys: Sequence[str],
              replicates: int = DEFAULT_REPLICATES, seed: int = DEFAULT_SEED,
              confidence: float = CONFIDENCE_LEVEL) -> Dict[str, object]:
    """从 compare_silent_control 产出的配对行直接算出全部区间。"""
    if not paired_rows:
        raise ValueError("no paired rows")
    sample_ids = [r["sample_id"] for r in paired_rows]
    gt = [r["gt"]["L3"] for r in paired_rows]
    pred_a = [r["original"]["L3"] for r in paired_rows]
    pred_b = [r["silent"]["L3"] for r in paired_rows]

    true_codes = encode(gt, class_keys)
    codes_a, codes_b = encode(pred_a, class_keys), encode(pred_b, class_keys)
    cluster_codes, n_clusters = cluster_membership(sample_ids)

    correct_a = codes_a == true_codes
    correct_b = codes_b == true_codes
    n = len(paired_rows)
    disc = discordant_counts(correct_a, correct_b)

    bootstrap = cluster_bootstrap(codes_a, codes_b, true_codes, len(class_keys),
                                  cluster_codes, n_clusters, replicates=replicates,
                                  seed=seed, confidence=confidence)
    analytic = paired_difference_interval(disc, confidence)

    for key, successes in (("original", int(correct_a.sum())), ("silent", int(correct_b.sum()))):
        lo, hi = wilson_interval(successes, n, confidence)
        bootstrap["accuracy"][key]["wilson_ci95"] = [lo, hi]
        bootstrap["accuracy"][key]["correct"] = successes
    bootstrap["accuracy"]["delta"]["analytic_ci95"] = [analytic[0], analytic[1]]
    bootstrap["accuracy"]["delta"]["discordant_pairs"] = disc

    bootstrap["method"] = {
        "accuracy": "wilson_score (closed form, seed-independent)",
        "difference": "cluster bootstrap percentile",
        "macro_f1": "cluster bootstrap percentile",
        "resampling_unit": "source_id (channel variants of one source resampled together)",
        "confidence_level": confidence,
        "replicates": replicates,
        "seed": seed,
        "rng": "numpy.random.default_rng (PCG64)",
    }
    bootstrap["sample_structure"] = {
        "samples": n,
        "clusters": int(n_clusters),
        "max_cluster_size": int(np.bincount(cluster_codes).max()),
    }
    bootstrap["note"] = ("Intervals describe sampling variability over this fixed diagnostic subset. "
                         "They do not cover decoding stochasticity and must not be read as coverage "
                         "over channel or sea-state conditions.")
    return bootstrap
