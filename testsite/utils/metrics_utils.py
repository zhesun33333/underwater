"""数值比较与容错工具。"""

import math


def tolerance_compare(predicted: float, ground_truth: float,
                      tolerance_pct: float) -> bool:
    """判断预测值是否在真实值的 tolerance_pct 范围内。

    Args:
        predicted: 模型预测的数值
        ground_truth: 真实数值
        tolerance_pct: 容差百分比，如 10 表示 ±10%

    Returns:
        True 如果 |pred - truth| / truth <= tolerance_pct / 100
    """
    if ground_truth == 0:
        return abs(predicted) < 1e-6
    relative_error = abs(predicted - ground_truth) / abs(ground_truth)
    return relative_error <= tolerance_pct / 100.0


def compute_mae(predictions: list, ground_truths: list) -> float:
    """Mean Absolute Error。"""
    if not predictions:
        return float("nan")
    errors = [abs(p - t) for p, t in zip(predictions, ground_truths)]
    return sum(errors) / len(errors)


def compute_rmse(predictions: list, ground_truths: list) -> float:
    """Root Mean Square Error。"""
    if not predictions:
        return float("nan")
    errors = [(p - t) ** 2 for p, t in zip(predictions, ground_truths)]
    return math.sqrt(sum(errors) / len(errors))


def compute_mape(predictions: list, ground_truths: list) -> float:
    """Mean Absolute Percentage Error (0-100)。"""
    if not predictions:
        return float("nan")
    total = 0.0
    count = 0
    for p, t in zip(predictions, ground_truths):
        if abs(t) < 1e-9:
            continue
        total += abs(p - t) / abs(t) * 100.0
        count += 1
    return total / count if count > 0 else float("nan")


def compute_tolerance_accuracy(predictions: list, ground_truths: list,
                               tolerance_pct: float) -> float:
    """计算在容差范围内的准确率 (0-1)。"""
    if not predictions:
        return float("nan")
    correct = sum(1 for p, t in zip(predictions, ground_truths)
                  if tolerance_compare(p, t, tolerance_pct))
    return correct / len(predictions)


