"""Auditable result statistics; no model inference and no plotting on import."""
from __future__ import annotations

import math
import re

import numpy as np

from testsite.reporting.figure_data import CLASS_ORDER, CLASS_TO_FAMILY

PREDICTION_ORDER = CLASS_ORDER + ["unknown", "cascade_error"]
DISPLAY_NAMES = CLASS_ORDER[:8] + ["Cargo", "Cruise", "Fishing", "Naval", "Underwater"]
STRICT_STATES = ["T1 invalid", "T1 wrong", "T2 invalid", "T2 leaf wrong", "Leaf correct"]
LEGACY_STATES = ["T1 wrong / unresolved", "T2 leaf wrong / unresolved", "Leaf correct"]


def label_key(value: str, aliases: dict | None = None) -> str:
    """Recognize saved taxonomy labels explicitly; never drop unmatched labels."""
    if not isinstance(value, str):
        raise ValueError(f"Non-string class label: {value!r}")
    if aliases and value in aliases:
        value = aliases[value]
    if value in PREDICTION_ORDER:
        return value
    names = {"cargo vessel": "cargo", "cargo ship": "cargo", "货船": "cargo",
             "cruise ship": "cruise", "邮轮": "cruise", "fishing vessel": "fishing",
             "fishing boat": "fishing", "渔船": "fishing", "naval vessel": "warship",
             "军舰": "warship", "underwater vehicle": "underwater_target",
             "underwater target": "underwater_target", "水下目标": "underwater_target",
             "水下航行器": "underwater_target", "unresolved": "unknown"}
    text = value.strip().lower()
    if text in names:
        return names[text]
    for key in CLASS_ORDER[:8]:
        if re.match(r"^" + re.escape(key) + r"(?=$|[\s(（]|[\u4e00-\u9fff])", value, re.I):
            return key
    raise ValueError(f"Unrecognized saved class label: {value!r}; provide an explicit label_aliases mapping")


def diagnostics(counts) -> dict:
    matrix = np.asarray(counts, dtype=float)
    if (matrix.shape != (13, 15) or not np.all(np.isfinite(matrix))
            or np.any(matrix < 0) or np.any(matrix != np.floor(matrix))):
        raise ValueError("Expected nonnegative integer 13 x 15 confusion counts")
    support = matrix.sum(axis=1)
    if np.any(support <= 0):
        raise ValueError("All thirteen true classes need positive support")
    tp = np.diag(matrix[:, :13])
    predicted = matrix[:, :13].sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(13), where=predicted > 0)
    recall = tp / support
    f1 = np.divide(2 * tp, support + predicted)
    return {"classes": list(CLASS_ORDER), "prediction_classes": PREDICTION_ORDER,
            "counts": matrix.astype(int).tolist(), "row_fraction": (matrix / support[:, None]).tolist(),
            "support": support.astype(int).tolist(), "n": int(support.sum()),
            "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(),
            "l3_accuracy": float(tp.sum() / support.sum()),
            "l3_macro_f1": float(np.mean(np.round(f1, 4))),
            "zero_predicted_precision": 0.0, "annotation_threshold_fraction": 0.1}


def diagnostics_from_metrics(metrics: dict, aliases: dict | None = None) -> dict:
    labels = [label_key(v, aliases) for v in metrics["l3_confusion_labels"]]
    if len(labels) != len(set(labels)) or not set(CLASS_ORDER) <= set(labels):
        raise ValueError("Confusion labels must map uniquely and cover all thirteen classes")
    matrix = np.asarray(metrics["l3_confusion_matrix"], dtype=float)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("Saved confusion matrix and label dimensions disagree")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0) or np.any(matrix != np.floor(matrix)):
        raise ValueError("Saved confusion counts must be finite nonnegative integers")
    output = np.zeros((13, 15), dtype=int)
    for i, truth in enumerate(labels):
        if truth not in CLASS_ORDER:
            if matrix[i].sum() != 0:
                raise ValueError("Unknown/cascade cannot be ground-truth classes")
            continue
        for j, prediction in enumerate(labels):
            output[CLASS_ORDER.index(truth), PREDICTION_ORDER.index(prediction)] += int(matrix[i, j])
    result = diagnostics(output)
    if result["n"] != metrics["total_samples"]:
        raise ValueError("Saved total_samples differs from confusion counts")
    if not math.isclose(result["l3_accuracy"], metrics["l3_accuracy"], abs_tol=1e-10):
        raise ValueError("Saved L3 accuracy differs from confusion counts")
    if not math.isclose(result["l3_macro_f1"], metrics["l3_macro_f1"], abs_tol=5e-5):
        raise ValueError("Saved macro F1 differs from confusion counts")
    per_class = metrics.get("l3_per_class", {})
    mapped = {label_key(key, aliases): value for key, value in per_class.items()}
    if len(mapped) != len(per_class) or set(mapped) != set(CLASS_ORDER):
        raise ValueError("Saved per-class metrics must cover thirteen unique labels")
    for i, leaf in enumerate(CLASS_ORDER):
        for key in ("precision", "recall", "f1", "support"):
            if not math.isclose(mapped[leaf][key], result[key][i], abs_tol=5.1e-5):
                raise ValueError(f"Saved {leaf} {key} disagrees with confusion counts")
    result["saved_label_mapping"] = dict(zip(metrics["l3_confusion_labels"], labels))
    return result


def canonical_predictions(rows: list[dict], records: list[dict], *, strict: bool) -> list[dict]:
    expected = {r["id"]: r for r in records}
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prediction sample IDs (do not combine merged and shard outputs)")
    if set(ids) != set(expected):
        raise ValueError(f"Prediction ID coverage mismatch: missing={len(set(expected)-set(ids))}, extra={len(set(ids)-set(expected))}")
    result = []
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        sid, truth = row["sample_id"], row["gt"]
        if truth != {key: expected[sid][key.lower()] for key in ("L1", "L2", "L3")}:
            raise ValueError(f"{sid}: prediction ground truth differs from the audited Eval manifest")
        t1, t2 = row["turn1_pred"], row["turn2_pred"]
        leaf = t2["L3"]
        if t1 not in ("active", "passive", "unknown") or leaf not in PREDICTION_ORDER:
            raise ValueError(f"{sid}: unsupported saved prediction label")
        skipped = row.get("cascade_skipped")
        if type(skipped) is not bool or skipped != (t1 != truth["L1"]):
            raise ValueError(f"{sid}: inconsistent cascade state")
        if skipped != (leaf == "cascade_error"):
            raise ValueError(f"{sid}: cascade label differs from execution state")
        status1, status2 = row.get("turn1_parse_status"), t2.get("parse_status")
        if strict:
            if status1 not in ("valid_option", "invalid_format"):
                raise ValueError(f"{sid}: five-state decomposition requires explicit Turn 1 parse status")
            if (status1 == "invalid_format") != (t1 == "unknown"):
                raise ValueError(f"{sid}: Turn 1 parse status and label disagree")
            if not skipped:
                if status2 not in ("valid_option", "invalid_format"):
                    raise ValueError(f"{sid}: five-state decomposition requires executed Turn 2 parse status")
                if (status2 == "invalid_format") != (leaf == "unknown"):
                    raise ValueError(f"{sid}: Turn 2 parse status and label disagree")
                if leaf in CLASS_ORDER:
                    family = CLASS_TO_FAMILY[leaf]
                    parent = "passive" if family == "ship_noise" else "active"
                    if (t2["L1"], t2["L2"]) != (parent, family):
                        raise ValueError(f"{sid}: inconsistent predicted hierarchy")
        result.append({"id": sid, "gt": truth, "t1": t1, "l3": leaf, "cascade": skipped,
                       "t1_status": status1, "t2_status": status2})
    return result


def analyze_predictions(rows: list[dict], *, strict: bool) -> tuple[dict, dict]:
    counts = np.zeros((13, 15), dtype=int)
    states = STRICT_STATES if strict else LEGACY_STATES
    outcomes = [0] * len(states)
    ids_by_state = [[] for _ in states]
    for row in rows:
        counts[CLASS_ORDER.index(row["gt"]["L3"]), PREDICTION_ORDER.index(row["l3"])] += 1
        correct = row["l3"] == row["gt"]["L3"]
        if strict:
            if row["t1_status"] == "invalid_format": index = 0
            elif row["cascade"]: index = 1
            elif row["t2_status"] == "invalid_format": index = 2
            else: index = 4 if correct else 3
        else:
            index = 0 if row["cascade"] else (2 if correct else 1)
        outcomes[index] += 1
        ids_by_state[index].append(row["id"])
    result = diagnostics(counts)
    cascades = sum(row["cascade"] for row in rows)
    if sum(outcomes[:2] if strict else outcomes[:1]) != cascades or sum(outcomes) != len(rows):
        raise ValueError("Decision-state counts do not conserve samples or cascades")
    if not math.isclose(outcomes[-1] / len(rows), result["l3_accuracy"], abs_tol=1e-12):
        raise ValueError("Leaf-correct state differs from confusion accuracy")
    return result, {"states": states, "counts": outcomes, "fractions": [n / len(rows) for n in outcomes],
                    "n": len(rows), "cascade_count": cascades, "executed_count": len(rows)-cascades,
                    "ids_by_state": ids_by_state,
                    "basis": "explicit_parse_status" if strict else "legacy_saved_labels_only"}


def make_strata(records: list[dict]) -> dict:
    """One fixed class-wise quantile partition shared by every model."""
    assignments, thresholds = {}, {}
    for leaf in CLASS_ORDER:
        pool = sorted((r for r in records if r["l3"] == leaf), key=lambda r: r["id"])
        field = "S_src" if CLASS_TO_FAMILY[leaf] == "ship_noise" else "G_h"
        if not pool or any(r.get(field) is None or isinstance(r[field], bool) for r in pool):
            raise ValueError(f"{leaf}: required metadata {field} is missing")
        values = np.asarray([r[field] for r in pool], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{leaf}: nonfinite {field}")
        cuts = np.quantile(values, [1/3, 2/3], method="linear")
        bins = np.searchsorted(cuts, values, side="left")
        thresholds[leaf] = {"field": field, "cutpoints": cuts.tolist(),
                            "counts": np.bincount(bins, minlength=3).tolist()}
        for row, value, tier in zip(pool, values, bins):
            if row["id"] in assignments:
                raise ValueError("Duplicate metadata ID")
            assignments[row["id"]] = {"class": leaf, "value": float(value), "stratum": int(tier)}
    return {"names": ["Low", "Middle", "High"], "quantile_method": "linear",
            "boundaries": "Low: x <= q1; Middle: q1 < x <= q2; High: x > q2",
            "ties": "equal values stay together; empty strata retained, not filled or rebalanced",
            "thresholds": thresholds, "assignments": assignments}


def metadata_performance(rows: list[dict], strata: dict) -> dict:
    if {r["id"] for r in rows} != set(strata["assignments"]):
        raise ValueError("Prediction and stratum IDs differ")
    counts = {leaf: np.zeros((3, 2), dtype=int) for leaf in CLASS_ORDER}
    for row in rows:
        item = strata["assignments"][row["id"]]
        if item["class"] != row["gt"]["L3"]:
            raise ValueError("Prediction and stratum ground truth differ")
        pair = counts[item["class"]][item["stratum"]]
        pair[0] += int(row["l3"] == row["gt"]["L3"])
        pair[1] += 1
    by_class = {leaf: {"correct": values[:, 0].tolist(), "n": values[:, 1].tolist(),
                       "recall": [float(c/n) if n else None for c, n in values]}
                for leaf, values in counts.items()}
    groups = {}
    for group, leaves in (("active", CLASS_ORDER[:8]), ("ship_noise", CLASS_ORDER[8:])):
        means, missing = [], []
        for tier in range(3):
            absent = [leaf for leaf in leaves if by_class[leaf]["recall"][tier] is None]
            missing.append(absent)
            means.append(None if absent else float(np.mean([by_class[leaf]["recall"][tier] for leaf in leaves])))
        groups[group] = {"macro_recall": means, "missing_classes": missing, "classes": leaves,
                         "n": [sum(by_class[leaf]["n"][tier] for leaf in leaves) for tier in range(3)]}
    return {"groups": groups, "by_class": by_class,
            "weighting": "equal class weights; all eight active or all five ship classes required in each stratum"}


def model_performance_range(runs: list[dict], group: str) -> dict:
    """Pointwise observed extrema across selected models, never a confidence band."""
    ids = [run["id"] for run in runs]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Model range requires nonempty, unique model IDs")
    values = [run["metadata"]["groups"][group]["macro_recall"] for run in runs]
    if any(len(row) != 3 for row in values):
        raise ValueError("Model range requires exactly three metadata strata")
    if any(v is not None and (isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1)
           for row in values for v in row):
        raise ValueError("Model range requires finite recall in [0, 1] or explicit missing values")
    result = {"model_ids": ids, "lower": [], "upper": [], "gap_pp": [],
              "lower_model_ids": [], "upper_model_ids": [], "missing_model_ids": []}
    for tier in range(3):
        missing = [model_id for model_id, row in zip(ids, values) if row[tier] is None]
        result["missing_model_ids"].append(missing)
        if missing:
            for key in ("lower", "upper", "gap_pp"): result[key].append(None)
            for key in ("lower_model_ids", "upper_model_ids"): result[key].append([])
            continue
        column = [row[tier] for row in values]
        low, high = min(column), max(column)
        result["lower"].append(low); result["upper"].append(high)
        result["gap_pp"].append(100 * (high-low))
        result["lower_model_ids"].append([model_id for model_id, v in zip(ids, column) if v == low])
        result["upper_model_ids"].append([model_id for model_id, v in zip(ids, column) if v == high])
    return result


def validate_comparison(value: dict) -> dict:
    """Consume the existing verified paired-comparison output; never infer pairs."""
    if value.get("status") != "complete" or value.get("comparison") != "original_minus_silent":
        raise ValueError("G requires a complete original_minus_silent comparison.json")
    a, b = value["original"], value["silent"]
    if a.get("integrity_verified") is not True or b.get("integrity_verified") is not True:
        raise ValueError("G requires verified original and silent run aggregates")
    if a["total_samples"] != b["total_samples"] or a["total_samples"] <= 0:
        raise ValueError("Paired comparison sample counts differ")
    result = {"n": a["total_samples"], "metrics": {},
              "run_signatures": {"original": a["run_signature"], "silent": b["run_signature"]}}
    for condition in ("original", "silent"):
        if value["counts"][condition]["samples"] != result["n"]:
            raise ValueError("Paired comparison execution counts differ from metric counts")
    for metric in ("l3_accuracy", "l1_accuracy"):
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (a[metric], b[metric])):
            raise ValueError("Invalid accuracy in paired comparison")
        delta = a[metric] - b[metric]
        if not math.isclose(delta, value["delta"][metric], abs_tol=1e-12):
            raise ValueError("Paired delta is not original minus silent")
        result["metrics"][metric] = {"original": a[metric], "silent": b[metric], "delta_pp": 100*delta}
    return result
