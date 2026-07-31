"""评估报告生成器 — 多轮对话层级分类指标。

输出 Markdown 格式报告，包含:
  1. 层级分类 (L1/L2/L3 Acc, Per-class F1, Confusion)
  2. 层级联合指标 (L2|L1, L3|L2, Joint)
  3. 分源质量分层 (PulseCom by TL, Ship by SNR)
  4. 推理质量 (Domain Term Match)
  5. Parse Tier 分布
  6. 结论与建议
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..core.scorer import HierarchicalMetrics, ReasoningMetrics


class ReportGenerator:
    """Markdown 评估报告生成器。"""

    def __init__(self, output_dir: str = "eval_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, hierarchical: HierarchicalMetrics,
                 model_name: str = "Underwater Acoustic LLM",
                 reasoning: Optional[ReasoningMetrics] = None) -> str:
        """生成完整评估报告。"""
        sections = [
            self._header(model_name),
            self._hierarchical_section(hierarchical),
            self._joint_section(hierarchical),
        ]

        if hierarchical.source_stratified:
            sections.append(self._source_stratified_section(hierarchical.source_stratified))

        if hierarchical.per_class_snr:
            sections.append(self._per_class_snr_section(hierarchical.per_class_snr))

        if reasoning:
            sections.append(self._reasoning_section(reasoning))

        sections.append(self._parse_tier_section(hierarchical))
        sections.append(self._recommendations(hierarchical))

        report = "\n".join(sections)

        ts = int(time.time())
        report_path = self.output_dir / f"eval_report_{ts}.md"
        report_path.write_text(report, encoding="utf-8")

        json_path = self.output_dir / f"eval_metrics_{ts}.json"
        json_path.write_text(json.dumps(self._to_dict(hierarchical, reasoning),
                                        ensure_ascii=False, indent=2), encoding="utf-8")

        return report

    @staticmethod
    def _to_dict(h: HierarchicalMetrics, r: Optional[ReasoningMetrics] = None) -> dict:
        result = {
            "total_samples": h.total_samples,
            "l1_accuracy": h.l1.accuracy if h.l1 else 0,
            "l2_accuracy": h.l2.accuracy if h.l2 else 0,
            "l3_accuracy": h.l3.accuracy if h.l3 else 0,
            "l3_macro_f1": h.l3.macro_f1 if h.l3 else 0,
            "l2_given_l1": h.l2_given_l1,
            "l3_given_l2": h.l3_given_l2,
            "joint_accuracy": h.joint_accuracy,
            "parse_tier_dist": h.parse_tier_dist,
        }
        # L3 逐类详细数据
        if h.l3 and h.l3.per_class:
            result["l3_per_class"] = h.l3.per_class
        if h.l3 and h.l3.confusion_matrix:
            result["l3_confusion_matrix"] = h.l3.confusion_matrix
            result["l3_confusion_labels"] = h.l3.confusion_labels
        if h.source_stratified:
            result["source_stratified"] = h.source_stratified
        if h.per_class_snr:
            result["per_class_snr"] = h.per_class_snr
        if r:
            result["reasoning"] = {
                "alignment_rate": r.alignment_rate,
                "contradiction_rate": r.contradiction_rate,
                "concept_confusion_rate": r.concept_confusion_rate,
                "vague_rate": r.vague_rate,
                "term_stacking_rate": r.term_stacking_rate,
                "cascade_skipped": r.cascade_skipped,
            }
        return result

    @staticmethod
    def _confusion_table(labels: List[str], cm: List[List[int]]) -> str:
        lines = ["| True \\ Predicted | " + " | ".join(labels) + " |"]
        lines.append("|" + "---|" * (len(labels) + 1))
        for i, row in enumerate(cm):
            lines.append(f"| {labels[i]} | " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)

    @staticmethod
    def _header(name: str) -> str:
        return f"# Underwater Acoustic LLM — Hierarchical Classification Evaluation Report\n\n**Model**: {name}\n**Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"

    @staticmethod
    def _hierarchical_section(h: HierarchicalMetrics) -> str:
        lines = ["---\n## 1. Hierarchical Classification\n"]
        for lm in [h.l1, h.l2, h.l3]:
            if lm is None:
                continue
            lines.append(f"### {lm.level} — {lm.total} 样本")
            lines.append(f"- Accuracy: **{lm.accuracy:.2%}**")
            lines.append(f"- Macro F1: **{lm.macro_f1:.4f}**\n")
        # L3 per-class
        if h.l3 and h.l3.per_class:
            lines.append("### L3 Per-Class Accuracy\n")
            lines.append("| Class | Precision | Recall | F1 | Support |")
            lines.append("|------|-----------|--------|----|---------|")
            for name, m in h.l3.per_class.items():
                lines.append(f"| {name} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.4f} | {m['support']} |")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _joint_section(h: HierarchicalMetrics) -> str:
        return f"""---
## 2. Hierarchical Joint Metrics

| Metric | Value | Description |
|------|------|------|
| L2\\|L1 | **{h.l2_given_l1:.2%}** | Conditional prob. L2 correct given L1 correct |
| L3\\|L2 | **{h.l3_given_l2:.2%}** | Conditional prob. L3 correct given L2 correct |
| Hierarchical Joint | **{h.joint_accuracy:.2%}** | L1 + L2 + L3 all correct |
"""

    @staticmethod
    def _source_stratified_section(data: dict) -> str:
        lines = [
            "---\n## 3. Source-Stratified Quality\n",
            "PulseCom stratified by transmission loss (TL), Ship stratified by line-spectrum SNR.\n",
        ]
        for src_key in ["PulseCom_TL", "Ship_SNR"]:
            src = data.get(src_key, {})
            if not src or not src.get("bins"):
                continue
            lines.append(f"### {src.get('name', src_key)}")
            lines.append(f"Total samples: {src.get('count', 0)}\n")
            lines.append("| Bin | Count | L3 Accuracy | Cascade Rate |")
            lines.append("|------|--------|-------------|-------------|")
            for b in src["bins"]:
                lines.append(
                    f"| {b['label']} | {b['count']} | **{b['l3_acc']:.2%}** | **{b['cascade_rate']:.2%}** |")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _per_class_snr_section(data: dict) -> str:
        per_bin = data.get("per_bin", {})
        degradation = data.get("degradation", {})
        if not per_bin:
            return ""

        lines = [
            "---\n## Per-Class SNR Degradation Analysis\n",
            "### L3 Accuracy by Class across SNR Bins\n",
        ]

        all_classes = sorted(degradation.keys(),
                             key=lambda c: degradation.get(c, 0), reverse=True)
        if not all_classes:
            return ""

        for label in per_bin:
            bin_data = per_bin[label]
            lines.append(f"#### {label}")
            lines.append("| Class | Count | Accuracy |")
            lines.append("|------|--------|----------|")
            for cls in all_classes:
                cd = bin_data.get(cls, {})
                cnt = cd.get("count", 0)
                if cnt == 0:
                    continue
                lines.append(f"| {cls} | {cnt} | **{cd['acc']:.2%}** |")
            lines.append("")

        if degradation:
            lines.append("### SNR Degradation Magnitude (≥15dB → ≤-5dB)\n")
            lines.append("| Class | Degradation |")
            lines.append("|------|----------|")
            for cls, drop in sorted(degradation.items(), key=lambda x: -x[1]):
                lines.append(f"| {cls} | {drop:.1%} |")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _reasoning_section(r: ReasoningMetrics) -> str:
        return f"""---
## 4. Reasoning Quality

| Metric | Value |
|------|------|
| Term Alignment | **{r.alignment_rate:.2%}** |
| Contradiction | {r.contradiction_rate:.2%} |
| Concept Confusion | {r.concept_confusion_rate:.2%} |
| Vague / Generic | {r.vague_rate:.2%} |
| Term Stacking | {r.term_stacking_rate:.2%} |
"""

    @staticmethod
    def _parse_tier_section(h: HierarchicalMetrics) -> str:
        lines = ["---\n## 5. Parse Tier Distribution\n"]
        lines.append("| Tier | Count | Ratio |")
        lines.append("|------|--------|------|")
        for tier in sorted(h.parse_tier_dist.keys()):
            cnt = h.parse_tier_dist[tier]
            lines.append(f"| {tier} | {cnt} | {cnt/h.total_samples:.1%} |")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _recommendations(h: HierarchicalMetrics) -> str:
        lines = ["---\n## 6. Conclusions and Recommendations\n"]
        if h.l3 and h.l3.accuracy < 0.30:
            lines.append("- L3 accuracy is low; consider checking if the taxonomy is overly fine-grained or training data is insufficient.")
        if h.l3 and h.l3.per_class:
            worst = sorted(h.l3.per_class.items(), key=lambda x: x[1]["f1"])[:3]
            worst_strs = [f"{n}(F1={m['f1']:.3f})" for n, m in worst]
            lines.append(f"- Weakest classes: {', '.join(worst_strs)}")
        if h.l2_given_l1 < 0.50:
            lines.append("- L2|L1 is low; L1 classification errors significantly impact downstream classification.")
        lines.append("")
        return "\n".join(lines)
