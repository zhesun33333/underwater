"""Prompt 鲁棒性消融: 同一音频, 用所有 prompt 模板分别提问, 检查答案一致性。

若不同问法答案一致 → 模型对措辞变化鲁棒。
若不一致 → 模型依赖特定 prompt 措辞, 泛化能力差。
"""

from typing import Dict, List

from ..core.loader import EvalSample
from ..core.inference import ModelInference
from ..core.parser import OutputParser


class PromptRobustnessEvaluator:
    """Prompt 鲁棒性消融 — 遍历所有 T1/T2 模板变体。"""

    def __init__(self, config: dict, inference: ModelInference):
        self.inference = inference
        self.parser = OutputParser(config)
        self.prompts = config["prompts"]["three_turn"]

    def evaluate(self, samples: List[EvalSample]) -> dict:
        """返回 T1 和 T2 的 prompt 一致性指标。"""
        t1_templates = self.prompts["turn1"]
        t2_active = self.prompts["turn2_active"]
        t2_passive = self.prompts["turn2_passive"]

        # ── T1 鲁棒性: 所有 T1 模板 → L1 一致性 ──
        t1_agreements = 0
        t1_total = 0
        t1_valid = 0
        t1_predictions = 0
        t1_per_sample = []  # 逐样本一致率, 用于 bootstrap CI

        for sample in samples:
            l1_votes: Dict[str, int] = {}
            for q1 in t1_templates:
                a1 = self.inference.generate(sample.audio_path, q1)
                pred = self.parser.parse_turn1(sample.sample_id, a1, prompt=q1)
                l1_votes[pred.L1] = l1_votes.get(pred.L1, 0) + 1
                t1_predictions += 1
                if pred.L1 != "unknown":
                    t1_valid += 1

            t1_total += 1
            top_count = max(l1_votes.values())
            score = top_count / len(t1_templates)
            t1_agreements += score
            t1_per_sample.append(score)

        # ── T2 鲁棒性: 固定 T1 (用第一个模板), 遍历所有 T2 模板 → L2/L3 一致性 ──
        t2_l2_agreements = 0
        t2_l3_agreements = 0
        t2_total = 0
        t2_l2_valid = 0
        t2_l3_valid = 0
        t2_predictions = 0
        t2_l2_per_sample = []
        t2_l3_per_sample = []

        for sample in samples:
            # 先用第一个 T1 模板确定 L1
            q1 = t1_templates[0]
            a1 = self.inference.generate(sample.audio_path, q1)
            pred1 = self.parser.parse_turn1(sample.sample_id, a1, prompt=q1)

            if pred1.L1 == "unknown":
                continue

            l1_label = "actively transmitted" if pred1.L1 == "active" else "source-radiated"
            t2_pool = t2_active if pred1.L1 == "active" else t2_passive

            l2_votes: Dict[str, int] = {}
            l3_votes: Dict[str, int] = {}
            for q2_template in t2_pool:
                q2 = q2_template.replace("{L1}", l1_label)
                history = [
                    {"from": "human", "value": q1},
                    {"from": "gpt", "value": a1},
                    {"from": "human", "value": q2},
                ]
                a2 = self.inference.chat(sample.audio_path, history)
                pred2 = self.parser.parse(sample.sample_id, a2, prompt=q2)

                l2_votes[pred2.L2] = l2_votes.get(pred2.L2, 0) + 1
                l3_votes[pred2.L3] = l3_votes.get(pred2.L3, 0) + 1
                t2_predictions += 1
                if pred2.L2 != "unknown":
                    t2_l2_valid += 1
                if pred2.L3 != "unknown":
                    t2_l3_valid += 1

            t2_total += 1
            score_l2 = max(l2_votes.values()) / len(t2_pool)
            score_l3 = max(l3_votes.values()) / len(t2_pool)
            t2_l2_agreements += score_l2
            t2_l3_agreements += score_l3
            t2_l2_per_sample.append(score_l2)
            t2_l3_per_sample.append(score_l3)

        return {
            "t1_agreement": t1_agreements / t1_total if t1_total > 0 else 0.0,
            "t2_l2_agreement": t2_l2_agreements / t2_total if t2_total > 0 else 0.0,
            "t2_l3_agreement": t2_l3_agreements / t2_total if t2_total > 0 else 0.0,
            "t1_parse_rate": t1_valid / t1_predictions if t1_predictions > 0 else 0.0,
            "t2_l2_parse_rate": t2_l2_valid / t2_predictions if t2_predictions > 0 else 0.0,
            "t2_l3_parse_rate": t2_l3_valid / t2_predictions if t2_predictions > 0 else 0.0,
            "t1_samples": t1_total,
            "t2_samples": t2_total,
            "t1_per_sample": t1_per_sample,
            "t2_l2_per_sample": t2_l2_per_sample,
            "t2_l3_per_sample": t2_l3_per_sample,
        }
