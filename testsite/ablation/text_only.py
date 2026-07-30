"""Text-Only 消融: 去掉音频, 仅用文本 prompt 做多轮对话。

期望: 模型无法依赖音频信号, 分类接近随机水平。
若 text-only 准确率显著 > 随机, 说明模型在利用 prompt 中的偏见而非音频。
"""

import random
from typing import List, Tuple

from ..core.loader import EvalSample
from ..core.inference import ModelInference
from ..core.parser import OutputParser, HierPrediction
from ..core.scorer import Scorer, HierarchicalMetrics
from ..eval.multi_turn import TurnResult


class TextOnlyEvaluator:
    """Text-Only 消融评估器 — 复刻多轮对话但去掉音频。"""

    def __init__(self, config: dict, inference: ModelInference):
        self.inference = inference
        self.parser = OutputParser(config)
        self.scorer = Scorer(config)
        self.prompts = config["prompts"]["three_turn"]

    def evaluate(
        self, samples: List[EvalSample]
    ) -> Tuple[HierarchicalMetrics, List[TurnResult]]:
        results: List[TurnResult] = []
        rng = random.Random(42)
        cascade_count = 0

        for sample in samples:
            # Turn 1: L1 判别 (无音频)
            q1 = rng.choice(self.prompts["turn1"])
            a1 = self.inference.generate(sample.audio_path, q1, no_audio=True)
            pred1 = self.parser.parse_turn1(sample.sample_id, a1, prompt=q1)
            t1_correct = (pred1.L1 == sample.gt["L1"])

            if t1_correct:
                l1_detected = pred1.L1
                if l1_detected == "active":
                    t2_templates = self.prompts["turn2_active"]
                else:
                    t2_templates = self.prompts["turn2_passive"]

                q2 = rng.choice(t2_templates).replace(
                    "{L1}", self._l1_display_name(l1_detected))

                history_t2 = [
                    {"from": "human", "value": q1},
                    {"from": "gpt", "value": a1},
                    {"from": "human", "value": q2},
                ]
                a2 = self.inference.chat(sample.audio_path, history_t2,
                                         no_audio=True)
                pred2 = self.parser.parse(sample.sample_id, a2, prompt=q2)
                t2_correct = (
                    pred2.L2 == sample.gt["L2"] and pred2.L3 == sample.gt["L3"]
                )
                skipped = False

                q3 = self.prompts["turn3"]
                history_t3 = history_t2 + [
                    {"from": "gpt", "value": a2},
                    {"from": "human", "value": q3},
                ]
                a3 = self.inference.chat(sample.audio_path, history_t3,
                                         no_audio=True)
            else:
                q2 = ""
                a2 = ""
                pred2 = HierPrediction(
                    sample_id=sample.sample_id,
                    L1="unknown", L2="cascade_error", L3="cascade_error",
                    parse_tier=0, raw_output="[cascade: T1 L1 incorrect]",
                )
                t2_correct = False
                skipped = True
                q3 = ""
                a3 = ""
                cascade_count += 1

            results.append(TurnResult(
                sample=sample,
                turn1_prompt=q1, turn1_output=a1, turn1_pred=pred1,
                turn1_correct=t1_correct,
                turn2_prompt=q2, turn2_output=a2, turn2_pred=pred2,
                turn2_correct=t2_correct,
                turn3_prompt=q3, turn3_output=a3,
                cascade_skipped=skipped,
            ))

        n = len(results)
        gts = [r.sample.gt for r in results]

        t1_l1_preds = [r.turn1_pred.L1 for r in results]
        t1_l1_gts   = [g["L1"] for g in gts]
        l1_metrics = self.scorer.compute_level(
            "L1", self.scorer.l1_keys, t1_l1_preds, t1_l1_gts)

        t2_l2_preds = [r.turn2_pred.L2 for r in results]
        t2_l3_preds = [r.turn2_pred.L3 for r in results]
        t2_l2_gts   = [g["L2"] for g in gts]
        t2_l3_gts   = [g["L3"] for g in gts]
        l2_metrics = self.scorer.compute_level(
            "L2", self.scorer.l2_keys, t2_l2_preds, t2_l2_gts)
        l3_metrics = self.scorer.compute_level(
            "L3", self.scorer.l3_keys, t2_l3_preds, t2_l3_gts)

        l1_correct_set = [r.turn1_pred.L1 == g["L1"]
                          for r, g in zip(results, gts)]
        l2_correct_set = [r.turn2_pred.L2 == g["L2"]
                          for r, g in zip(results, gts)]

        l1_c = sum(l1_correct_set)
        l2_c = sum(l2_correct_set)
        l1_l2_c = sum(a and b for a, b in zip(l1_correct_set, l2_correct_set))
        l2_l3_c = sum(
            r.turn2_pred.L2 == g["L2"] and r.turn2_pred.L3 == g["L3"]
            for r, g in zip(results, gts))
        all3_c = sum(
            r.turn1_pred.L1 == g["L1"]
            and r.turn2_pred.L2 == g["L2"]
            and r.turn2_pred.L3 == g["L3"]
            for r, g in zip(results, gts))

        metrics = HierarchicalMetrics(
            total_samples=n,
            l1=l1_metrics,
            l2=l2_metrics,
            l3=l3_metrics,
            l2_given_l1=l1_l2_c / l1_c if l1_c > 0 else 0.0,
            l3_given_l2=l2_l3_c / l2_c if l2_c > 0 else 0.0,
            joint_accuracy=all3_c / n if n > 0 else 0.0,
        )
        return metrics, results

    @staticmethod
    def _l1_display_name(l1_key: str) -> str:
        if l1_key == "active":
            return "主动信号"
        elif l1_key == "passive":
            return "被动信号"
        return "该信号"
