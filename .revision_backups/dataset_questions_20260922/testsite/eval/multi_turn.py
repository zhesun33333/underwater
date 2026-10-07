"""多轮对话评估 — 三轮对话。

对三轮对话格式的样本:
  1. Turn1: 发送 T1 prompt + 音频 → 解析 L1
  2. Turn2: 如果 T1 正确, 继续追问 L2+L3
            如果 T1 错误, 跳过 T2/T3, L2/L3 记为 cascade_error
  3. Turn3: 追问判断依据 → Domain Term Match (无论 T2 是否正确都执行)
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass

from ..core.loader import EvalSample
from ..core.inference import ModelInference
from ..core.parser import OutputParser, HierPrediction
from ..core.scorer import Scorer, HierarchicalMetrics, ReasoningMetrics


@dataclass
class TurnResult:
    """单条样本的三轮对话评估结果。"""
    sample: EvalSample
    turn1_prompt: str
    turn1_output: str
    turn1_pred: HierPrediction
    turn1_correct: bool
    turn2_prompt: str
    turn2_output: str
    turn2_pred: HierPrediction
    turn2_correct: bool
    turn3_prompt: str
    turn3_output: str
    cascade_skipped: bool              # 因 T1 错误而跳过 T2/T3


class MultiTurnEvaluator:
    """三轮对话评估器。

    T1 错误 → 级联终止，跳过 T2 和 T3。
    T1 正确 → T2 → T3 (无论 T2 是否正确都追问推理)。
    """

    def __init__(self, config: dict, inference: ModelInference):
        self.inference = inference
        self.parser = OutputParser(config)
        self.scorer = Scorer(config)
        self.prompts = config["prompts"]["three_turn"]
        self.batch_size = max(1, int(config.get("model", {}).get("batch_size", 1)))

    def evaluate(
        self, samples: List[EvalSample]
    ) -> Tuple[HierarchicalMetrics, ReasoningMetrics, List[TurnResult]]:
        results: List[TurnResult] = []
        import random
        rng = random.Random(42)

        cascade_count = 0

        for start in range(0, len(samples), self.batch_size):
            chunk = samples[start:start + self.batch_size]
            q1s = [rng.choice(self.prompts["turn1"]) for _ in chunk]
            a1s = self.inference.generate_batch(
                [s.audio_path for s in chunk], q1s)
            states = []
            eligible = []
            for i, (sample, q1, a1) in enumerate(zip(chunk, q1s, a1s)):
                pred1 = self.parser.parse_turn1(sample.sample_id, a1, prompt=q1)
                ok = pred1.L1 == sample.gt["L1"]
                state = dict(sample=sample, q1=q1, a1=a1, pred1=pred1,
                             t1_correct=ok)
                states.append(state)
                if ok:
                    templates = (self.prompts["turn2_active"] if pred1.L1 == "active"
                                 else self.prompts["turn2_passive"])
                    q2 = rng.choice(templates).replace(
                        "{L1}", self._l1_display_name(pred1.L1))
                    history2 = [
                        {"from": "human", "value": q1},
                        {"from": "gpt", "value": a1},
                        {"from": "human", "value": q2},
                    ]
                    state.update(q2=q2, history2=history2)
                    eligible.append(state)

            if eligible:
                a2s = self.inference.chat_batch(
                    [x["sample"].audio_path for x in eligible],
                    [x["history2"] for x in eligible])
                histories3 = []
                for state, a2 in zip(eligible, a2s):
                    pred2 = self.parser.parse(
                        state["sample"].sample_id, a2, prompt=state["q2"])
                    state.update(a2=a2, pred2=pred2,
                                 t2_correct=(pred2.L2 == state["sample"].gt["L2"]
                                             and pred2.L3 == state["sample"].gt["L3"]))
                    histories3.append(state["history2"] + [
                        {"from": "gpt", "value": a2},
                        {"from": "human", "value": self.prompts["turn3"]},
                    ])
                a3s = self.inference.chat_batch(
                    [x["sample"].audio_path for x in eligible], histories3)
                for state, a3 in zip(eligible, a3s):
                    state["a3"] = a3

            for state in states:
                if not state["t1_correct"]:
                    state.update(q2="", a2="", a3="", t2_correct=False,
                        pred2=HierPrediction(
                            sample_id=state["sample"].sample_id, L1="unknown",
                            L2="cascade_error", L3="cascade_error", parse_tier=0,
                            raw_output="[cascade: T1 L1 incorrect]"))
                    cascade_count += 1
                results.append(TurnResult(
                    sample=state["sample"], turn1_prompt=state["q1"],
                    turn1_output=state["a1"], turn1_pred=state["pred1"],
                    turn1_correct=state["t1_correct"], turn2_prompt=state["q2"],
                    turn2_output=state["a2"], turn2_pred=state["pred2"],
                    turn2_correct=state["t2_correct"],
                    turn3_prompt=self.prompts["turn3"] if state["t1_correct"] else "",
                    turn3_output=state["a3"],
                    cascade_skipped=not state["t1_correct"]))

        n = len(results)
        gts = [r.sample.gt for r in results]

        # ── T1: L1 指标 (T1 是模型独立判断 L1 的唯一时机) ──
        t1_l1_preds = [r.turn1_pred.L1 for r in results]
        t1_l1_gts   = [g["L1"] for g in gts]
        l1_metrics = self.scorer.compute_level(
            "L1", self.scorer.l1_keys, t1_l1_preds, t1_l1_gts)

        # ── T2: L2 / L3 指标 ──
        t2_l2_preds = [r.turn2_pred.L2 for r in results]
        t2_l3_preds = [r.turn2_pred.L3 for r in results]
        t2_l2_gts   = [g["L2"] for g in gts]
        t2_l3_gts   = [g["L3"] for g in gts]
        l2_metrics = self.scorer.compute_level(
            "L2", self.scorer.l2_keys, t2_l2_preds, t2_l2_gts)
        l3_metrics = self.scorer.compute_level(
            "L3", self.scorer.l3_keys, t2_l3_preds, t2_l3_gts)

        # ── 跨轮条件指标 ──
        l1_correct_set = [r.turn1_pred.L1 == g["L1"] for r, g in zip(results, gts)]
        l2_correct_set = [r.turn2_pred.L2 == g["L2"] for r, g in zip(results, gts)]

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

        # Parse tier 分布
        tiers: Dict[int, int] = {}
        for r in results:
            t = r.turn2_pred.parse_tier
            tiers[t] = tiers.get(t, 0) + 1

        hierarchical = HierarchicalMetrics(
            total_samples=n,
            l1=l1_metrics,
            l2=l2_metrics,
            l3=l3_metrics,
            l2_given_l1=l1_l2_c / l1_c if l1_c > 0 else 0.0,
            l3_given_l2=l2_l3_c / l2_c if l2_c > 0 else 0.0,
            joint_accuracy=all3_c / n if n > 0 else 0.0,
            parse_tier_dist=tiers,
        )

        # ── T3: 推理质量 ──
        t3_mask = [not r.cascade_skipped for r in results]
        t3_texts = [r.turn3_output for r, m in zip(results, t3_mask) if m]
        t3_l3s   = [r.turn2_pred.L3 for r, m in zip(results, t3_mask) if m]
        reasoning = self.scorer.compute_reasoning_quality(t3_texts, predicted_l3=t3_l3s)
        reasoning.cascade_skipped = cascade_count

        if cascade_count > 0:
            print(f"  [Cascade] {cascade_count}/{n} 条因 T1 错误跳过 T2/T3")

        return hierarchical, reasoning, results

    def _l1_display_name(self, l1_key: str) -> str:
        if l1_key == "active":
            return "actively transmitted"
        elif l1_key == "passive":
            return "source-radiated"
        return "unidentified"
