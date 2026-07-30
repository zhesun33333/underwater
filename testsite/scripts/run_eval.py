#!/usr/bin/env python
"""运行多轮对话评估流程。

用法:
    python -m testsite.scripts.run_eval
    python -m testsite.scripts.run_eval --data sft_test.jsonl
    python -m testsite.scripts.run_eval --mock 50
    python -m testsite.scripts.run_eval --ablation text_only
    python -m testsite.scripts.run_eval --ablation all
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from testsite.config import load_config
from testsite.core.loader import DataLoader
from testsite.core.inference import ModelInference
from testsite.core.scorer import Scorer
from testsite.eval.multi_turn import MultiTurnEvaluator
from testsite.reporting.report import ReportGenerator
from testsite.ablation.text_only import TextOnlyEvaluator
from testsite.ablation.prompt_robustness import PromptRobustnessEvaluator


def main():
    parser = argparse.ArgumentParser(description="水声大模型层级分类评估")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径 (默认: config/eval_config.yaml)")
    parser.add_argument("--data", type=str, default=None,
                        help="测试数据 JSONL 路径 (默认: mock 模式)")
    parser.add_argument("--audio-root", type=str, default="processed_audio",
                        help="处理后音频根目录 (默认 processed_audio)")
    parser.add_argument("--mock", type=int, default=None,
                        help="Mock 模式生成 N 条样本 (默认无数据时用 50)")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="结果输出目录")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["text_only", "prompt_robustness", "all"],
                        help="运行消融实验")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["mock", "qwen2_audio", "aero1_audio",
                                 "voxtral_mini", "af_next", "kimi_audio"],
                        help="覆盖配置文件中的 model.backend")
    parser.add_argument("--model-id", type=str, default=None,
                        help="覆盖配置文件中的 model.model_id")
    args = parser.parse_args()

    # ============================================================
    # 1. 加载配置
    # ============================================================
    print("=" * 60)
    print("  水声大模型 层级分类评估")
    print("=" * 60)

    config = load_config(args.config)
    if args.backend:
        config["model"]["backend"] = args.backend
    if args.model_id:
        config["model"]["model_id"] = args.model_id

    # 自动按模型名创建子目录, 避免结果互相覆盖
    backend_name = config["model"]["backend"]
    if args.output_dir == "eval_results" and backend_name != "mock":
        args.output_dir = f"eval_results/{backend_name}"

    # ============================================================
    # 2. 加载数据
    # ============================================================
    loader = DataLoader(config)

    if args.data:
        print(f"\n加载测试数据: {args.data}")
        samples = loader.load(jsonl_path=args.data, audio_root=args.audio_root)
    else:
        n_mock = args.mock if args.mock is not None else 50
        print(f"\n无数据文件, 使用 Mock 模式 ({n_mock} 条)")
        samples = _mock_samples(config, n_mock)

    if not samples:
        print("错误: 无样本可评估")
        return 1

    print(f"  加载 {len(samples)} 条样本")

    # ============================================================
    # 3. 初始化
    # ============================================================
    inference = ModelInference(config)

    # ============================================================
    # 4. 多轮对话评估
    # ============================================================
    print(f"\n{'=' * 60}")
    print("  多轮对话评估")
    print(f"{'=' * 60}")

    multi_eval = MultiTurnEvaluator(config, inference)
    hier_metrics, reasoning, _ = multi_eval.evaluate(samples)

    print(f"  T1 L1 Accuracy:     {hier_metrics.l1.accuracy:.2%}")
    print(f"  T2 L2 Accuracy:     {hier_metrics.l2.accuracy:.2%}")
    print(f"  T2 L3 Accuracy:     {hier_metrics.l3.accuracy:.2%}")
    print(f"  L3 Macro F1:        {hier_metrics.l3.macro_f1:.4f}")
    print(f"  Joint (三级全对):    {hier_metrics.joint_accuracy:.2%}")
    print(f"  L2|L1:              {hier_metrics.l2_given_l1:.2%}")
    print(f"  L3|L2:              {hier_metrics.l3_given_l2:.2%}")
    print()
    if hier_metrics.source_stratified:
        for src_key in ["PulseCom_TL", "Ship_SNR"]:
            src = hier_metrics.source_stratified.get(src_key, {})
            if not src:
                continue
            print(f"\n  {'─' * 50}")
            print(f"  {src.get('name', src_key)}")
            print(f"  总样本: {src.get('count', 0)}")
            print(f"  {'区间':<16} {'样本':>5} {'L3 Acc':>8} {'Cascade':>8}")
            for b in src.get("bins", []):
                print(f"  {b['label']:<16} {b['count']:>5} {b['l3_acc']:>7.2%} {b['cascade_rate']:>7.2%}")

    if hier_metrics.per_class_snr:
        deg = hier_metrics.per_class_snr.get("degradation", {})
        if deg:
            print(f"\n  SNR 退化最严重的 3 个类别:")
            for cls, drop in sorted(deg.items(), key=lambda x: -x[1])[:3]:
                print(f"    {cls}: {drop:.0%}")

    print(f"\n  T3 Alignment:       {reasoning.alignment_rate:.2%}")
    print(f"  T3 Contradiction:   {reasoning.contradiction_rate:.2%}")
    print(f"  概念混淆:            {reasoning.concept_confusion_rate:.2%}")
    print(f"  空洞泛化:            {reasoning.vague_rate:.2%}")
    print(f"  术语堆砌:            {reasoning.term_stacking_rate:.2%}")

    # ============================================================
    # 5. 消融实验
    # ============================================================
    if args.ablation in ("text_only", "all"):
        print(f"\n{'=' * 60}")
        print("  消融: Text-Only (去掉音频)")
        print(f"{'=' * 60}")

        text_eval = TextOnlyEvaluator(config, inference)
        text_metrics, text_results = text_eval.evaluate(samples)

        # Bootstrap 95% CI
        l1_ci = Scorer.bootstrap_ci(text_results,
                     lambda rs: sum(1 for r in rs if r.turn1_pred.L1 == r.sample.gt["L1"]) / len(rs))
        l2_ci = Scorer.bootstrap_ci(text_results,
                     lambda rs: sum(1 for r in rs if r.turn2_pred.L2 == r.sample.gt["L2"]) / len(rs))
        l3_ci = Scorer.bootstrap_ci(text_results,
                     lambda rs: sum(1 for r in rs if r.turn2_pred.L3 == r.sample.gt["L3"]) / len(rs))
        jt_ci = Scorer.bootstrap_ci(text_results,
                     lambda rs: sum(1 for r in rs
                         if r.turn1_pred.L1 == r.sample.gt["L1"]
                         and r.turn2_pred.L2 == r.sample.gt["L2"]
                         and r.turn2_pred.L3 == r.sample.gt["L3"]) / len(rs))

        def _fmt(ci):
            return f"[{ci[0]:.2%}, {ci[1]:.2%}]" if ci[0] is not None else "[—]"

        print(f"  L1 Accuracy:   {text_metrics.l1.accuracy:.2%}  (95% CI {_fmt(l1_ci)})")
        print(f"  L2 Accuracy:   {text_metrics.l2.accuracy:.2%}  (95% CI {_fmt(l2_ci)})")
        print(f"  L3 Accuracy:   {text_metrics.l3.accuracy:.2%}  (95% CI {_fmt(l3_ci)})")
        print(f"  Joint:         {text_metrics.joint_accuracy:.2%}  (95% CI {_fmt(jt_ci)})")

    if args.ablation in ("prompt_robustness", "all"):
        print(f"\n{'=' * 60}")
        print("  消融: Prompt 鲁棒性 (遍历所有模板)")
        print(f"{'=' * 60}")

        robust_eval = PromptRobustnessEvaluator(config, inference)
        robust = robust_eval.evaluate(samples)

        def _score_ci(scores):
            """对逐样本一致率做 bootstrap CI。"""
            if len(scores) < 5:
                return "[—]"
            ci = Scorer.bootstrap_ci(scores, lambda xs: sum(xs) / len(xs))
            if ci[0] is None:
                return "[—]"
            return f"[{ci[0]:.2%}, {ci[1]:.2%}]"

        t1_ci = _score_ci(robust["t1_per_sample"])
        t2_l2_ci = _score_ci(robust["t2_l2_per_sample"])
        t2_l3_ci = _score_ci(robust["t2_l3_per_sample"])

        print(f"  T1 L1 一致率 ({robust['t1_samples']} 样本): "
              f"{robust['t1_agreement']:.2%}  (95% CI {t1_ci})")
        print(f"  T2 L2 一致率 ({robust['t2_samples']} 样本): "
              f"{robust['t2_l2_agreement']:.2%}  (95% CI {t2_l2_ci})")
        print(f"  T2 L3 一致率 ({robust['t2_samples']} 样本): "
              f"{robust['t2_l3_agreement']:.2%}  (95% CI {t2_l3_ci})")

    # ============================================================
    # 6. 生成报告
    # ============================================================
    print(f"\n{'=' * 60}")
    print("  生成评估报告")
    print(f"{'=' * 60}")

    report_gen = ReportGenerator(output_dir=args.output_dir)
    report_text = report_gen.generate(
        hier_metrics,
        model_name="水声大模型",
        reasoning=reasoning,
    )

    # ============================================================
    # 7. 总结
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"  评估完成")
    print(f"  L3 Accuracy: {hier_metrics.l3.accuracy:.2%}")
    print(f"  Joint:       {hier_metrics.joint_accuracy:.2%}")
    print(f"  报告:        {args.output_dir}/")
    print(f"{'=' * 60}")

    return 0


def _mock_samples(config, n):
    """用 MockModel 生成模拟样本 (用于测试框架)。"""
    from testsite.core.inference import MockModel
    from testsite.core.loader import EvalSample
    import random
    rng = random.Random(42)

    l3_classes = [
        ("CW", "pulse", "active"), ("LFM", "pulse", "active"), ("HFM", "pulse", "active"),
        ("2FSK", "communication", "active"), ("4FSK", "communication", "active"),
        ("BPSK", "communication", "active"), ("QPSK", "communication", "active"),
        ("OFDM", "communication", "active"),
        ("cargo", "ship_noise", "passive"), ("cruise", "ship_noise", "passive"),
        ("fishing", "ship_noise", "passive"), ("warship", "ship_noise", "passive"),
        ("underwater_target", "ship_noise", "passive"),
    ]

    samples = []
    for i in range(n):
        l3, l2, l1 = rng.choice(l3_classes)
        sample_id = f"mock_{l3}_{i:06d}"
        audio_path = f"mock/audio/{l3}/{sample_id}.wav"
        samples.append(EvalSample(
            sample_id=sample_id,
            audio_path=audio_path,
            gt={"L1": l1, "L2": l2, "L3": l3},
            metadata={
                "snr_db": rng.uniform(-15, 25),
                "ssp_complexity": rng.uniform(0, 10),
            },
        ))
    return samples


if __name__ == "__main__":
    sys.exit(main())
