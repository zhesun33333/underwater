# UA-Bench 评测框架

正式评测使用数据集内固定的三轮问题：两次分类回答确定三层标签，第三轮输出辅助解释。

| 轮次 | 任务 | 计分 |
|---|---|---|
| Turn 1 | 主动发射或被动辐射 | L1 |
| Turn 2 | 在对应分支选择一个完整类别路径 | 同时确定 L2、L3 |
| Turn 3 | 给出支持所选类别的声学描述 | 五项词汇诊断指标，不新增分类标签 |

Active 指为探测或通信而主动发射的信号，Passive 指舰船或水下航行目标自身辐射的噪声，与接收机监听模式无关。

## 独立发布与输入

发布完整 `testsite/`、Step 2 生成的 JSONL 和相应处理后 WAV。评测不需要 `archive/`、原始源音频、原始 JSONC 或顶层共享 Python 文件。随代码提供的 `config/eval_config.yaml` 仍用于类别定义和默认运行设置，使用者无需另备问题配置文件。

每条记录包含 `id`、`audio`、三个 human 问题所在的 `conversations`、标准化 `_gt`，以及供诊断使用的 `_meta`。新版 Step 2 还写入 `record_schema_version` 和 `qa_prompt_version`。模型仅收到音频、固定问题和本次生成的对话历史，不接收 `_gt`、`_meta` 或参考答案。

完整说明见 [数据发布与评测](DATASET_EVALUATION.md)。

## 快速使用

在包含 `testsite/` 的目录运行。基础 Mock 检查需要 Python 和 PyYAML；静音生成还需要 NumPy、SoundFile。真实模型按相应后端文档准备环境。`requirements.txt` 是历史环境导出，含机器相关路径，不应作为通用安装清单直接使用。

```powershell
python -m testsite.scripts.run_eval --mock 26 --backend mock --output-dir results/mock_check
python -m testsite.scripts.run_eval --data release/sft_test.jsonl --audio-root release --backend qwen25_omni --model-id MODEL_PATH --output-dir results/model_run_001
```

`MODEL_PATH` 替换为实际模型位置。`--mock` 只选择合成数据，使用模拟后端必须同时指定 `--backend mock`。可用后端及参数以 `python -m testsite.scripts.run_eval --help` 为准。

每次直接运行应指定新的输出目录，避免固定名称的预测与协议文件被覆盖。多 GPU 脚本 `scripts/run_multigpu.sh` 自动创建独立运行目录。手动分片需要一致的 `--run-id`，汇总方法见数据发布文档。

## 解析与门控

- Prompt 和参考答案仍要求单个大写选项字母。
- 接收器允许单个 ASCII 大写或小写字母，忽略首尾空白，并验证该选项确实在当前问题中。
- `A`、`a`、` a ` 等价；`A.`、`(A)`、`Answer: A`、类别名、多个字母、全角字母和越界选项无效。
- 不从模型自由文本中恢复类别，原始回答原样保存。
- Turn 1 错误或无效：终止 Turn 2/3，后续分类记为 `cascade_error`，仍计入总体准确率分母。
- Turn 1 正确：执行 Turn 2 和 Turn 3。Turn 2 无效时 L2/L3 为 `unknown`，仍执行解释轮次。

## 指标与输出

分类报告包括 L1/L2/L3 Accuracy、L3 Macro F1、L2|L1、L3|L2、逐类 P/R/F1 和混淆矩阵。不再单独报告 Joint。

解释报告包括 alignment、contradiction、concept confusion、vague、term stacking。它们衡量解释与预测类别的词汇关系，不等同于声学证据正确性或人工语义评分。公式和当前边界行为见 [评估框架设计文档](评估框架设计文档.md)。

每次主评测写入 `protocol_shard_NNN.json`、`predictions_shard_NNN.jsonl`、`eval_metrics_*.json` 和 `eval_report_*.md`。预测文件保留实际问题、原始输出、解析结果和门控状态。

## 当前实验范围

主实验使用固定评测集，补充实验以原音频与等时长、等采样率静音对照为主，复用相同指标和三轮流程。操作见 [静音对照](SILENT_CONTROL.md)。遗留 `--ablation text_only/prompt_robustness/all` 入口仍存在，但不属于当前论文最小实验计划；它们的附加结果不会自动成为主报告中的独立完整实验产物。

## 维护

`core/` 提供数据加载、单字母解析、评分、模型后端和共享提示词；`eval/multi_turn.py` 编排三轮；`reporting/` 生成报告与图表；`scripts/` 提供运行、汇总、静音生成及模型部署入口。

```powershell
python -m unittest discover -s testsite/tests -v
```

`tests/` 是可重复的回归检查，应保留。历史试跑输出、一次性调试脚本和编辑器临时副本不属于发布所需文件。
