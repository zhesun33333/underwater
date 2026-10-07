# 使用数据集中的固定问题评测

正式多轮评测直接读取每条 JSONL 记录的 `conversations`，按顺序取三个 `from: "human"` 的 `value`，原样作为三轮问题。问题中应已包含完整选项和回答要求，不在评测时补充、替换或随机选择。

```powershell
python -m testsite.scripts.run_eval --data dataset/sft_test_highquality.jsonl --audio-root processed_audio --backend qwen25_omni
```

无需额外提供提示词配置文件。运行器仍使用随评测代码提供的默认模型和评分设置，模型位置等可通过命令行覆盖。

- 音频取自记录的 `audio` 字段，样本身份取自 `id`。
- 真值必须取自内嵌 `_gt`，不再查找原始 JSON/JSONC 元数据。缺少标签时会明确报错。
- `_meta` 保存源身份、生成与传播参数及诊断统计量，仅供评估器使用，不加入模型上下文。
- 三个参考答案不载入模型对话历史。历史中的回答全部来自本次模型生成。
- 第一轮错误或无效仍终止后续对话；第一轮正确时，第二轮使用记录中的固定分支问题。
- 第二轮无论正确与否，继续发送记录中的第三题。指标及计分公式不变。
- 缺少三个非空 human 问题会报错，不会回退到配置模板。
- 预测文件保存实际发送的三轮问题及记录中的 `qa_prompt_version`。未执行的轮次问题为空。

`--mock` 创建用于测试的合成样本，其问题在样本构造时固定。显式调用 `prompt_robustness` 消融时才会使用额外的替代模板；它是修改问题的独立实验，不属于正式固定问题评测。

更新生成器不会自动改变现有 JSONL 中的问题。发布前应确认数据集中的问题已经包含所需的单字母限制和第三轮要求。

## 独立发布

新版两个 Step 2 的 train/val/test JSONL 均内嵌标准化 `_gt`、`_meta` 和 `record_schema_version`。直接发布这些 JSONL 与 `audio` 指向的处理后 WAV 即可，不需要原始源音频、原始 JSON/JSONC、archive 或上级目录的共享 Python 文件。JSONL 不内嵌 WAV 数据，WAV 必须随数据发布。

保持 WAV 相对于 `--audio-root` 的路径即可移动整个数据包。新版路径使用 `/`，加载器也兼容旧 Windows 分隔符。术语表和共享提示词位于 `testsite/core/`，配置随 `testsite/config/` 发布。

在包含 `testsite` 文件夹的目录运行，例如：

```powershell
python -m testsite.scripts.run_eval --data release/sft_test.jsonl --audio-root release --backend qwen25_omni --model-id /path/to/model --output-dir results/new_run
```

旧版已经包含完整 `_gt` 的评测子集仍可直接使用；旧版未内嵌 `_gt` 的 Step 2 输出需要重新生成。缺失源端质量统计量不会阻止分类评测，但基于质量的评测子集筛选会明确报错，不能用缺失值冒充排序分数。筛选脚本优先使用新版记录内嵌的统计量，不读取原始元数据。

## 完整性检查与汇总

加载时检查整份输入清单的 ID 唯一性，坏 JSON、缺失字段或无法加载的所选样本会报错；所选样本缺少音频文件也会终止。每一轮后端返回的回答数必须等于请求数，最终结果 ID 必须与输入一致。

多 GPU 启动脚本会在输出根目录下创建新的 `run_时间戳_随机后缀` 子目录，并只读取本次预期的分片文件。所有分片共享本次 `run-id`。若手工分别启动分片，必须传入同一个 `--run-id`；新实验应使用新的值。

独立汇总现在必须指定原始完整输入清单：

```powershell
python -m testsite.scripts.merge_predictions --input predictions.jsonl --output metrics.json --manifest dataset/sft_test_highquality.jsonl
```

汇总会拒绝重复、缺失或额外 ID、混合运行签名、清单哈希不符、内嵌真值不符，以及问题或门控状态不一致的结果。部分样本测试不能冒充完整清单评测。旧结果缺少运行签名时会明确拒绝，不自动补造签名。

数据导出脚本将 `--output` 作为名称前缀，每次创建带时间戳的新目录及压缩包，不删除旧目录。缺少 WAV 或重复 ID 会终止导出，而不会生成缺样本的正式压缩包。
