# 论文结果图 D–G

代码在 `visualization/`，入口与输出在 `visualization/figures/results/`。采用 180 mm 横向、Times New Roman 9 pt、无边框图例；D 为 180×120 mm，E/F/G 为 180×85 mm。2026-10-10 归档已生成 D、E、F、G 四张最新结果图。下文的“归档核验”表示核对已保存的预测、指标、运行协议与输入哈希，并不表示重新运行模型。

## 2026-10-10 最新结果

源包为 `C:\Users\admin\WorkBuddy\sz_bench\outputs\UAbench_全部结果_20261010_v2.tar.gz`，配置示例为 `visualization/results.20261010.example.json`，本机使用忽略的 `visualization/results.local.json`。九个唯一模型各有 2,600 条原始音频预测、两份协议分片、保存的汇总指标，以及 2,600 条原音频与静音的逐样本配对比较。

| 图 | 当前内容 | 已核对范围 |
|---|---|---|
| D `class_diagnostics` | Qwen2.5-Omni 的 13 类混淆矩阵及 P/R/F1 | 2,600 条逐样本预测、保存指标和原始运行协议 |
| E `error_decomposition` | 九模型五种互斥结局 | 每模型 2,600 条；显式解析状态 |
| F `metadata_performance` | 九模型的主动信号/舰船元数据分层召回 | 同一 Eval 的分层映射；各层九模型最高—最低实测范围 |
| G `silence_control` | 九模型原音频与匹配静音的 L3/L1 Accuracy | 每模型 2,600 对；配对记录、输入哈希及两条件计数 |

九模型为 Qwen2-Audio、Aero1-Audio、Voxtral-Mini、AF-Next、Gemma4-12B、Gemma4-E2B、MiDaShengLM、Qwen2.5-Omni 和 Voxtral-Small。D 的 `diagnostic_model` 固定为 Qwen2.5-Omni，作为单模型的诊断示例，不代表九模型的平均表现。原先选用的 Qwen2-Audio 在八个主动类全部落入 Cascade，难以展示叶类混淆；Qwen2.5-Omni 的主动/舰船正确数分别为 143/1600、198/1000，两组 Cascade 分别为 702/1600、12/1000，13 类中 7 类的对角线非零。选择依据是两类信号均保留了可观察的正确与错误叶类，而非仅比较总准确率；它仍有显著舰船预测偏置，1000 条舰船样本中 884 条预测为图中的 Naval（内部类别为 Warship）。九模型的总体比较仍见 E/F/G。F 的图例及上下界使用全部九个模型。

该包中六个原始运行绑定 SHA-256 前缀 `049a9a35` 的归档清单，另外三个（Voxtral-Mini、Gemma4-12B、Voxtral-Small）绑定本地测试集清单 `248f827f`。两份清单的 2,600 个 ID、真实标签、问题、版本和元数据逐条相同；差别只在 `audio` 字段的路径形式：归档清单是 Linux 绝对路径，本地清单是相对路径。九个静音运行绑定同一份静音清单 `56e2050d`。比较结果保留原 Linux 输入路径，配置中的 54 条 `input_relocations` 按原 SHA-256 映射到已导入文件，原始 `comparison.json` 未被改写。

`archived_verified` 核对归档 SHA、已读取文件与导入索引、清单和预测的完整 ID 覆盖、协议分片/运行签名/分类体系、保存指标与逐样本预测的一致性，以及 G 的配对输出。它另外报告归档中受保护实现的 11 个代码哈希与当前工作区代码的对比：经 CRLF→LF 换行归一化后，每个模型至少 10/11 一致；Qwen2-Audio 和 Aero1-Audio 为 11/11，其余七个模型的 `testsite/core/inference.py` 仍不同。因此这批图只主张对归档运行的核验，不主张可用当前推理代码逐位重现这些历史运行。结果包没有 WAV 或模型权重；本流程只使用保存结果，不重跑推理。

从空目录重新导入时，在 `D:\sz_workplace\UAbench` 运行（导入目标必须为空；本机已完成导入，可跳过第一行）：

```powershell
python -m visualization.import_result_archive "C:\Users\admin\WorkBuddy\sz_bench\outputs\UAbench_全部结果_20261010_v2.tar.gz" "visualization\cache\results_20261010_sources"
Copy-Item visualization\results.20261010.example.json visualization\results.local.json -Force
python -m visualization prepare-results
python -m visualization render D E F G
python -m visualization.verify_results_tavotto --results-config visualization/results.local.json D E F G
```

最后一行应使用已安装 Tavotto 0.18.0 的 Python 解释器（见主 [README.md](README.md) 的 `$plotPython`）；它运行本机 safe worker 和出版预检，并核对最终 PDF/PNG，不调用 MCP 交互画布。若基础 Eval 缓存尚未准备，先运行 `python -m visualization prepare`。导入目标换位置时，应同步改配置中的 `archive_index`、预测、协议、指标、各运行的 `manifest`、顶层 `silent_manifest`、配对及重定位路径。当前审计摘要在 `visualization/cache/results_20261010/results.audit.json`，数值缓存为 `results.json.gz`；图的伴随 `.json`/`.data.json.gz` 保留所用结果与输入溯源。

## 历史旧包审计与可用范围

审计对象为 `D:\sz_workplace\temp_files\水生大模型评估结果.zip`。包内 154 个文件，其中 29 个 JSON、26 个 JSONL，没有 Python 源码、运行协议或静音配对结果。目录中的名称只作为旧运行标签，不凭目录名称推断模型权重版本。

| 旧运行 | 完整预测 | 旧包可生成图 |
|---|---|---|
| Qwen2-Audio | 无，只有汇总指标 | D，旧版默认诊断对象 |
| Aero1-Audio | 无，只有汇总指标 | D，可修改配置选中 |
| Voxtral-Mini | 无，只有汇总指标 | D，可修改配置选中 |
| AF-Next | 2,600 条 | D、E 三状态预览、F |
| Gemma4-12B | 2,600 条 | D、E 三状态预览、F |
| Gemma4 (run1) | 2,600 条 | D、E 三状态预览、F |
| MiDaShengLM | 2,600 条 | D、E 三状态预览、F |
| Qwen2.5-Omni | 2,600 条 | D、E 三状态预览、F |
| Voxtral-Small | 2,600 条 | D、E 三状态预览、F |

六组完整预测的 ID 集合、真实标签均与当前审核后的 Eval 一致，每类 200 条；没有重复、缺失或额外 ID，级联状态一致。预测重算混淆矩阵与保存指标一致。三组仅有指标的运行可核对矩阵、样本数与 P/R/F1 的内部一致性，但不能验证样本身份、问题或运行协议。

旧包没有 `turn1_parse_status` / `turn2_pred.parse_status`。因此不把 `parse_tier` 当作格式有效性，不用现在的解析器重解旧的长文本输出，也不把缺失字段补成 `valid_option`。旧图显式标为 **Legacy preview**，不能作为当前协议的正式实验结果。包中合并预测与分片是重复表示，配置只选一套；ZIP 直接按唯一成员路径后缀读取，不解压、不执行包内内容、不扫描 checkpoint 作为额外模型。

## 图的统计定义

- **D：`class_diagnostics`**。固定 13 个真实类别、15 个预测列，包括 `Unresolved` 和 `Cascade`。行分母包含全部样本；即使失败列全零也保留。P/R/F1 从同一矩阵重算，并与保存值核对；Precision 在无预测时为 0。仅标注占本行 10% 及以上的单元格，完整计数保存在伴随数据。支持旧中英文类别名及 `label_aliases` 显式映射，未知或重复映射会报错。
- **E：`error_decomposition`**。最新 `archived_verified` 结果使用显式解析状态，九模型均画五类：T1 无效、T1 有效但错误、T2 无效、T2 有效但叶类错误、叶类正确。五类互斥，合计各模型的 2,600 条；前两类之和对应未进入 T2 的级联样本，最后一类对应 L3 Accuracy。历史旧包缺少显式状态，只能生成三状态预览。
- **F：`metadata_performance`**。用审核后的 Eval 元数据，在每个叶类内分别对 Active 的 `G_h` 和 Ship 的 `S_src` 计算 1/3、2/3 分位点（NumPy `linear`）。Low 为 `x <= q1`，Middle 为 `q1 < x <= q2`，High 为 `x > q2`。相同数值不拆散，因此不强制每层同样大小。先算每类每层召回率，再对 Active 八类或 Ship 五类等权平均；任何必需类别出现空层时该组该层记为缺失，不按剩余类别偷偷重加权。各模型共用同一份 ID→分层映射。旧包的 `source_stratified` / `per_class_snr` 不直接复用。
- **G：`silence_control`**。使用九份已保存的 `comparison.json` 与逐样本 `paired_predictions.jsonl`，核对每组 2,600 个 ID/真实标签、两条件的完整原始预测、各自的原音频/静音清单与三轮问题、运行签名、计数及六个输入文件的 SHA-256。两个清单的 SHA-256 分别匹配对应比较结果与协议；两条件逐样本预测分别按自身清单验证，清单的 ID/真实标签相同且除 `audio` 路径外所有字段一致。原音频运行还须与 D/E/F 中的同一模型运行签名一致。左右两面板为 L3/L1 Accuracy，原音频点与静音点成对连线，旁边标 `100*(original-silent)` 个百分点；负值和零值照实保留。图是配对控制的描述性结果，不是显著性检验。包内无 WAV，无法在此流程重新核验静音波形字节；没有真实静音结果时仍会报错，而不会生成虚构图。

所有代码只分析已保存结果，不修改 parser/scorer/multi_turn 等受运行协议哈希保护的评测核心文件。

F 的两面板使用相同、从零开始的纵轴范围，上限随所选模型的最大召回率调整并留出余量；最终范围记录在伴随 JSON 的 `display_ylim` 中。

F 的各模型使用不同的颜色、线型和点形。内置模型 ID 的样式固定绑定，改变顺序或选择子集时保持一致；新增模型若需跨图保持一致，应在 `plot_results.py` 的 `_metadata_styles` 中登记 ID。样式保存在 `model_styles` 中。

F 的浅色阴影表示每个分层中九个当前所绘模型的最低到最高召回率，`Δ` 标注两者相差的百分点数。这是所选模型的实测表现范围，不是统计置信区间或理论上下界；不同分层的极值可能来自不同模型，改变模型子集也会改变范围。任一所绘模型在某层缺失时，该层上下界不计算，阴影在缺失处断开。每层极值、对应模型（含并列）和缺失模型保存在 `between_model_range`。仅有一个模型时不显示阴影与差距标注。层间直线仅连接有序分层的汇总值，不估计连续条件下的表现。

D 的混淆图采用浅蓝到深青蓝的圆角方块。颜色在 0–100% 上线性映射，正值方块边长为 `0.45 + 0.50 * sqrt(p)` 个格宽（`p` 为行占比），零值用浅灰小点。方块大小提供随占比增加的辅助线索，面积不表示严格比例；最小边长用于让小占比仍可见。具体映射记录在伴随 JSON 的 `heatmap_encoding` 中。

## 历史旧包版式预览

在 `D:\sz_workplace\UAbench` 运行（解释器见主 README）：

```powershell
Copy-Item visualization/results.legacy.example.json visualization/results.local.json
python -m visualization prepare-results
python -m visualization render D E F
```

首次使用或基础数据配置改变时先运行 `python -m visualization prepare`。本段会把 `results.local.json` 切回历史预览配置；复现最新结果时，应再复制 `results.20261010.example.json` 并重新执行 `prepare-results`。`results.local.json` 不进 Git；ZIP、预测、缓存和生成图也不进 Git。

`runs` 数组决定模型顺序，`label` 决定显示名，`diagnostic_model` 明确选择 D 的分析对象，不按分数或图形外观自动挑选。配置的 ZIP 路径按实际位置修改；成员后缀必须唯一，不依赖压缩包中文根目录的编码。审计清单保存在 `cache/results_legacy/results.audit.json`。

## 将来新的本地运行

1. 复制 `results.verified.example.json` 为新的配置文件，例如 `results.local.json`，将 `mode` 保持为 `verified`。这是将来用当前工作区代码产生的新运行入口；2026-10-10 归档使用前文的 `archived_verified` 配置。
2. 为每个模型列出本次完整运行的 `predictions_shard_*.jsonl` 和全部 `protocol_shard_*.json`。如果使用合并预测，则只填合并文件，不能再同时填原分片。
3. `visualization/config.local.json` 的 `eval_manifest` 必须指向本次实际使用的清单。准备阶段会调用现有 `aggregate()` 检查清单、问题、ID 覆盖、分片、taxonomy、运行签名及评测实现哈希。缺少协议不自动降级为旧结果模式；混合运行或代码版本不符时明确报错。
4. 重新准备并出图：

```powershell
python -m visualization prepare-results
python -m visualization render D E F
```

可用 `--results-config visualization/其他配置.json` 指定配置，置于子命令之前；通过 Tavotto 直接打开脚本时，可设置进程环境变量 `UABENCH_RESULTS_CONFIG`。基础数据配置仍用 `--config` / `UABENCH_FIGURE_CONFIG`。

新的静音对照完成后，在结果配置增加：

```json
"comparisons": [
  {"id": "qwen2_audio", "label": "Qwen2-Audio", "path": "../testsite/outputs/silent_comparison/comparison.json"}
]
```

将示例路径替换为实际比较输出路径，保留 `comparison.json` 记录的原始输入文件及其路径，以便哈希核验。再次运行 `prepare-results` 和 `render G`。普通 `verified` 模式只画 G 时允许 `runs: []`、`diagnostic_model: null`；`archived_verified` 模式则要求每个 G 比较都有匹配的 D/E/F 原音频运行。若原始比较记录包含另一台机器的绝对路径，可像最新归档配置一样提供经 SHA-256 核对的 `input_relocations`。

## 溯源与 Tavotto

每张图保存 `.pdf/.png/.json/.data.json.gz`。伴随记录包含源归档或本地文件 SHA-256、数据模式、样本支持数、状态计数、分层阈值、每类每层 n 与正确数、配对核验、输入清单和代码哈希。结果源文件、配置或统计代码变化后，缓存会要求重新 `prepare-results`。

四个入口均为无参 `main()`，导入无 I/O，产物名称静态可解析，脚本与产物同目录。D 使用矢量网格，E/F/G 为矢量统计图。当前 `figures/results/verification_tavotto.json` 记录四图的 Tavotto 本机 safe-worker 与出版预检、PDF 的嵌入字体/矢量对象/物理尺寸、PNG 的 300 dpi 和所有产物 SHA-256。本次预检无阻断项，仅有与用户确认的 180 mm 宽、压缩高度相应的画幅比例提示。本聊天此前的 Tavotto 工作区授权被宿主自动拒绝，未绕过授权运行 MCP 交互预检；本机验收不等同于交互画布验收。

图 C 独立读取源音频与传播音频，不依赖模型结果归档。其配对准备及绘图入口已实现，见 [CHANNEL_GUIDE.md](CHANNEL_GUIDE.md)。
