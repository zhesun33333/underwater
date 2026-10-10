# UA-Bench

水声信号数据处理与音频语言模型评测代码。三层标签通过两次分类回答获得，第三轮用于解释诊断。

- `archive/`：信道处理、Step 2 标注、源级划分、测试子集筛选及数据导出。
- [testsite](testsite/README.md)：可独立发布的三轮评测框架。
- [评估框架设计](testsite/评估框架设计文档.md)：解析、门控、指标及当前边界行为。
- [数据发布](testsite/DATASET_EVALUATION.md)：Step 2 JSONL 与 WAV 的自包含数据接口。
- [论文可视化](visualization/README.md)：独立的数据图 A/B/C 与结果图 D/E/F/G 入口，兼容 Tavotto；[同源传播对比说明](visualization/CHANNEL_GUIDE.md)介绍 C 的配对与归一化，[结果图使用说明](visualization/RESULTS_GUIDE.md)说明九模型新结果、静音配对和归档协议校验。
- [最小补充实验方案](消融实验重新设计方案.md)：原音频与等长静音对照。

正式发布需要完整 `testsite/`、内嵌标签的 JSONL 及对应处理后 WAV。评测问题来自数据集，不需要原始源数据或单独的问题配置。

维护时保留可重复回归测试。`.revision_backups/` 为本地恢复资料，不属于发布内容。
