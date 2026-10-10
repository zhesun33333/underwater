# UA-Bench 论文可视化

已实现数据图 B、A1、A2、C 与结果图 D、E、F、G。D–G 已接入 2026-10-10 的九模型完整运行与静音配对结果；论文绘图不运行模型、不重算传播、不修改发布 JSONL 或音频。结果来源、校验范围和重绘命令见 [RESULTS_GUIDE.md](RESULTS_GUIDE.md)。

## 目录

```text
visualization/
  config.example.json       # 可复制的路径与排版配置
  config.local.json         # 本机配置，Git 忽略
  project.py                # 数据准备、缓存校验、输出溯源
  plot_distribution.py      # 图 B：完整 test 与 Eval 的 ECDF
  acoustic_analysis.py      # 代表样本、符号窗、STFT、Welch、谱质心统计
  plot_acoustic.py           # 图 A1/A2
  channels.example.json     # C 的示例类别、源归档和缓存配置
  channel_data.py            # C 的 source/channel 配对、选样与校验
  plot_channel.py            # 图 C：源与传播后的声学结构
  import_result_archive.py   # 安全导入双层 tar 结果并保存 SHA 索引
  results.20261010.example.json # 九模型归档结果配置示例
  results.py / archive_protocol.py / paired_archive_validation.py
  plot_results.py            # 图 D/E/F/G 的独立出版版式
  figures/data/
    selection_distributions.py / .pdf / .png / .json / .data.json.gz
    acoustic_active.py / .pdf / .png / .json / .data.json.gz
    acoustic_ships.py / .pdf / .png / .json / .data.json.gz
    channel_examples.py / .pdf / .png / .json / .data.json.gz
  figures/results/
    class_diagnostics.py / .pdf / .png / .json / .data.json.gz
    error_decomposition.py / .pdf / .png / .json / .data.json.gz
    metadata_performance.py / .pdf / .png / .json / .data.json.gz
    silence_control.py / .pdf / .png / .json / .data.json.gz
  cache/                    # 头索引及紧凑绘图数据，Git 忽略
  tests/
```

公共读取在 `testsite/reporting/figure_data.py`，可供论文图和旧绘图复用；`testsite/` 仍可单独发布。图件脚本和产物在同一目录，一张图一个稳定 stem。生成文件和大缓存不加入 Git。

## 本机运行

在仓库根目录 `D:\sz_workplace\UAbench` 运行。已安装的 Tavotto 工作解释器为：

```powershell
$plotPython = 'C:\Users\admin\AppData\Roaming\Tavotto\mcp-runtime\venv\Scripts\python.exe'
& $plotPython -m pip install -r visualization/requirements.txt
& $plotPython -m visualization prepare
& $plotPython -m visualization render B A
```

在 Codex 外的终端，如果 MSIX 虚拟化使上述路径不可见，本机实际解释器为：
`C:\Users\admin\AppData\Local\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Roaming\Tavotto\mcp-runtime\venv\Scripts\python.exe`。
这些机器相关路径只属于本机运行说明，不写进绘图脚本。

首次迁移到其他机器时，先复制 `config.example.json` 为 `config.local.json`，修改其中的数据路径。相对路径按配置文件所在目录解析。可用 `python -m visualization --config <配置路径> prepare`；单独执行图件脚本或通过 Tavotto 使用另一配置时，设置进程环境变量 `UABENCH_FIGURE_CONFIG`。

`prepare` 会流式读取两份正式清单；完整 test 的时长来自已审核的 WAV 头索引，Eval 的时长重新读取 2,600 个实际 WAV 头。它检查 ID、13 类支持数、子集关系，以及共有记录的标签、源身份、实际时长、质量统计量、频率和信号参数。图 A1 只解码选中的八条音频；A2 为总体统计读取全部 1,000 条 Eval 舰船音频。

本机初始头索引来自上一轮全部 56,221 条 WAV 头审核。原审核还验证了 2,600 条导出音频与归档成员的 SHA-256 一致；审核摘要保存在 `cache/original_archive_audit.json`。导入旧索引只登记其来源，不冒充本次重新读取了全部归档。头索引 sidecar 绑定清单、索引 SHA-256 及归档大小/修改时间；数据改动后需要重建索引再 `prepare`。原始元数据时长不会作为缺失 WAV 的替代。

新机器没有头索引，或归档变更后，使用以下命令重建。程序只读取匹配成员的 WAV 头，不解包音频；也支持无扩展名的 `warship_ok`。已有本机缓存不需重复执行。

```powershell
& $plotPython -m testsite.reporting.build_header_index --manifest C:/sz_data/dataset_v2/sft_test.jsonl --archive C:/sz_data/v1 --output visualization/cache/full_test_wav_headers.jsonl
& $plotPython -m visualization prepare
```

缓存读取校验派生数据 SHA-256，以及清单、索引、归档和每条 Eval WAV 的大小/修改时间。它能检测普通文件变更；不声称会在每次画图时重新对数百 GB 归档做内容哈希。改变配置或公共读取代码后需重新 `prepare`。

## 图中数据与统计口径

**图 B** 为 2×2 阶梯 ECDF：实际时长、主动信号中心/载频、主动信道能量增益 `G_h`、舰船源端线谱与连续谱功率比 `S_src`。家族颜色固定，完整 test 虚线、Eval 实线；统计单位为清单记录，两个信道输出分别计数。每个面板标有效总 n，伴随数据保存每家族有效/缺失 n、缺失原因、逐条 ID、数值和 ECDF 坐标。`G_h` 不是正的 transmission loss；`S_src` 不是接收端 SNR。信号频率与 BELLHOP 当时使用的计算频率分开保存。

**图 A1** 展示八类主动信号，**A2** 展示五类舰船代表谱和一幅各类谱质心统计。每类在 Eval 内，按实际时长、家族质量统计量以及主动类的真实中心/载频计算 median/IQR 距离；零 IQR 维度剔除，ID 打破并列。样本不依赖模型预测，也不按图形外观手选。

A2 第六幅对每类全部 200 条 Eval 记录，使用与代表谱相同的 Welch 设置，对整个 WAV 计算 0–1000 Hz 内的谱质心 `sum(f * PSD) / sum(PSD)`；这里的 PSD 是线性功率密度。点表示中位数，横线表示第 25–75 百分位区间，不是置信区间；不同信道记录分别计数。每条音频 SHA-256、谱质心值和类别分位数写入伴随数据。静音、无效数值或不一致采样率会明确报错，不静默剔除。

脉冲展示完整输入；通信按固定符号周期规则裁剪中心窗口。STFT 窗长会根据单载波符号率设置，OFDM 使用单独的分析设置；实际采样点和所有参数都写入伴随数据。功率图只展示时频能量结构，不能据此验证 PSK 的相位状态数。每个家族共用功率参考，不逐面板调亮。舰船使用共同 Welch 参数和频率范围；数字振幅功率谱不标为物理声压级，所显示的低频范围也不代表更高频处没有能量。

已确认排版为宽 **180 mm 横向版**、**Times New Roman 9 pt**、无边框图例；每个子图单独显示轴标题与刻度。A1 使用两行四列，上排是三类脉冲与 OFDM，下排是四类单载波调制；单载波横轴为相对截取起点的毫秒，原始 WAV 时间和采样点保留在伴随数据中。A2 使用完整 2×3，右下角是谱质心统计。图件直接按最终宽度保存，未用 `bbox_inches='tight'` 改变 PDF 物理尺寸。时频图栅格嵌入 300 dpi，文字、坐标和曲线保持矢量。

## Tavotto

数据图库在 `D:\sz_workplace\UAbench\visualization\figures\data`，结果图库在 `D:\sz_workplace\UAbench\visualization\figures\results`。八个入口均有无参 `main()`、导入阶段无数据读取/出图、静态字面量 PDF/PNG 文件名、不调用 `plt.show()`。Tavotto 的渲染 worker 调用 `main()` 时不写数据缓存或溯源 JSON；独立运行脚本时才写伴随记录。

本机可执行真实引擎验收：

```powershell
& $plotPython -m visualization.verify_tavotto
& $plotPython -m visualization.verify_results_tavotto --results-config visualization/results.local.json D E F G
```

两个验收适配器针对 Tavotto **0.18.0**：生成注册表，实际调用 safe worker，检查可编辑元素和出版预检，并核对最终 PDF 页尺寸、字体/矢量对象、PNG 分辨率及图件哈希。报告分别在 `figures/data/verification_tavotto.json` 与 `figures/results/verification_tavotto.json`。普通画图只依赖 `requirements.txt`；此可选验收另需 `tavotto[worker]==0.18.0`。升级 Tavotto 后需重新核对适配器。D–G 最新图的本机预检无阻断项；画幅比例提示对应已确认的 180 mm 横向紧凑版。

本地引擎验收不等同于 MCP 或交互画布验收。已载入 Tavotto MCP 工具的会话可对以上图库调用 `tavotto_refresh_project`，再按需打开对应 stem 修改和预检；首次访问按插件提示授权具体图库目录。

当前 B、A1 为 **180×110 mm**，A2 为 **180×90 mm**。2026-10-09 将画布高度、行间空白及底部区域压缩，同时保留 180 mm 宽、9 pt 字号和全部面板；选样规则、数值、谱分析参数和功率参考保持不变。验收报告以图件哈希绑定当前产物。

各图的紧凑高度已写入绘图代码默认值，新数据沿用相同版式。必要时可在 `config.local.json` 的 `style` 中以毫米覆盖高度：

| 图 | 高度配置键 | 180 mm 宽时的默认高度 |
|---|---|---|
| A1 | `active_height_mm` | 110 mm |
| A2 | `ship_height_mm` | 90 mm |
| B | `distribution_height_mm` | 110 mm |
| C | `channel_height_mm` | 115 mm |
| D | `diagnostic_height_mm` | 120 mm |
| E | `error_height_mm` | 85 mm |
| F | `metadata_height_mm` | 85 mm |
| G | `silence_height_mm` | 85 mm |

调整配置后按常规先重新 `prepare`；结果图再执行 `prepare-results`。只更新绘图代码的默认高度和间距时，可直接运行对应 `render`，不需要重跑评测。D 的十三行类别保留较多高度，使圆角方块、数值和 P/R/F1 标记仍可辨认。

A2 的六面板图保留五条代表谱和分析参数，并汇总各类全部 Eval 录音的谱质心；此统计单独记录于 `centroid_summary` 和 `raw_data.ship_centroids`。

六面板图的数值与哈希见 `figures/data/acoustic_ships.json` 和 `acoustic_ships.data.json.gz`；B、A1、A2、C 的本机 Tavotto 检查见 `figures/data/verification_tavotto.json`。此前 MCP 工作区授权未完成，因此本机检查不代表交互画布验收。

## 回归检查

```powershell
& $plotPython -m unittest discover -s visualization/tests -v
& $plotPython -m unittest testsite.tests.test_figure_data -v
```

`.json` 保存输入与代码哈希、软件版本、尺寸及图件哈希；`.data.json.gz` 保存可复算的绘图数据、样本选择及分析参数。修改图件代码后重新运行对应入口，再执行验收；不用重跑评测。

结果图使用单独的 `results.local.json` 和 `prepare-results` 命令。当前本机配置为 `archived_verified`，从保存的九组预测、精确清单和签名协议复核 D–G；它会如实记录与本机源码的版本差异。未来用当前检出代码直接运行的新实验仍可采用更严格的 `verified` 模式。旧 ZIP 仅供 `legacy_preview` 历史对照。完整步骤见 [RESULTS_GUIDE.md](RESULTS_GUIDE.md)。

图 C 的同源配对入口为 `figures/data/channel_examples.py`：先配置 `channels.local.json`，运行 `prepare-channels`，之后 `render C`。三行展示脉冲、通信、舰船的源音频及真实传播输出，完整定义与归档读取说明见 [CHANNEL_GUIDE.md](CHANNEL_GUIDE.md)。
