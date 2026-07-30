# 水声大模型评估框架 (testsite)

## 目录

- [1. 评估目标](#1-评估目标)
- [2. 核心概念](#2-核心概念)
- [3. 评估流程](#3-评估流程)
- [4. 指标体系详解](#4-指标体系详解)
- [5. 模块说明](#5-模块说明)
- [6. 使用方法](#6-使用方法)
- [7. 扩展指南](#7-扩展指南)

---

## 1. 评估目标

本框架评估水声大模型在**两级分类任务**上的表现：

| 层级 | 任务 | 说明 |
|------|------|------|
| **粗分类** | 信号类型识别 | 判断音频属于 CW / LFM / HFM / Digital / Ship / Bio 中的哪一类 |
| **细分类** | 信号参数识别 | 在粗分类正确的前提下，进一步识别信号的详细参数（频率、带宽、调制方式等） |

### 设计原则

- **超越 loss**：loss 下降只能说明模型学会了语言模式，不能说明模型真正理解了水声信号
- **分维度量化**：粗分类和细分类分别评分，看清模型在不同粒度上的能力
- **定位弱项**：通过混淆矩阵和逐类 F1 找出模型容易混淆的类别
- **物理合理性**：不仅看答案对不对，还看推理是否合理（预留推理链评分接口）

---

## 2. 核心概念

### 2.1 粗分类 (Coarse Classification)

模型从一段水声音频中判断信号的大类。

```
输入: WAV 音频 (30s, 16kHz, mono)
输出: 6选1文本判断

示例模型输出:
  【信号类型】: LFM脉冲
  【判断依据】: 频谱较宽，时频图中可见频率随时间线性变化的条纹
```

评估做的事情：从这段文字里提取出"LFM脉冲"，跟真实标签比较。

### 2.2 细分类 (Fine Classification)

在明确信号类型后，进一步识别具体参数。

以 LFM 为例：
```
输入: WAV 音频 (已确定为 LFM)
输出: 4个参数

【中心频率】: 8500 Hz        ← 数值参数
【带宽】: 2000 Hz             ← 数值参数
【脉宽】: 0.2 s               ← 数值参数
【调频方向】: 上调频          ← 分类参数
```

评估做的事情：
- 数值参数（频率、带宽等）：计算预测值与真实值的偏差（MAE / RMSE / 百分比误差）
- 分类参数（调频方向、调制方式等）：计算分类准确率

### 2.3 答案解析 (Parsing)

模型输出的是自然语言文本，不是结构化的标签。解析器用三级回退策略从文字中提取答案：

| 层级 | 方法 | 成功率目标 |
|------|------|-----------|
| **Tier 1** | 定界符匹配：找 `【信号类型】: LFM脉冲` 这样的标记 | >90% |
| **Tier 2** | 关键词匹配：在全文搜索"LFM"、"线性调频"等关键词 | 兜底 5-10% |
| **Tier 3** | LLM 兜底：用小模型重新解析（预留接口） | <5% |

---

## 3. 评估流程

```
                    ┌──────────────┐
                    │  Step 1      │
                    │  加载样本     │  ← label.jsonl / Mock数据
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Step 2      │
                    │  粗分类评估   │  ← 构造prompt → 模型推理 → 解析 → 对比
                    └──────┬───────┘
                           │
                    粗分类正确? ─── 否 ──→ 记录为Coarse Error，细分类跳过
                           │
                          是
                           │
                    ┌──────▼───────┐
                    │  Step 3      │
                    │  细分类评估   │  ← 逐信号类型使用不同的prompt模板
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Step 4      │
                    │  消融实验     │  ← SNR扫描 / 时长扫描
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Step 5      │
                    │  生成报告     │  ← Markdown报告 + 混淆矩阵图 + 错误CSV
                    └──────────────┘
```

### 关键决策：细分类为什么只在粗分类正确时才计算？

如果模型把 LFM 误判为 CW，之后回答"这个CW的中心频率是 XXXX"——这个答案没有评估价值，因为前提就错了。

这跟考试阅卷的逻辑一样：选择题答错了，后面的大题即使步骤对，也没法给基础分。

---

## 4. 指标体系详解

### 4.1 粗分类指标

| 指标 | 公式/含义 | 用途 |
|------|----------|------|
| **Overall Accuracy** | 正确样本数 / 总样本数 | 综合评价 |
| **Per-class Precision** | TP / (TP + FP) | 模型说"这是XX"时，有多可信 |
| **Per-class Recall** | TP / (TP + FN) | 真实的XX类，有多少被识别出来 |
| **Per-class F1** | 2 × P × R / (P + R) | 平衡Precision和Recall |
| **Macro F1** | 各类F1的均值 | 不受类别不均衡影响 |
| **混淆矩阵** | 6×6 矩阵 | 哪些类容易混淆 |

### 4.2 细分类指标

**分类参数**（如调制方式、调频方向、包络类型）：

| 指标 | 含义 |
|------|------|
| **Accuracy** | 该参数分类正确的样本比例 |
| **Per-class F1** | 对于多分类参数（如调制方式有7种），逐类计算 |

**数值参数**（如中心频率、带宽、脉宽）：

| 指标 | 公式 | 含义 |
|------|------|------|
| **MAE** | mean(\|pred - true\|) | 预测值与真实值的平均绝对偏差 |
| **RMSE** | sqrt(mean((pred - true)²)) | 对大误差敏感 |
| **MAPE** | mean(\|pred - true\| / true) × 100% | 相对误差百分比 |
| **Tolerance Acc** | pred 在 true ± X% 之内的比例 | 如容差10%，8500±850Hz以内都算对 |

### 4.3 综合指标

| 指标 | 含义 |
|------|------|
| **Coarse Accuracy** | 粗分类对几条 |
| **Fine Joint Accuracy** | 粗分类对 + 该信号所有细分类参数全对 |
| **Fine-given-Coarse Acc** | 粗分类对了的前提下，细分类全对的比例 |

### 4.4 消融指标

| 消融维度 | 测试方法 | 输出 |
|----------|----------|------|
| **SNR 鲁棒性** | 按 SNR 分档 (-10dB~30dB) 分别测准确率 | Accuracy vs SNR 曲线 |
| **音频时长** | 分别测 1s/3s/5s/10s/30s 的准确率 | Accuracy vs Duration 曲线 |
| **多径强度** | 按路径数分档分别测准确率 | Accuracy vs Paths 曲线 |

---

## 5. 模块说明

```
testsite/
├── config/
│   ├── __init__.py            # YAML 配置加载
│   └── eval_config.yaml       # 所有评估配置（类别、prompt、消融参数）
│
├── core/                      # 核心模块
│   ├── loader.py              # 数据加载 + Mock数据生成器
│   ├── inference.py           # 模型推理接口（MockModel + 真实模型预留）
│   ├── parser.py              # 三级回退解析器（定界符→关键词→LLM）
│   └── scorer.py              # 指标计算引擎（分类+回归）
│
├── eval/                      # 评估流程编排
│   ├── coarse.py              # 粗分类评估（prompt→推理→解析→评分）
│   └── fine.py                # 细分类评估（按信号类型分发）
│
├── ablation/                  # 消融实验
│   ├── subsets.py             # 数据子集构建器（按SNR/时长/多径筛选）
│   └── runner.py              # 消融实验编排器
│
├── reporting/                 # 报告生成
│   ├── report.py              # Markdown 文本报告
│   ├── charts.py              # matplotlib 图表（混淆矩阵/SNR曲线/F1柱状图）
│   └── error_analysis.py      # 错误案例导出 CSV
│
├── utils/                     # 工具函数
│   ├── text_utils.py          # 中文文本归一化、定界符提取、数值解析
│   └── metrics_utils.py       # MAE/RMSE/MAPE 计算、容差比较
│
└── scripts/                   # 命令行入口
    ├── run_eval.py            # 运行完整评估（粗分类+细分类+消融+报告）
    └── run_ablation.py        # 单独运行消融实验
```

### 5.1 核心模块详解

#### parser.py — 答案解析器

最核心的模块。输入是模型的自然语言输出，输出是结构化预测。

```
模型输出: "【信号类型】: LFM脉冲\n【判断依据】: 频率线性变化"

Tier 1 (定界符):
  extract_delimited_field(text, "信号类型") → "LFM脉冲"
  match_coarse_label("LFM脉冲") → "LFM"
  → parse_tier=1 ✓

Tier 2 (关键词) — 仅在Tier1失败时:
  扫描全文，匹配关键词: "线性调频" → "LFM"
  → parse_tier=2

Tier 3 (LLM兜底) — 仅在Tier1+Tier2都失败时:
  调用小模型提取
  → parse_tier=3
```

#### scorer.py — 指标计算引擎

输入结构化预测和 ground truth，输出所有指标。

粗分类：
```python
y_true = ["CW", "LFM", "LFM", "Digital", ...]
y_pred = ["CW", "LFM", "HFM",  "Digital", ...]

→ Accuracy = 3/4 = 75%
→ 混淆矩阵:
     CW: 1→CW, 0→other
     LFM: 1→LFM, 1→HFM (误判)
     Digital: 1→Digital
```

细分类数值参数：
```python
pred_freqs = [8200, 9100, 7800]
true_freqs = [8500, 8500, 8500]

→ MAE = (300 + 600 + 700) / 3 = 533 Hz
→ MAPE = (3.5% + 7.1% + 8.2%) / 3 = 6.3%
→ Tolerance Acc (10%) = 2/3 = 66.7%  # 8200→8500差3.5% ✓, 9100→差7.1% ✓, 7800→差8.2% ✓
```

#### inference.py — 模型推理接口

```python
class ModelInference:
    def __init__(self, config):
        # 根据 backend 选择模型:
        #   "mock" → MockModel (返回预设格式文本)
        #   "slam_llm" → SLAM-LLM (WavLM + Vicuna, 预留)
        #   "llamafactory" → LLaMAFactory LoRA (Qwen2-Audio, 预留)

    def generate(audio_path, prompt) -> str:
        # 统一接口，返回模型文本输出
```

---

## 6. 使用方法

### 6.1 安装依赖

```bash
pip install pyyaml numpy matplotlib
```

### 6.2 快速开始（用 Mock 数据测试框架）

```bash
cd testsite/
python -m testsite.scripts.run_eval --mock-samples 100
```

这会：
1. 生成 100 条模拟测试样本（6类均匀分布）
2. 用 MockModel 模拟模型推理（预设 85% 粗分类准确率 + 70% 细分类准确率）
3. 运行粗分类 + 细分类评估
4. 输出 Markdown 报告 + 混淆矩阵图

### 6.3 使用真实数据

```bash
python -m testsite.scripts.run_eval --data /path/to/test_split.jsonl
```

### 6.4 只跑消融实验

```bash
# SNR 消融
python -m testsite.scripts.run_ablation --mock-samples 200 --ablation snr

# 所有消融
python -m testsite.scripts.run_ablation --mock-samples 200 --ablation all
```

### 6.5 评估输出

```
eval_results/
├── eval_report_20260626_143000.md   # 完整评估报告
├── confusion_matrix.png             # 混淆矩阵热力图
├── per_class_f1.png                 # 各类F1柱状图
├── snr_curve.png                    # SNR鲁棒性曲线（如果运行了消融）
├── duration_curve.png               # 时长影响曲线
└── error_analysis.csv               # 错误案例明细
```

---

## 7. 扩展指南

### 7.1 接入真实模型

在 `config/eval_config.yaml` 中修改：

```yaml
model:
  backend: "slam_llm"       # 或 "llamafactory"
  model_path: "/path/to/model"
  encoder_path: "/path/to/encoder"
  device: "cuda:0"
```

然后在 `core/inference.py` 中实现对应的 `_init_slam_llm()` 方法。

### 7.2 添加新的信号类型

在 `config/eval_config.yaml` 中：

1. `coarse.labels` 添加新条目
2. `fine.<新类型>` 定义参数列表和 prompt 模板
3. `utils/text_utils.py` 的 `COARSE_KEYWORD_MAP` 添加关键词

### 7.3 添加新的消融维度

1. `config/eval_config.yaml` 的 `ablation` 添加配置
2. `ablation/subsets.py` 添加子集构建方法
3. `ablation/runner.py` 添加消融运行方法

### 7.4 添加 LLM-as-Judge 推理链评分

预留了接口。当需要评估模型推理质量时：
1. 创建 `judge/` 目录
2. 实现 DeepSeek API 调用
3. 在 scorer 中添加推理链评分指标
4. 在 prompt 模板中要求模型输出 `【判断依据】` 字段
