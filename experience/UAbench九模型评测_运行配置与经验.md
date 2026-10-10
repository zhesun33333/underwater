# UAbench 九模型评测 · 运行配置与经验总结

> 记录于 2026-10-10。基于 9 个开源音频大模型在 UAbench（水声信号三级层级分类，2600 样本）上的三轮评测实测，沉淀**实例配置、环境版本、每个模型的运行参数与踩坑经验**，目标是让后续复现/维护可以照表操作，少走弯路。
>
> 配套完整结果见 `UAbench_九模型评测结果汇总_20261010.md`（在 `outputs/` 下），本文只讲"怎么跑起来 + 踩过哪些坑"。

---

## 0. 一句话结论

- 9 个模型分布在 **3 套实例**上：实例① = AutoDL（`autodl1`，`torch 2.5.1/tf 4.54.0`，跑需要兼容补丁的 3 个模型）；实例② = AutoDL（`autodl`，`torch 2.6.0/tf 5.14.1`，跑 3 个不需要补丁的模型）；实例③ = SeetaCloud A800 80GB 双卡（先 `2.6.0/5.14.1` 跑 Gemma-12B，再降级到 `2.5.1/4.54.0` 跑 Voxtral）。
- **只有实例①的 3 个模型（Qwen2.5-Omni / Qwen2-Audio / Aero-1-Audio）需要 `inference.py` 的 3 处 transformers≥4.54 兼容补丁**；其余 6 个模型在各自环境下直接能跑。
- 仓库代码改动已封存于本地提交 `aa48f24`（未 push）。服务端仓库非 git 跟踪，改动需手动同步。

---

## 1. 三套实例与硬件/镜像总览

| 实例 | 服务商 | 卡型与显存 | 运行时镜像 / CUDA | SSH host（见 `~/.ssh/config`） | 本次状态 |
|---|---|---|---|---|---|
| ① | AutoDL（region-9） | 双卡 **40GB** 级（实测 Omni 峰值 39.9/40GB，推断 A100/A800-40G） | AutoDL 标准 PyTorch 镜像，CUDA 12.4 运行时（torch 预编译 `cu124`）；具体镜像名见 AutoDL 控制台 | `autodl1`（端口 49813） | 已开（探针时 GPU 已卸载，"No devices found"，但环境/代码都在） |
| ② | AutoDL（region-9） | 双卡（卡型未单独测量，同 AutoDL 家族） | AutoDL 标准 PyTorch 镜像，CUDA 12.4 运行时（`cu124`） | `autodl`（端口 30799） | 已开 |
| ③ | SeetaCloud | **A800 80GB 双卡**（实测 Voxtral-Small 峰值 71.4/80GB） | SeetaCloud 标准 PyTorch 镜像，CUDA 12.4 运行时（`cu124`） | `seeta`（端口 27247，connect.nma1.seetacloud.com） | 关闭（本次未开） |

> 注：AutoDL / SeetaCloud 的"具体镜像名"在各自控制台实例详情里，SSH 内拿不到。复现时选同代 PyTorch + CUDA 12.x 的镜像即可，版本以第 2 节为准。

---

## 2. 各实例 Python / 深度学习栈版本（实测，2026-10-10 探针）

服务端统一用 **conda `base` 环境**（`/root/miniconda3/bin/python`，Python 3.12.3），没有单独的评测虚拟环境。

| 实例 | torch | transformers | Python | soundfile | librosa | accelerate |
|---|---|---|---|---|---|---|
| ① `autodl1` | **2.5.1+cu124** | **4.54.0** | 3.12.3 | 0.14.0 | 1.0.0 | 1.15.0 |
| ② `autodl` | **2.6.0+cu124** | **5.14.1** | 3.12.3 | 0.14.0 | 1.0.0 | 1.15.0 |
| ③ `seeta` | 初始 **2.6.0 / 5.14.1** → 降级 **2.5.1 / 4.54.0** | 同上 | 3.12.3 | 0.14.0 | 1.0.0 | 1.15.0 |

**关键经验**：实例① 与 实例③降级后 是同一套"兼容栈"（`2.5.1 / 4.54.0`）；实例② 用的是**更新的** `5.14.1`。两套栈都能跑通，区别只在于——凡是走 Qwen2-Audio 家族 forward 的模型（Qwen2-Audio / Aero-1-Audio）和 Qwen2.5-Omni，必须在 `4.54.0` 下配合第 4 节的补丁；Voxtral 在 `5.14.1` 下会出问题，所以实例③要降级。

> 降级脚本（实例③）：`downgrade_to_voxtral.sh` → `torch 2.6.0→2.5.1`、`transformers 5.14.1→4.54.0`。新装栈走阿里云 pip 镜像，PyTorch 走官方 `download.pytorch.org` 反而快。

---

## 3. 九个模型的实例分配 + 运行参数表（核心）

> "峰值显存"为实测；"是否需要补丁"指第 4 节的 3 处改动。`--batch-size` 对 Qwen2-Audio / Aero **无效**（后端无 `generate_batch` 覆写，逐条循环），写了也不报错只是不生效。

| # | 模型 | 后端 | 实例 | 卡 / 显存 | 环境(torch/tf) | batch | 实测峰值显存 | 需补丁 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Qwen2.5-Omni** | `qwen25_omni` | ① autodl1 | 双卡 40GB | 2.5.1 / 4.54.0 | 32 | **39.9 / 40 GB (97%)** | 是（补丁①） | 双卡 batch32 是 40GB 卡实际上限；最强，ΔL3 +4.73 明显用上音频 |
| 2 | **Qwen2-Audio** | `qwen2_audio` | ① autodl1 | 双卡 40GB | 2.5.1 / 4.54.0 | 32（无效） | ~19 GB / 条 | 是（补丁②） | 逐条循环，GPU 利用率 ~50%；L1 两条件全同 38.46 |
| 3 | **Aero-1-Audio** | `aero1_audio` | ① autodl1 | 双卡 40GB | 2.5.1 / 4.54.0 | 64（无效） | 单样本低（~5GB 量级） | 是（补丁②+③） | 远程代码导入已移除的 `Qwen2AudioFlashAttention2`；L1 61.31/61.42 |
| 4 | **AF-Next** | `af_next` | ② autodl | 双卡 | 2.6.0 / 5.14.1 | 16（真 batch） | — | 否 | L1 最高 87.58；ΔL3 +0.0235 |
| 5 | **Gemma-4-E2B** | `gemma4`(e2b) | ② autodl | 双卡 | 2.6.0 / 5.14.1 | 64（真 batch） | — | 否 | L1 51.62 / 有效子集 57.16 |
| 6 | **MiDashengLM** | `midashenglm` | ② autodl | 双卡 | 2.6.0 / 5.14.1 | 32（真 batch） | — | 否 | ΔL3 = 0（对音频不敏感） |
| 7 | **Gemma-4-12B** | `gemma4`(12b) | ③ seeta A800 | A800 80GB 双卡 | 2.6.0 / 5.14.1（先跑） | 96（128 OOM） | **64.5 / 80 GB**（128 约 78GB OOM） | 否 | 44.5% 拒答"未提供音频"；L1 27.58 / 有效子集 49.65；重复验证可复现 |
| 8 | **Voxtral-Small-24B** | `voxtral` | ③ seeta A800 | A800 80GB 双卡 | 2.5.1 / 4.54.0（降级后） | 128 | **71.4 / 80 GB** | 否 | ΔL1 +9.42pp（音频产生**负向**影响，静音反而更高）；L3 +2.15pp（CI 不含 0） |
| 9 | **Voxtral-Mini-3B** | `voxtral` | ③ seeta A800 | A800 80GB 双卡 | 2.5.1 / 4.54.0（降级后） | 128 | — | 否 | ΔL1 = 0；T1 预测 2600 条 100% 相同（对音频完全不敏感） |

**复现要点**：
- 实例③ 必须**先跑 Gemma-12B（在 2.6.0/5.14.1），再降级，再跑 Voxtral**，顺序别反——Voxtral 在 5.14.1 下起不来。
- Gemma-4-12B 的 batch 别超过 96（128 直接 OOM）；Voxtral-Small batch128 峰值 71.4GB，离 80GB 有余量但别再加。
- Qwen2.5-Omni batch32 已顶到 40GB 卡 97%，**不要再加 batch**。

---

## 4. transformers ≥ 4.54 兼容性补丁（必读，决定能否跑 Qwen2.5-Omni / Qwen2-Audio / Aero）

根因：transformers 4.54 做了两处破坏性改动，正好命中这 3 个后端。补丁都已写入 `testsite/core/inference.py` 并提交为 `aa48f24`。

**补丁① · Qwen2.5-Omni eos/pad 修复（`Qwen25OmniBackend`）**
- 现象：checkpoint 的 `generation_config.json` 只带 `_from_model_config`，eos/pad 藏在嵌套 `thinker_config` 里 → `GenerationConfig` 的 `eos_token_id`/`pad_token_id` 为空 → `generate()` 没有停止符，一直解码到 `max_new_tokens`，输出整段"Human:" 幻觉，单字母解析器全部拒掉 → 指标全废（修复前 2600 条全 invalid）。
- 修法：从 tokenizer 把 eos/pad 填进 `self.model.generation_config`，并在 `generate()` 显式传 `eos_token_id`/`pad_token_id`。

**补丁② · forward 未声明 kwarg 过滤（`_filter_unsupported_forward_kwargs`）**
- 现象：`generate()` 会向 `model(**inputs)` 注入 `cache_position`；Qwen2-Audio / Aero 的 `forward` 未声明该参数 → `TypeError: ... got an unexpected keyword argument 'cache_position'`。
- 修法：模块级 helper，用 **`functools.wraps` 捕获绑定方法 `original = model.forward`**（不能用 `type(model).forward` 未绑定版本，否则被 `model.forward(...)` 调用时缺 `self` 直接崩），按其真实签名过滤掉未声明 kwarg 后透传。底层语言模型在 `cache_position=None` 时会自行推算位置，所以静默丢弃不影响解码。
- **必须 `functools.wraps`**：否则 `inspect.signature` 变 `(*args, **kwargs)`，transformers 的 `_validate_model_kwargs` 会把 `input_features` 判成"未被使用"而报 `ValueError`。
- 在 `Qwen2AudioBackend._load_model` 与 `Aero1AudioBackend._load_model` 各自 `return` 前调用 `_filter_unsupported_forward_kwargs(model, "<label>")`。

**补丁③ · Aero-1-Audio 的 `Qwen2AudioFlashAttention2` 占位注入（`Aero1AudioBackend._load_model`）**
- 现象：Aero 远程代码 `modeling_aero.py` 导入 `Qwen2AudioFlashAttention2`，而 transformers 4.54 **已移除**该类 → `ImportError`。
- 修法：在 `from_pretrained` **之前**，向 `transformers.models.qwen2_audio.modeling_qwen2_audio` 注入占位类 `_Qwen2AudioFlashAttention2Shim(_qa.Qwen2AudioAttention)`（**必须继承真实注意力类**，不能写成独立报错类，否则会污染其 forward）。该类只在 `_attn_implementation == "flash_attention_2"` 时使用，评测用 `sdpa` → 占位永不触发，导入即可通过。

> 备份（实例① 服务端 `/root/`）：`inference.py.pre_omni_patch`、`inference.py.pre_forward_compat`。

---

## 5. 评测运行流程与脚本

**脚本位置**：`testsite/scripts/`
- `run_eval.py`：加载 JSONL → 三轮对话 → 解析 → 计分 → 报告。
- `run_multigpu.sh`：批量启动封装，支持环境变量覆盖：`DATA_PATH` / `AUDIO_ROOT` / `OUTPUT_ROOT`（默认 `eval_results/$backend`）。
- `generate_silent_control.py`：生成静音对照素材。
- `compare_silent_control.py`：配对比较原音频 vs 静音。

**数据集**：`testset_export/sft_test_highquality.jsonl`，**2600 样本**（13 类 × 200）。每条内嵌 3 个 human 问题、`_gt`、`_meta`、`record_schema_version = ua_bench_record_v1`。正式评测**只用数据集中内嵌的 3 个问题**，缺问题直接报错。

**评测协议要点**：
- 输出必须是**单个 ASCII 选项字母**（大小写均可）；`A.`/`(A)`/`Answer: A`/类名/多选/全角一律 `invalid_format`。Raw output 原样保存。
- **级联门控**：T1（L1）错或无效 → 跳过 T2/T3，L2/L3 记 `cascade_error`；T1 对 → T2 + T3（T2 无论对错都发 T3）。级联与无效仍计入准确率分母。
- GT 只来自 JSONL 内嵌 `_gt`；旧结果无签名会被 `merge_predictions` 拒绝（需带 `--protocol <protocol_shard_*.json>`）。

**输出目录**：`eval_results/relabel_v1/<backend>/{original,silent}/run_<时间戳>_<id>/`，含 `merged/metrics.json`（权威）、`predictions.jsonl`、`protocol` 分片。

---

## 6. 静音对照实验（paired silent control）方法论

目的：判断模型到底**用没用上音频**（原音频 − 静音，差值不假设为正）。

- `generate_silent_control.py` 产出 `controls/<stamp>/`：`original.jsonl`（音频改**绝对路径**指向 `testset_export`）、`silent.jsonl`（相对 `audio/000000.wav`）、`pairs.jsonl`、`audio/`（2600 个静音 wav）、`generation.json`（含 `pairs_sha256`）。
- `compare_silent_control.py` **硬编码**聚合 `control/original.jsonl`；`integrity.py:64` 要求 `prediction.manifest_sha256 == sha256(manifest)`，否则报 "prediction was not produced from this manifest"。
- **大坑（已踩）**：已有"原音频" run 的 `manifest_sha256 = 248f827f`（源清单），而 `control/original.jsonl` 哈希 `= 049a9a35`（绝对路径改写版），二者不等 → 校验必失败 → 看似"必须重跑原音频"。**其实不用**：`generate_silent_control.py` 只是把源清单的 `audio` 改写成绝对路径，题目与 `_gt` 完全一致，且 `generation.json` 的 `source_manifest_sha256 = 248f827f` 恰等于已有 run 的 manifest。
- **解法（`controls/paired_reuse_20261010/`）**：把 `original.jsonl` **字节拷贝**源清单（sha=248f827f）→ 已有原音频预测即可通过 `integrity.py:64`；原音频用**软链**挂回真实文件（不复制 2.2GB）；只重跑**静音一半** + 比较。5200 个软链，不占磁盘。
- 差值 `delta = 原音频 − 静音`；报告给出 95% CI（按 `source_id` 聚类 bootstrap）与 McNemar 检验，CI 不含 0 才算"显著用上音频"。

---

## 7. 关键坑与经验（踩过的雷，按出现频率排）

1. **CRLF vs LF 假阳性**：本地文件是 Windows CRLF，服务端是 LF。`diff` 不加 `--strip-trailing-cr` 会把整篇判成"全不同"。**任何跨本地/服务端比对都必须 `diff --strip-trailing-cr`**。本次终审用递归 `diff -r --strip-trailing-cr -q` 确认实例① 整树 0 差异。
2. **`--batch-size` 对 Qwen2-Audio / Aero 无效**：这俩后端没覆写 `generate_batch`，继承基类逐条循环。GPU 利用率仅 ~50%，显存 19GB / 5.2GB 也印证单样本。真 batch 的只有 `qwen25_omni / voxtral / af_next / gemma4 / midashenglm`。
3. **A800 必须降级才能跑 Voxtral**：Voxtral 在 `transformers 5.14.1` 下起不来；实例③ 先 `2.6.0/5.14.1` 跑完 Gemma-12B，再 `downgrade_to_voxtral.sh` 降到 `2.5.1/4.54.0` 跑 Voxtral。
4. **Gemma-4-12B 44.5% 拒答**：输出"未提供音频"，整批 invalid。看真实能力要切到 **`turn1_parse_status = valid_option` 的有效子集**口径（L1 从 27.58 → 49.65）。并且两独立 run 指标完全一致 → 确定性可复现，不是偶发。
5. **`parse_tier_dist` 是误导性指标**：它是跨 T1/T2/T3 全轮次统计（含级联跳过的空输出），**不能**当 T1 解析失败率。T1 格式依从必须用 **`turn1_parse_status`**（`valid_option` / `invalid_format`）。例：Voxtral-Mini `parse_tier_dist` tier0=1064 看着像 41% 无效，但 T1 实际 invalid=0。
6. **`integrity.py` manifest 哈希陷阱**：见第 6 节，别一看到哈希不等就重跑原音频，先确认清单是怎么派生的（`source_manifest_sha256` 才是同源判据）。
7. **SSH 抖动**：AutoDL / SeetaCloud 连接偶发 `Connection closed / refused / timed out`。所有 scp/ssh 都套**重试循环**（3–5 次，间隔 2–3s）。`~/.ssh/config` 里 `Host autodl/autodl1/seeta` 用无密码专用密钥 `id_ed25519_autodl` + `BatchMode=yes`；**显式带 host:port 会触发密码提示而非密钥，非交互下必然失败**——一律用 config 里的 Host 别名。
8. **磁盘**：AutoDL 容器 93G/100G 已用（剩 ~7.7G），评测产物+模型缓存很占空间。跑完及时 `scp` 回本机并清实例，或让用户下载后删实例。

---

## 8. 结果速查（九模型，详见总报告）

> 口径：L1 为全量 2600 的 L1 准确率；"用上音频？"看 ΔL3 的 95% CI 是否不含 0（静音对照）。

| 模型 | L1 原 / 静 | L3|L2（条件指标） | ΔL3（原−静） | 用上音频？ | 一句话 |
|---|---|---|---|---|---|
| Qwen2.5-Omni | 72.54 / 51.35 | 13.12→8.38 | +4.73（CI 不含0） | **是（最强）** | McNemar 182/59 |
| AF-Next | 87.58 / — | — | +0.0235 | 是（弱） | L1 最高 |
| Qwen2-Audio | 38.46 / 38.46 | 7.31/7.73 | −0.0042（CI 含0） | 否 | L2\|L1=100% 退化模式 |
| Aero-1-Audio | 61.31 / 61.42 | 6.88/7.27 | −0.0038（CI 含0） | 否 | 贴多数类基线 |
| MiDashengLM | 57.50 / 58.60 | 21.95 | 0（CI 含0） | 否 | Δ=0 |
| Gemma-4-E2B | 51.62 / 57.16(有效) | 21.15 | — | — | 格式瑕疵型无效 |
| Gemma-4-12B | 27.58 / 49.65(有效) | 20.99 | +0.0069 | 弱 | 44.5% 拒答 |
| Voxtral-Small | 56.77 / 47.35 | 28.71 | +2.15pp（CI 不含0） | **是（负向）** | 静音反而更高 |
| Voxtral-Mini | 59.08 / 59.08 | 24.20 | — | 否 | T1 100% 相同 |

**基线**（论文必备对比）：随机基线 L1 50% / L2 33.3% / L3 7.7%；**多数类基线 L1 = 61.54%**（GT active 1600 / passive 1000）。九模型中**仅 AF-Next(87.58) 与 Qwen2.5-Omni(72.54) 高于多数类基线**（Aero 61.31 贴线，其余 6 个明显低于）。

---

## 9. 代码同步状态（重要）

- 本次 9 模型运行的**全部仓库代码改动** = `inference.py` 的 3 处补丁（第 4 节），已封存于本地提交 **`aa48f24`**（**未 push**）。
- 终审（2026-10-10）：实例①(`autodl1`) `testsite/` 整树递归比对本地** 0 差异**；实例②(`autodl`) `inference.py` 的 93 行差异**纯属缺失补丁的干净基线**（其模型不需要），4 个脚本全部一致；实例③ 只切环境、未改仓库代码。
- **服务端仓库不是 git 仓库**（`/root/autodl-tmp/underwater` 无 `.git`），改动靠手动 scp 同步；本地是 `github.com/zhesun33333/underwater` 的克隆。
- 若要从本地仓库重跑 Qwen2-Audio / Aero / Qwen2.5-Omni，**必须基于 `aa48f24`**，否则会再次撞 `cache_position` TypeError / Aero `ImportError`。

---

## 10. 论文红线（别踩）

- 测试集是 **"有利条件下的诊断子集"**，不能声称鲁棒性 / 全海况代表性。
- 结论收窄为 **"limited performance under the proposed evaluation protocol"**。
- 报告"用上音频"必须基于**静音对照的 CI / McNemar**，不能只看绝对指标。
- 所有指标口径统一用 `merged/metrics.json`；T1 格式依从用 `turn1_parse_status`，禁用 `parse_tier_dist` 当失败率。

---

## 11. 复现清单（checklist）

1. 起实例：实例①/② 用 AutoDL（① 需 `torch 2.5.1/tf 4.54.0`，② 用 `2.6.0/5.14.1`）；实例③ SeetaCloud A800（先 `2.6.0/5.14.1`，跑完 Gemma-12B 后降级到 `2.5.1/4.54.0`）。
2. 拉代码：本地 `aa48f24` 推到实例（或 scp `testsite/core/inference.py`），确认 3 处补丁在位（grep `_filter_unsupported_forward_kwargs|Qwen2AudioFlashAttention2|eos_token_id`）。
3. 数据集：`testset_export/sft_test_highquality.jsonl`（2600）就位；音频本体 2.2GB（用户自有，按需下载）。
4. 跑原音频：`run_multigpu.sh`，按第 3 节 batch（Omni≤32 / Gemma-12B≤96 / Voxtral≤128）。
5. 静音对照：用 `controls/paired_reuse_20261010/` 复用原音频预测，只重跑静音一半 + `compare_silent_control.py`。
6. 收尾：产物 `scp` 回本机；`diff --strip-trailing-cr` 核对代码；清实例磁盘。

---

*附：本文件与 `UAbench_九模型评测结果汇总_20261010.md`（结果报告）、`UAbench_全部结果_20261010_v2.tar.gz`（完整归档）配套使用。环境版本以第 2 节实测为准；GPU 卡型以第 1 节记录为准（实例①/② 探针时 GPU 已卸载，卡型依峰值显存推断）。*
