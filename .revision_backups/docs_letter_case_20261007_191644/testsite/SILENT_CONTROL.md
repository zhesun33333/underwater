# 等时长、等采样率静音对照

在仓库根目录执行：

```powershell
python -m testsite.scripts.generate_silent_control --data dataset/sft_test_highquality.jsonl --audio-root processed_audio --output-prefix silent_control
```

脚本始终创建新的时间戳目录，不修改原音频或原 JSONL。输入须为非空、无损编码 WAV。保留每条音频的精确帧数、采样率、声道数和编码子类型，不裁剪、不重采样、不归一化。生成后逐块验证解码样本为零。

输出内容：

- `audio/`：静音 WAV。
- `silent.jsonl`：静音条件的固定问题、标签与元数据。
- `original.jsonl`：对应原音频清单，使用原文件的绝对路径，不复制原音频。该清单仅适用于能访问这些路径的机器，迁移机器时需要更新路径。
- `pairs.jsonl`：样本 ID、音频路径、帧数、采样率、声道数、子类型及两份音频的 SHA-256。
- `generation.json`：生成状态、源清单及两份条件清单的 SHA-256。仅 `status=complete` 表示生成与校验全部完成。

两份条件清单都可直接交给现有评测器。下面的 `CONTROL_DIR` 应替换为实际生成目录，后端与模型路径按实际运行环境填写：

```powershell
python -m testsite.scripts.run_eval --data CONTROL_DIR/original.jsonl --audio-root CONTROL_DIR --backend qwen25_omni --model-id MODEL_PATH --output-dir RESULTS/original
python -m testsite.scripts.run_eval --data CONTROL_DIR/silent.jsonl --audio-root CONTROL_DIR --backend qwen25_omni --model-id MODEL_PATH --output-dir RESULTS/silent
```

模型始终收到音频输入，静音条件不使用 `--ablation text_only`。两种条件保持相同模型、解码配置、固定问题、选项顺序、门控、解析与计分规则。按相同样本 ID 做后续配对分析，不把两个条件的预测混为一次实验汇总。各条件单独汇总时分别传入其自身清单。

## 评估口径待讨论

现有指标可分别计算原音频和静音条件，报告有符号的差值。静音的原标签仅作为配对参照，不代表静音具有该物理类别。

`invalid_format` 表示回答未满足单字母协议，不能直接称为拒绝。第一轮未通过导致的 `cascade_error` 也不能称为第二轮拒绝。是否增加人工核验或明确拒绝识别，需另行确定，本次生成器未修改评测指标或增加拒绝分类。
