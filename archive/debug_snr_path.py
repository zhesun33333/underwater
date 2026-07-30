"""
快速验证: 打印几个 PulseCom 样本的 SNR 计算路径
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.json_parser import load_jsonc, get_center_freq, get_geometry

PROCESSED_ROOT = Path("processed_audio")

# 找几个不同类型的 PulseCom 样本
samples_to_check = []

# 找一个 CW (短脉冲)
for p in (PROCESSED_ROOT / "PulseCom" / "jsonc" / "CW").glob("*.jsonc"):
    samples_to_check.append(("CW", p))
    break

# 找一个 2FSK (通信信号)
for p in (PROCESSED_ROOT / "PulseCom" / "jsonc" / "2FSK").glob("*.jsonc"):
    samples_to_check.append(("2FSK", p))
    break

# 找一个 HFM
for p in (PROCESSED_ROOT / "PulseCom" / "jsonc" / "HFM").glob("*.jsonc"):
    samples_to_check.append(("HFM", p))
    break

for label, jp in samples_to_check:
    meta = json.loads(jp.read_text(encoding="utf-8"))
    snr_stored = meta.get("bellhop_output", {}).get("snr_after_channel_db")
    tl_stored = meta.get("bellhop_output", {}).get("estimated_transmission_loss_db")
    signal_dur = meta.get("signal_duration_s")
    audio_dur = meta.get("audio_duration_s")
    cat = meta.get("signal_category")

    print(f"\n{'='*50}")
    print(f"{label}: {meta['id']}")
    print(f"  signal_category: {cat}")
    print(f"  signal_duration_s: {signal_dur}")
    print(f"  audio_duration_s: {audio_dur}")
    print(f"  stored SNR: {snr_stored}")
    print(f"  stored TL:  {tl_stored}")

    if signal_dur is not None:
        dur = float(signal_dur)
        total = float(audio_dur or 30)
        print(f"  signal_dur >= 0.85 * total: {dur} >= {0.85*total:.2f} → {dur >= 0.85*total}")
