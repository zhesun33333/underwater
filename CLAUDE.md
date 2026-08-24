# Underwater Acoustic LLM — Project Memory

## Project overview
- Goal: Evaluate open-source audio LLMs on underwater acoustic signal classification (3-level hierarchy)
- Classification: L1 (active/passive) → L2 (pulse/comm/ship_noise) → L3 (13 classes)
- 13 L3 classes: CW, LFM, HFM, 2FSK, 4FSK, BPSK, QPSK, OFDM, cargo, cruise, fishing, warship, underwater_target
- Test set: 2600 high-quality samples (200/class), exported to `testset_export/`

## Directory structure
```
underwater/
├── archive/                    # Synthesis pipeline (data generation)
│   ├── config.yaml / config_ship.yaml
│   ├── pipeline_step1_channel.py / pipeline_ship_step1_channel.py
│   ├── pipeline_step2_qa.py / pipeline_ship_step2_qa.py
│   ├── filter_test_set.py     # Filter high-quality test set, embed _gt labels
│   ├── export_test_set.py     # Export WAV + JSONL as tar
│   ├── diagnose_snr.py / diagnose_ssp.py
│   ├── dataset_stats.py       # Dataset statistics (fixed performance + progress)
│   ├── run_pipeline.sh
│   ├── utils/ (bellhop_runner, json_parser, ssp_sampler)
│   └── raw_data/ (PulseCom + 05_ship_radiated_noise)
├── testsite/                   # Evaluation framework
│   ├── config/eval_config.yaml # Taxonomy, prompts, model config
│   ├── core/
│   │   ├── inference.py        # 6 backends: mock, qwen2, aero1, voxtral, af_next, kimi
│   │   ├── loader.py           # DataLoader (embedded _gt support)
│   │   ├── parser.py           # Output → L1/L2/L3 extraction
│   │   └── scorer.py           # Metrics, source-stratified (tertile), reasoning quality
│   ├── eval/multi_turn.py      # T1→T2→T3 orchestration
│   ├── reporting/
│   │   ├── report.py           # Markdown report + JSON metrics
│   │   └── visualize.py        # 10 chart figures
│   ├── scripts/
│   │   ├── run_eval.py         # CLI (--backend --model-id --data --audio-root)
│   │   └── download_models.sh  # Model download script
│   └── ablation/ (text_only, prompt_robustness)
└── testset_export/             # Packaged test set (JSONL + audio/)
```

## Key decisions & fixes made
1. **Unified L3 term pool**: _L3_TERMS in pipeline_step2_qa.py, pipeline_ship_step2_qa.py, scorer.py all synchronized after quality review
2. **Embedded GT in JSONL**: filter_test_set.py embeds `_gt` and `_meta` fields; loader.py reads them first
3. **Source-stratified evaluation**: replaced fixed-threshold SNR bins with data-driven tertile splits (PulseCom by TL, Ship by SNR), removed meaningless SSP stratification
4. **Visualization**: 10 charts, all English labels (Chinese font missing on Linux), PR scatter handles overlapping origin points
5. **cascade_skipped in JSON**: report.py saves it; visualize.py falls back to L1 accuracy if missing
6. **PulseCom SNR = TL**: not a bug — PulseCom signals are clean (no background noise), SNR computation measures channel attenuation
7. **SSP complexity constant**: all 100 profiles from same shallow-water region, gradient variance ~0.0038 for all
8. **Ship SNR is real**: line-spectrum to continuous-background ratio from original data generation (~ -22 to -8 dB)
9. **export_test_set.py**: now only exports WAV + JSONL (no JSON metadata files), since GT is embedded
10. **transformers version**: 4.54.0 (upgraded from 4.45.2 for Aero-1-Audio compatibility); Qwen2-Audio local path issue fixed by env vars

## Synthesis pipeline flow
1. `pipeline_step1_channel.py --config config.yaml` → BELLHOP convolution → `processed_audio/PulseCom/`
2. `pipeline_ship_step1_channel.py --config config_ship.yaml` → broadband CIR → `processed_audio/05_ship_radiated_noise/`
3. `pipeline_step2_qa.py --config config.yaml` → 3-turn QA → `qa_output/sft_{train,val,test}.jsonl`
4. `pipeline_ship_step2_qa.py --config config_ship.yaml` → `qa_output/sft_ship_{train,val,test}.jsonl`
5. Merge: `cat qa_output/sft_*.jsonl > dataset/sft_*.jsonl`
6. `dataset_stats.py` → statistics

## Evaluation flow
1. `filter_test_set.py` → `dataset/sft_test_highquality.jsonl` (2600 samples, with `_gt`)
2. `export_test_set.py --no-compress` → `testset_export.tar` (JSONL + WAV)
3. `python -m testsite.scripts.run_eval --data ... --audio-root ... --backend <name> --model-id <path>`
4. `python -m testsite.reporting.visualize --metrics ... --model-name "..."`

## Model backends implemented
| Backend key | Model | Size | Status |
|---|---|---|---|
| `mock` | MockModel | - | Test framework |
| `qwen2_audio` | Qwen2-Audio-7B | 7B | Done (L3≈8.0%) |
| `aero1_audio` | Aero-1-Audio (LMMs-Lab) | 1.5B | Done (L3≈8.1%) |
| `voxtral_mini` | Voxtral-Mini-3B (Mistral) | 3B | Done |
| `voxtral_small` | Voxtral-Small-24B-2507 (Mistral) | 24B | Vast.ai 4x A100 80GB scripts ready; GPU validation pending |
| `af_next` | Audio-Flamingo-Next (NVIDIA) | 7B | Gated — needs HF auth |
| `kimi_audio` | Kimi-Audio-7B (Moonshot) | 10B | Needs Ampere GPU (V100 doesn't work) |

## Next model candidates
- Qwen2.5-Omni-7B (ModelScope: `Qwen/Qwen2.5-Omni-7B`) — best replacement for AF-Next
- Eureka-Audio-Instruct (ModelScope: `lys1999/Eureka-Audio-Instruct`) — 1.7B MoE, Baidu

## Server environment
- AutoDL, CUDA 13.0, PyTorch 2.5.1+cu124
- transformers 4.54.0, flash-attn installed
- HF access via `HF_ENDPOINT=https://hf-mirror.com` (may timeout)
- ModelScope primary download source
- Model paths: `~/autodl-tmp/models/models/<repo>/snapshots/master`
- Run with: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`

## Model paths on server
```
# Qwen2-Audio-7B
$HOME/autodl-tmp/models/models/Qwen2-Audio-7B-Instruct/snapshots/master
# Aero-1-Audio
$HOME/autodl-tmp/models/lmms-lab/Aero-1-Audio
# Voxtral-Mini-3B
$HOME/autodl-tmp/models/mistralai/Voxtral-Mini-3B-2507
# Voxtral-Small-24B (Vast.ai default)
/workspace/models/Voxtral-Small-24B-2507
# Kimi-Audio-7B (ModelScope)
$HOME/autodl-tmp/models/models/moonshotai--Kimi-Audio-7B-Instruct/snapshots/master
```

## Known quirks
- VoxtralBackend: encoder-decoder (VoxtralForConditionalGeneration, not AutoModelForCausalLM), native tensor batching, audio format is `{"type": "audio", "path": "..."}`
- Aero1AudioBackend: needed `sed -i 's/Qwen2AudioFlashAttention2/Qwen2AudioAttention/' modeling_aero.py` (transformers rename)
- KimiAudioBackend: needs `from kimia_infer.api.kimia import KimiAudio`, FlashAttention (Ampere+), additionally downloads whisper-large-v3
- All backends: `--backend` and `--model-id` CLI args override YAML config
