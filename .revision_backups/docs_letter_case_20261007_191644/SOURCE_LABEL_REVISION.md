# Source-based Active/Passive wording (source_origin_v2)

Active means a signal deliberately transmitted for detection or communication.
Passive means incidental noise radiated by an operating vessel or underwater vehicle.
It is a source category, not a receiver operating mode. Internal evaluation keys
(active/passive), A/B ordering, all 13 leaf labels, audio, and split rules are unchanged.

## Changed paths

- `source_label_prompts.py`: shared eight Turn 1 templates and validated answers.
- `archive/pipeline_step2_qa.py` and `archive/pipeline_ship_step2_qa.py`: use shared prompts and record `qa_prompt_version`.
- `archive/utils/json_parser.py`: passive descriptive label is `source-radiated`.
- `shared_terminology.py`: source-based wording in generated explanations.
- `testsite/config/eval_config.yaml`: corrected options, three default evaluation templates, and aliases.
- Evaluator, text-only, robustness, mock output, and statistics wording updated.
- New evaluation prediction records include the version and a protocol JSON sidecar.
  Default real-model output directories include the version to separate new runs.

## Existing data and results

Existing JSONL datasets and saved model results were NOT rewritten. Code changes
apply to future generation. Existing result metrics are not corrected-prompt results.
Original modified code/config files are preserved under `.revision_backups/`.
Use a controlled rerun to quantify the wording effect; do not relabel old metrics as v2.

## Regenerate QA without rerunning acoustics

From `archive/`, with configs pointing to the existing processed audio/metadata:

```sh
python pipeline_step2_qa.py --config config.yaml
python pipeline_ship_step2_qa.py --config config_ship.yaml
```

These commands overwrite the configured QA output files. Use separate qa_output
paths for v2 to keep old data available. Regenerate merged/filtered/exported QA
artifacts from these outputs before training or distributing the corrected dataset.
When comparing evaluation versions, keep the original 2600 audio sample IDs fixed.
The evaluator obtains prompts from its config rather than the dataset dialogues.

## Checks

```sh
python -m unittest discover -s archive/tests -p test_source_label_prompts.py -v
```
