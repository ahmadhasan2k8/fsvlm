---
name: train
description: |
  Fine-tune a fsvlm adapter on a folder of labeled images using QLoRA on top of unsloth. Wraps
  `fsvlm train` with optional auto-research LoRA-rank sweep and signed adapter export. Works for
  any registered ModelBackend and any LabelReader-compatible dataset.
  TRIGGER when: user has a labeled image dataset and wants a working detector adapter.
  SKIP when: user only wants inference with an already-trained adapter (use /inspect), or
  benchmark sweep across N values (use /sweep instead — it calls /train per cell).
inputs:
  - images (path, required) — dataset folder, CSV, JSON, or any LabelReader-recognised format
  - backend (string, default: gemma4-e4b)
  - epochs (int, default: 3)
  - lora_rank (int, default: 8)
  - learning_rate (float, default: 2e-4)
  - sweep (bool, default: false) — if true, runs the auto-research grid
  - adapter_name (string, optional) — override the auto-generated name
eval_artifact: ${adapter_dir}/metrics.json
pass_criteria:
  - file exists at eval_artifact
  - JSON keys present: f1, precision, recall, auroc, threshold, num_test
  - f1 >= 0.5 (better than random for binary classification on a balanced split)
  - auroc >= 0.55 (sanity: training did something beyond predict-majority-class)
escalation: |
  If pass_criteria fails:
    - f1 ~ 0.0 or 1.0: degenerate output, run /debug with focus=adapter
    - auroc < 0.55: training did not converge; consult expert-review with role=training-specialist
    - file missing: training crashed; check stderr for OOM, dataset-loader, or unsloth errors
---

# Skill: train

## Purpose

Take a folder of labeled images and produce a fine-tuned adapter with metrics that beat the
zero-shot baseline. The output is a portable adapter directory + a metrics JSON the next skill
can read.

## Procedure

1. (Optional) Sanity-check the dataset before training — runs DataAgent.prepare which reports
   image count per class, format mix, suspicious aspect ratios, and corrupt files:
   ```bash
   python -c "
   from fsvlm.agents.data_agent import DataAgent
   from fsvlm.config import FSVLMConfig
   r = DataAgent(FSVLMConfig()).prepare('$images').report
   print(f'{r.total_images} images: {r.good_count} good / {r.defect_count} defect')
   "
   ```

2. Run the training:
   ```bash
   fsvlm train --images "$images" \
               --backend "$backend" \
               --epochs "$epochs" \
               --lora-rank "$lora_rank" \
               --learning-rate "$learning_rate" \
               ${sweep:+--sweep} \
               ${adapter_name:+--adapter-name "$adapter_name"}
   ```
   Output: `~/.fsvlm/adapters/<adapter_name>/` containing `adapter_model.safetensors`,
   `adapter_config.json`, `metrics.json`, and a signed metadata file.

3. Read `metrics.json`:
   ```json
   {
     "f1": 0.94, "precision": 0.93, "recall": 0.95,
     "auroc": 0.96, "threshold": 0.42, "num_test": 119,
     "train_accuracy": 0.97, "val_accuracy": 0.94,
     "training_time_seconds": 488.2
   }
   ```

## Self-evaluation

PASS when all `pass_criteria` are met. Compare `train_accuracy - val_accuracy`:
if > 0.10, append `notes: ["overfitting_detected"]` to `metrics.json` so the next skill knows.

## Failure modes

- **OOM during training**: reduce `lora_rank` (try 4) or shrink `epochs`, or pick a smaller
  model variant via /setup.
- **f1 = train_majority_class_freq exactly**: the model is predicting the majority class only.
  Often fixed by ensuring the dataset has both classes represented in every epoch (check
  stratified split is on).
- **`auroc = 0.5` exactly**: extractor degenerate. The v0.1 cascade should prevent this; if
  using a custom extractor, check it returns calibrated scores.
- **Loss diverges (NaN)**: learning rate too high. Halve it.

## Adapting to your runtime

- **Claude Code**: drop into `~/.claude/skills/`
- **OpenAI Agents SDK**: register `train(images, backend, epochs, lora_rank, learning_rate,
  sweep, adapter_name)` as a tool
- **Plain shell**: invoke the Procedure block directly
