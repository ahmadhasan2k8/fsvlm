---
name: validate
description: |
  Evaluate a trained adapter on a held-out test dataset. Produces metrics (AUROC, F1, precision,
  recall, accuracy, threshold), a confusion matrix, a confidence histogram, and a failure
  gallery. Distinct from /inspect: validate computes ground-truth-aware metrics; inspect just
  emits predictions.
  TRIGGER when: user has an adapter and a labeled test set and wants metrics + diagnostics.
  SKIP when: user wants predictions only (use /inspect) or wants to run a sweep across N values
  (use /sweep).
inputs:
  - adapter (path, required) — adapter directory
  - test_images (path, required) — labeled test set (LabelReader-recognised)
  - backend (string, default: gemma4-e4b)
  - output_dir (path, default: $adapter/validation/) — where to write the report
  - report_format (string, default: html) — html | json | both
eval_artifact: ${output_dir}/validation_report.json
pass_criteria:
  - file exists at eval_artifact
  - JSON keys: metrics, confusion_matrix, confidence_histogram, failure_gallery, summary
  - metrics has: auroc, f1, precision, recall, accuracy, threshold, num_test
  - num_test > 0 (real evaluation happened, not an empty split)
escalation: |
  If pass_criteria fails or metrics are surprising:
    - num_test == 0: test set is empty or LabelReader rejected everything; check /debug
    - auroc < 0.55 on a non-trivial test set: training did not transfer; consult expert-review
      with role=training-specialist + the recipe used in /train
    - confusion matrix collapsed (all predictions one class): degenerate model; same as above
---

# Skill: validate

## Purpose

Generate the report that goes into a paper / README / Slack post. Metrics + confusion matrix +
failure gallery + a plain-English summary of the model's typical mistakes.

## Procedure

1. Run the validation:
   ```bash
   fsvlm validate --adapter "$adapter" \
                  --images "$test_images" \
                  --backend "$backend" \
                  --output "$output_dir" \
                  --format "$report_format"
   ```

2. Inspect the produced JSON. Schema:
   ```json
   {
     "metrics": {"auroc": 0.94, "f1": 0.92, "precision": 0.93, "recall": 0.91,
                 "accuracy": 0.92, "threshold": 0.42, "num_test": 119},
     "confusion_matrix": [[58, 4], [5, 52]],
     "confidence_histogram": {"correct": [...], "incorrect": [...]},
     "failure_gallery": [
       {"image_path": "...", "predicted": "good", "actual": "defect",
        "confidence": 0.61, "reasoning": "VLM described the surface as uniform"}
     ],
     "summary": "Adapter achieves AUROC 0.94 on 119 held-out images. The most common
                 failure mode is missed defects in the lower-right quadrant — 4 of 9
                 misses share that location. Suggest adding more lower-right examples."
   }
   ```

3. If `report_format` includes html, an HTML report is also written for human review.

## Self-evaluation

PASS if all `pass_criteria` are met. Additionally compare against the adapter's training
`metrics.json`: if `validate.auroc < train.auroc - 0.10`, append `notes:
["large_train_test_gap"]` to the report — likely overfit to the training pool's split.

## Failure modes

- **`num_test == 0`**: the LabelReader returned no samples. Common cause: wrong path, wrong
  reader for the format, or the dataset's split file has no test rows.
- **All-good or all-defect predictions**: confusion matrix is one column. Run `/debug` with
  focus=adapter; this usually means the score extractor returned a constant.
- **HTML report empty / template error**: Jinja2 template lookup failed; check the adapter
  was produced by a fsvlm version compatible with the current install.

## Adapting to your runtime

- **Claude Code**: drop into `~/.claude/skills/`
- **OpenAI Agents SDK**: register `validate(adapter, test_images, backend, output_dir,
  report_format)`
- **Plain shell**: invoke the Procedure block directly
