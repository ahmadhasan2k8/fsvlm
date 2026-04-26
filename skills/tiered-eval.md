---
name: tiered-eval
description: |
  Run zero-shot, few-shot, and full-train evaluations of a backend on the same fixed held-out
  split. Produces a comparison report that should show monotonic improvement across the three
  tiers. The proof-of-lift protocol — if monotonicity fails, either the framework is broken or
  the dataset is saturating, and either is worth knowing.
  TRIGGER when: user wants to verify fine-tuning actually helps on a given (backend, dataset,
  category) before sweeping all the N values.
  SKIP when: user already has zero-shot + full-train numbers and just wants intermediate N
  values (use /sweep). Skip when only inference is wanted (use /inspect).
inputs:
  - dataset (string, required)
  - category (string, required)
  - backend (string, default: gemma4-e4b)
  - few_shot_n (int, default: 30)
  - seeds (list of int, default: [42])
  - results_path (path, default: research/tiered_results.json)
eval_artifact: ${results_path}
pass_criteria:
  - file exists at eval_artifact
  - JSON has three rows for this (dataset, category, backend) tuple, one per tier:
    zero_shot, few_shot, full_train
  - rows share the same test split (same num_test value)
  - monotonicity: zero_shot.auroc <= few_shot.auroc <= full_train.auroc within ±0.02 noise
escalation: |
  If monotonicity fails:
    - zero_shot > few_shot: dataset is saturating OR few-shot training hurt; consult
      /expert-review with role=training-specialist
    - few_shot > full_train: training is overfitting at full data; reduce epochs or LoRA rank
    - all three identical: degenerate extractor; run /debug with focus=adapter
---

# Skill: tiered-eval

## Purpose

The "does fine-tuning actually help?" gate. Three runs on the same held-out split: the base
model alone (zero-shot), an adapter trained on N labeled examples (few-shot), and an adapter
trained on the full training pool (full-train). Result table shows the lift each tier provides.

## Procedure

```bash
python research/tiered_validation.py \
  --dataset "$dataset" \
  --category "$category" \
  --backend "$backend" \
  --few-shot-n "$few_shot_n" \
  --seeds "${seeds[@]}" \
  --results-path "$results_path"
```

The driver:
1. Splits the dataset's test set with a fixed seed (50/50 holdout for MVTec, official protocol
   for VisA, stratified for DeepPCB) — same split for all three tiers.
2. Runs zero-shot inference on the test split with the bare backend.
3. Trains a few-shot adapter on `few_shot_n` examples from the training pool, evaluates.
4. Trains a full-train adapter on the whole training pool, evaluates.
5. Appends three rows to the results JSON, all sharing the same `tier_run_id` so they can be
   compared cleanly.

Output schema (one row per tier):
```json
{
  "dataset": "mvtec", "category": "hazelnut", "backend": "gemma4-e4b",
  "tier": "few_shot",
  "n_samples": 30, "seed": 42,
  "auroc": 0.941, "f1": 0.918,
  "num_test": 119,
  "tier_run_id": "tiered-2026-04-25T21:00",
  "git_hash": "...", "recipe_version": "..."
}
```

## Self-evaluation

PASS if all three tiers exist for the chosen `(dataset, category, backend)`, share the same
`num_test`, and AUROC is monotonic within noise. Any monotonicity violation is FAIL with
`reason` named.

## Failure modes

- **Zero-shot AUROC ≈ 0.5**: extractor degenerate (typical of v0 keyword-match on categories
  where the model emits prose). Switch to v0.1 cascade and re-run.
- **Few-shot beats full-train**: classic overfitting at full data. Reduce epochs from 3 to 2,
  or shrink LoRA rank.
- **All three identical**: backend is stuck in majority-class prediction. /debug with
  focus=adapter; common cause is a wrong dataset reader (good/defect inverted).

## Adapting to your runtime

- **Claude Code**: drop into `~/.claude/skills/`
- **OpenAI Agents SDK**: register `tiered_eval(dataset, category, backend, few_shot_n, seeds,
  results_path)`
- **Plain shell**: invoke the Procedure block directly

## Pairs with

- **/sweep** — once monotonicity holds, run the full N-curve
- **/verdict** — classify per-tier rows against the prior baseline
- **/expert-review** — diagnose monotonicity violations
