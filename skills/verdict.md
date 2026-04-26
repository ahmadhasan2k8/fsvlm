---
name: verdict
description: |
  Classify result rows against a prior baseline using anti-Goodhart tolerances. Assigns one of
  four statuses per cell: `new_baseline` | `keep` | `noop` | `discard`. Pure computation — does
  not edit the working tree, run git, or call the model. The next skill (or a human) decides
  what to act on.
  TRIGGER when: /sweep just appended new rows and you want to know which ones are real wins,
  no-ops, or regressions.
  SKIP when: there are no unclassified rows in the results log.
inputs:
  - results_path (path, default: research/dataset_size_results.json)
  - recipe_version (string, optional) — classify only this cohort; default: all unclassified
  - write (bool, default: true) — mutate row statuses in-place; if false, dry-run to stdout
eval_artifact: ${results_path}
pass_criteria:
  - every previously-unclassified row in the targeted cohort now has status in
    {new_baseline, keep, noop, discard}
  - the row count of the file did not change (verdict only mutates `status`, never adds/removes)
  - JSON is still valid
escalation: |
  /verdict is mechanical. It does not escalate by itself. Surface its summary to:
    - the user, if running interactively
    - /expert-review, if running inside /autoresearch — domain expert decides whether the
      verdict's "keep" is meaningful
    - the calling skill, which may roll back recipe files on `discard`
---

# Skill: verdict

## Purpose

Turn raw measurement rows into actionable status labels using fixed numerical thresholds. The
load-bearing anti-Goodhart primitive: `noop` is a status, not silent acceptance.

## Procedure

```bash
python research/verdict.py \
  --results "$results_path" \
  ${recipe_version:+--recipe-version "$recipe_version"} \
  ${write:+--write}
```

Verdict rules (defined in `research/verdict.py` constants):

| Status         | Trigger                                          |
|----------------|--------------------------------------------------|
| `new_baseline` | No prior baseline exists for this cell           |
| `keep`         | AUROC or F1 lift ≥ `ABS_LIFT` (default 0.03) vs baseline, beyond noise |
| `noop`         | All metrics within ±`NOOP_TOL` (default 0.02) of baseline — no behavioural change |
| `discard`      | Worse than baseline beyond `NOISE` (default 0.02)|

Rows are grouped by cell `(dataset, category, n_samples, backend)` and aggregated by mean ±
stdev across seeds before classification.

Output (printed to stdout regardless of `--write`):
```json
{
  "summary": {"new_baseline": 9, "keep": 2, "noop": 1, "discard": 0, "total": 12},
  "per_cell": [
    {"cell": ["mvtec", "hazelnut", 30, "gemma4-e4b"],
     "n_runs": 3, "mean_auroc": 0.943, "std_auroc": 0.005,
     "baseline_recipe": "v0.1-extractor-fix", "baseline_mean_auroc": 0.941,
     "status": "noop", "reason": "AUROC within ±0.02 of baseline"}
  ]
}
```

## Self-evaluation

PASS if every targeted row has a status field in the four-element set after the run, and the
file's row count is unchanged.

## Tuning

The thresholds (`ABS_LIFT`, `NOOP_TOL`, `NOISE`) are constants at the top of `research/verdict.py`.
Tune per project — they are calibrated for AUROC / F1 in the 0.7–0.99 range on small test sets.
For other metrics or larger test sets, retune.

## Failure modes

- **All rows classified `new_baseline`**: no prior cells with the same `(dataset, category, n,
  backend)` key — expected for a first sweep.
- **All rows classified `keep`**: every measurement beat baseline by ≥0.03 — could be a real
  wholesale improvement, or a calibration shift in the extractor. Sanity-check with /tiered-eval.
- **Mass `discard`**: usually means the recipe regressed. Roll back via `git checkout` of the
  recipe files; the `verdict` skill itself does not mutate the working tree.

## Adapting to your runtime

- **Claude Code**: drop into `~/.claude/skills/`
- **OpenAI Agents SDK**: register `verdict(results_path, recipe_version, write)`
- **Plain shell**: `python research/verdict.py ...`

## Pairs with

- **/sweep** — produces the rows verdict classifies
- **/expert-review** — separates "is this number better?" (verdict) from "is it meaningful?"
- **/autoresearch** — uses verdict's output to decide the next pass
