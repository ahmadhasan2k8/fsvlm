---
name: sweep
description: |
  Run a parametric (dataset × category × N × seed) sweep. The atomic step of any benchmark-
  driven research arc. Each cell is one /train + /validate, with its row appended to the
  results log carrying git_hash + recipe_version provenance. Adaptive — works for any backend
  and any LabelReader-recognised dataset.
  TRIGGER when: user wants to measure how performance varies with N labeled examples, OR
  compare backends/recipes on the same data.
  SKIP when: user wants a single train+eval run (use /train + /validate); single inference
  (use /inspect).
inputs:
  - dataset (string, required) — registered LabelReader name (mvtec | visa | deeppcb | …)
  - categories (list, required) — category slugs within the dataset
  - n_values (list of int, required) — labeled-example counts per cell
  - seeds (list of int, required) — at least 3 for credibility
  - backend (string, default: gemma4-e4b)
  - epochs (int, default: 3)
  - recipe_version (string, default: vN-<auto-incremented>) — cohort tag for grouping
  - results_path (path, default: research/dataset_size_results.json)
eval_artifact: ${results_path}
pass_criteria:
  - exactly len(categories) × len(n_values) × len(seeds) new rows appended
  - every new row has: dataset, category, n_samples, seed, auroc, f1, git_hash,
    recipe_version, status (set to null pre-verdict)
  - every row's git_dirty == false (clean tree at run time)
  - no row has auroc == 0.5 exactly (degenerate-extractor canary)
escalation: |
  If pass_criteria fails:
    - missing rows: a cell's training crashed; check stderr per cell, run /debug
    - any row has git_dirty == true: stop the sweep, commit the recipe change first
    - any row has auroc == 0.5 exactly: extractor degenerate, run /debug with focus=adapter
    - sweep completes but cells look weird: hand the new rows to /verdict, then
      /expert-review with role=training-specialist
---

# Skill: sweep

## Purpose

The data-collection step of an autoresearch pass. Produces N rows of comparable measurements
across the cells the hypothesis cares about. Pure data generation — no judgment, no decisions.

## Procedure

1. Verify the working tree is clean (provenance integrity):
   ```bash
   git diff --quiet && git diff --cached --quiet || { echo "uncommitted changes; commit first"; exit 1; }
   ```

2. Capture the recipe SHA + version:
   ```bash
   GIT_HASH=$(git rev-parse HEAD)
   RECIPE=${recipe_version:-v$(date +%s)}
   ```

3. Iterate every cell. For each `(category, n, seed)`:
   ```bash
   bash research/run_sweep.sh \
     --dataset "$dataset" \
     --category "$cat" \
     --n-value "$n" \
     --seed "$seed" \
     --backend "$backend" \
     --epochs "$epochs" \
     --recipe-version "$RECIPE" \
     --results-path "$results_path"
   ```
   This appends one row per call to the results log. The driver script is intentionally thin —
   it composes existing `fsvlm train` + `fsvlm validate` subprocess calls.

4. After the loop, count new rows:
   ```bash
   EXPECTED=$(( ${#categories[@]} * ${#n_values[@]} * ${#seeds[@]} ))
   ACTUAL=$(python -c "
   import json
   rows = json.load(open('$results_path'))
   new = [r for r in rows if r.get('recipe_version') == '$RECIPE']
   print(len(new))
   ")
   ```

## Self-evaluation

PASS if `ACTUAL == EXPECTED` AND every new row has the required keys + non-degenerate metrics.
FAIL with the missing-cell list otherwise.

## Failure modes

- **OOM mid-sweep**: a cell with high N exceeds VRAM. Reduce `epochs` or pick smaller backend.
- **Non-determinism between seeds is huge** (>0.05 stdev): the recipe is unstable; recommend
  rank reduction or LR halving in the next pass via /expert-review.
- **All cells produce the same metric**: dataset reader is broken or loading the same split
  regardless of N. Run /debug with focus=dataset.

## Adapting to your runtime

- **Claude Code**: drop into `~/.claude/skills/`
- **OpenAI Agents SDK**: register `sweep(dataset, categories, n_values, seeds, backend,
  epochs, recipe_version, results_path)`
- **Plain shell**: invoke `bash research/run_sweep.sh ...` directly per cell

## Pairs with

- **/verdict** — classify each new row vs the prior baseline
- **/tiered-eval** — check zero-shot → few-shot → full-train monotonicity on the same split
- **/expert-review** — domain-expert sanity check before locking the verdict
- **/autoresearch** — orchestrates this skill + the others into the full loop
