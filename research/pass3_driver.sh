#!/usr/bin/env bash
# Pass 3 — AUROC-vs-N curve on 3 cleanest categories, extended with tiny-N at the low end.
# Tests the "tiger-analogy" hypothesis: does the curve lift meaningfully at N=2 or N=3 on
# distinctive-defect categories?
#
# N values: {2, 3, 5, 10, 20, 30, 40, 60, 100}
# - N=2 is "1-shot per class" (the cleanest tiger-analogy test)
# - N=1 dropped because balanced sampler gives 0-good + 1-defect → degenerate
# Categories: hazelnut (MVTec distinctive), candle (VisA distinctive), pcb (DeepPCB distinctive)
# - metal_nut dropped per Pass 2/6 degeneracy findings
# Seeds: {42, 1337, 7} — 3 seeds per cell to report mean ± stdev
# Label source: metadata (simpler paper story; within 0.01 of agent on candle per Pass 2a)
# Recipe: Pass 1 baseline (rank=8, LR=2e-4, alpha=8) — validated by Pass 2b as the best config
# Recipe_version: v0.3-curve (extractor v0.1 inherited via git a197a9b)

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
RESULTS=research/dataset_size_results.json

echo "=== [pass3-driver] AUROC-vs-N curve — 3 cats × 9 N values × 3 seeds = 81 runs ==="
echo "=== [pass3-driver] Started at $(date -Iseconds) ==="

bash research/run_sweep.sh \
  --datasets mvtec visa deeppcb \
  --categories hazelnut candle pcb \
  --n-values 2 3 5 10 20 30 40 60 100 \
  --seeds 42 1337 7 \
  --label-source metadata \
  --recipe-version v0.3-curve \
  --output "$RESULTS"

echo "=== [pass3-driver] Pass 3 complete at $(date -Iseconds) ==="
