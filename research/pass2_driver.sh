#!/usr/bin/env bash
# Pass 2a driver — orchestrates zero-shot re-runs + 3-arm label-source ablation.
# Each sub-sweep writes to the same results JSON; --resume skips already-recorded rows.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
RESULTS=research/dataset_size_results.json

echo "=== [pass2-driver] Step 1/4 — zero-shot re-runs (v0.1 extractor) on 4 categories ==="
bash research/run_sweep.sh \
  --datasets mvtec visa deeppcb \
  --categories hazelnut metal_nut candle pcb \
  --n-values 0 \
  --seeds 42 \
  --label-source metadata \
  --recipe-version v0.1-extractor-fix \
  --output "$RESULTS"

for source in thin metadata agent; do
  echo "=== [pass2-driver] Training arm: label_source=$source ==="
  bash research/run_sweep.sh \
    --datasets mvtec visa \
    --categories hazelnut metal_nut candle \
    --n-values 30 \
    --seeds 42 1337 \
    --label-source "$source" \
    --recipe-version v0.1-extractor-fix \
    --output "$RESULTS"
done

echo "=== [pass2-driver] Pass 2a complete at $(date -Iseconds) ==="
