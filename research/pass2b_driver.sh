#!/usr/bin/env bash
# Pass 2b — recipe bump on per-category winning arm from Pass 2a.
# Per-category winner from prior expert review (Option C):
#   - metadata label source for mvtec/hazelnut + mvtec/metal_nut
#   - agent label source for visa/candle
# Recipe: rank=16, alpha=8 (scale 0.5), LR=1e-4 — targeting metal_nut threshold collapse.
#
# Uses FSVLM_* env vars to override defaults without editing config.py — pydantic-settings
# BaseSettings picks these up automatically and each training subprocess inherits them.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
RESULTS=research/dataset_size_results.json

export FSVLM_DEFAULT_LORA_RANK=16
export FSVLM_DEFAULT_LORA_ALPHA=8
export FSVLM_DEFAULT_LEARNING_RATE=1e-4

echo "=== [pass2b-driver] Recipe: rank=16, alpha=8 (scale 0.5), LR=1e-4 ==="
echo "=== [pass2b-driver] Arm 1/2 — metadata label source on mvtec/hazelnut + metal_nut ==="
bash research/run_sweep.sh \
  --datasets mvtec \
  --categories hazelnut metal_nut \
  --n-values 30 \
  --seeds 42 1337 \
  --label-source metadata \
  --recipe-version v0.2-rank16-lr1e4-fix \
  --output "$RESULTS"

echo "=== [pass2b-driver] Arm 2/2 — agent label source on visa/candle ==="
bash research/run_sweep.sh \
  --datasets visa \
  --categories candle \
  --n-values 30 \
  --seeds 42 1337 \
  --label-source agent \
  --recipe-version v0.2-rank16-lr1e4-fix \
  --output "$RESULTS"

echo "=== [pass2b-driver] Pass 2b complete at $(date -Iseconds) ==="
