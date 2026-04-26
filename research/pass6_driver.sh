#!/usr/bin/env bash
# Pass 6 — metal_nut rescue via subtype-stratified sampling.
# Tests the last reasonable capacity-ish fix before declaring metal_nut an honest failure mode.
# Runs ONE metal_nut cell at N=30 × 2 seeds × metadata label source with Pass 1's recipe
# (rank=8, LR=2e-4) but with --stratified-subtypes to force all 4 defect subtypes into each
# training draw.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
RESULTS=research/dataset_size_results.json

# NOTE: no FSVLM_* env vars — we want Pass 1's recipe exactly.
# Only the sampling strategy differs from Pass 2a's metal_nut run.

echo "=== [pass6-driver] metal_nut rescue — stratified subtype sampling ==="
bash research/run_sweep.sh \
  --datasets mvtec \
  --categories metal_nut \
  --n-values 30 \
  --seeds 42 1337 \
  --label-source metadata \
  --stratified-subtypes \
  --recipe-version v0.1-stratified-metal-nut-rescue \
  --output "$RESULTS"

echo "=== [pass6-driver] Pass 6 complete at $(date -Iseconds) ==="
