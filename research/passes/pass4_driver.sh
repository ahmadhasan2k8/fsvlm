#!/usr/bin/env bash
# Pass 4 — Tier A completion. Remaining 12 MVTec + 9 VisA categories at N=30.
# Also re-runs v0.1 zero-shots on all 21 new cats to cement the extractor-methodology finding.
#
# Pre-registered predictions live in research/defect_taxonomy.json. Distinctive-dominant
# cats are predicted to reach AUROC >= 0.85 on the trained arm; subtle-dominant predicted
# to reproduce metal_nut-style degeneracy; mixed to scatter.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
RESULTS=research/dataset_size_results.json

export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== [pass4-driver] MVTec 12 remaining categories ==="
bash research/run_sweep.sh \
  --datasets mvtec \
  --categories cable capsule carpet grid leather pill screw tile toothbrush transistor wood zipper \
  --n-values 0 30 \
  --seeds 42 1337 7 \
  --label-source metadata \
  --recipe-version v0.3-tier-a \
  --output "$RESULTS"

echo "=== [pass4-driver] VisA 9 remaining categories ==="
bash research/run_sweep.sh \
  --datasets visa \
  --categories capsules cashew chewinggum macaroni1 macaroni2 pcb2 pcb3 pcb4 pipe_fryum \
  --n-values 0 30 \
  --seeds 42 1337 7 \
  --label-source metadata \
  --recipe-version v0.3-tier-a \
  --output "$RESULTS"

echo "=== [pass4-driver] Pass 4 complete at $(date -Iseconds) ==="
