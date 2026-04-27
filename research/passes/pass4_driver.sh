#!/usr/bin/env bash
# Pass 4 — Tier A completion. Staged 8 → 14 → 24 categories so the cheap
# fail-fast bet on the taxonomy rule runs first; expansion is conditional
# on the rule holding.
#
# Pre-registered predictions live in research/defect_taxonomy.json:
#   distinctive-dominant categories should show N=2 knee → AUROC ≥ ~0.85 by N=30
#   subtle-dominant categories should show flat / degenerate curves (metal_nut-style)
#   mixed should scatter
#
# N grid: {2, 10, 30} × 3 seeds. Trims N=5 from the v0.3-curve grid since
# the headline (knee at N=2) only needs two anchor points + one midpoint.
#
# Usage:
#   bash research/passes/pass4_driver.sh           # runs stage 1 (8 cats)
#   bash research/passes/pass4_driver.sh stage2    # runs stage 2 (6 more cats)
#   bash research/passes/pass4_driver.sh stage3    # runs stage 3 (10 more cats)
#   bash research/passes/pass4_driver.sh all       # runs all three stages back-to-back
#
# The autoresearch loop reads pass4 from queue.json and dispatches the right
# stage based on which previous stages have rows in the results JSON.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

RESULTS=research/dataset_size_results.json
RECIPE=v0.3-tier-a
COMMON_ARGS=(
  --n-values 2 10 30
  --seeds 42 1337 7
  --label-source metadata
  --recipe-version "$RECIPE"
  --output "$RESULTS"
)

export PYTORCH_ALLOC_CONF=expandable_segments:True

stage_name=${1:-stage1}

stage1_minimum() {
  echo "=== [pass4-driver] STAGE 1 — Tier-A-minimum (8 cats × 3 N × 3 seeds = 72 cells) ==="
  echo "  Goal: fastest defensible test of the distinctive-vs-subtle taxonomy rule."
  echo "  Distinctive (4): mvtec/cable, mvtec/transistor, mvtec/tile, visa/fryum"
  echo "  Subtle (3):      mvtec/capsule, mvtec/screw, visa/macaroni1"
  echo "  Mixed (1):       visa/cashew"
  echo ""
  bash research/run_sweep.sh \
    --datasets mvtec \
    --categories cable transistor tile capsule screw \
    "${COMMON_ARGS[@]}"
  bash research/run_sweep.sh \
    --datasets visa \
    --categories fryum macaroni1 cashew \
    "${COMMON_ARGS[@]}"
  echo "=== [pass4-driver] Stage 1 complete at $(date -Iseconds) ==="
}

stage2_balanced() {
  echo "=== [pass4-driver] STAGE 2 — extend to 14 cats (+6 categories: 3 distinctive + 2 subtle + 1 mixed) ==="
  echo "  Distinctive added (3): mvtec/bottle, mvtec/toothbrush, visa/chewinggum"
  echo "  Subtle added (2):      mvtec/wood, visa/macaroni2"
  echo "  Mixed added (1):       visa/capsules"
  echo ""
  bash research/run_sweep.sh \
    --datasets mvtec \
    --categories bottle toothbrush wood \
    "${COMMON_ARGS[@]}"
  bash research/run_sweep.sh \
    --datasets visa \
    --categories chewinggum macaroni2 capsules \
    "${COMMON_ARGS[@]}"
  echo "=== [pass4-driver] Stage 2 complete at $(date -Iseconds) ==="
}

stage3_full() {
  echo "=== [pass4-driver] STAGE 3 — extend to 24 cats (full Tier A coverage, +10 categories) ==="
  echo "  Distinctive added (7): mvtec/carpet, mvtec/grid; visa/pcb1, pcb2, pcb3, pcb4, pipe_fryum"
  echo "  Subtle added (3):      mvtec/leather, mvtec/pill, mvtec/zipper"
  echo ""
  bash research/run_sweep.sh \
    --datasets mvtec \
    --categories carpet grid leather pill zipper \
    "${COMMON_ARGS[@]}"
  bash research/run_sweep.sh \
    --datasets visa \
    --categories pcb1 pcb2 pcb3 pcb4 pipe_fryum \
    "${COMMON_ARGS[@]}"
  echo "=== [pass4-driver] Stage 3 complete at $(date -Iseconds) ==="
}

case "$stage_name" in
  stage1) stage1_minimum ;;
  stage2) stage2_balanced ;;
  stage3) stage3_full ;;
  all)
    stage1_minimum
    stage2_balanced
    stage3_full
    ;;
  *)
    echo "unknown stage: $stage_name (expected stage1 | stage2 | stage3 | all)" >&2
    exit 1
    ;;
esac
