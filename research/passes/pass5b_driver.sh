#!/usr/bin/env bash
# Pass 5b — ICL ablation. The single most important test for the paper's viability.
# Runs Gemma 4 in in-context-learning mode on the same 3 categories Pass 3 used,
# at matched N ∈ {2, 4, 8}, 3 seeds, no weight updates. Compares head-to-head
# against Pass 3's fine-tuned numbers.
#
# If fine-tuned AUROC > ICL AUROC by >= 0.02 on the majority of cells → few-shot
# fine-tuning IS a real contribution; sprint to NeurIPS E&D continues.
#
# If fine-tuned ≈ ICL (within 0.02) → reviewer critique is validated; paper pivots
# to extractor methodology + description quality + deployability + taxonomy.
# STOP the 15-day sprint, target CVPR 2027 VAND with honest reframing.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== [pass5b-driver] ICL baseline — 3 cats × 3 N values × 3 seeds = 27 runs ==="
echo "=== [pass5b-driver] Started at $(date -Iseconds) ==="

# Activate your environment here (conda activate, venv source, etc.)
[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] && source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fsvlm

python -u research/icl_baseline.py \
  --datasets mvtec visa deeppcb \
  --categories hazelnut candle pcb \
  --n-shots 2 4 8 \
  --seeds 42 1337 7 \
  --output research/dataset_size_results.json

echo "=== [pass5b-driver] Pass 5b complete at $(date -Iseconds) ==="
