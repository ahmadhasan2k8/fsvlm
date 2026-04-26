#!/usr/bin/env bash
# Pass 3 resume after GPU OOM fragmentation crash.
# The parent sweep process accumulated 10 GB of GPU memory across inference cells,
# causing new training subprocesses' model-loads to OOM at candle N=60 onward.
# Fix: set PYTORCH_ALLOC_CONF=expandable_segments:True so torch uses a pool
# allocator resistant to fragmentation.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
RESULTS=research/dataset_size_results.json

export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== [pass3-resume] Resuming at $(date -Iseconds) ==="
echo "=== --resume will skip ~48 already-recorded rows ==="

bash research/run_sweep.sh \
  --datasets mvtec visa deeppcb \
  --categories hazelnut candle pcb \
  --n-values 2 3 5 10 20 30 40 60 100 \
  --seeds 42 1337 7 \
  --label-source metadata \
  --recipe-version v0.3-curve \
  --output "$RESULTS"

echo "=== [pass3-resume] Complete at $(date -Iseconds) ==="
