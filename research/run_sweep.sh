#!/usr/bin/env bash
# Wrapper — always run the dataset-size sweep inside the fsvlm conda env.
#
# Forwards all arguments to dataset_size_sweep.py. Writes a combined stdout+stderr log
# with timestamp under research/logs/ for later inspection.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/research/logs"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/sweep_${TS}.log"

# Activate conda env
# shellcheck source=/dev/null
# Activate your environment here (conda activate, venv source, etc.)
[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] && source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fsvlm

cd "$REPO_ROOT"

echo "[wrapper] Sweep starting at $(date -Iseconds)" | tee "$LOG"
echo "[wrapper] Args: $*" | tee -a "$LOG"

# Run unbuffered so tail -f shows live progress
python -u research/dataset_size_sweep.py "$@" 2>&1 | tee -a "$LOG"

echo "[wrapper] Sweep finished at $(date -Iseconds)" | tee -a "$LOG"
echo "[wrapper] Log: $LOG"
