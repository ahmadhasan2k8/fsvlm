#!/usr/bin/env bash
# Download the DeepPCB dataset.
# Official source: https://github.com/tangsanli5201/DeepPCB
# License: per-repo (verify at clone time — academic use).
#
# Produces: research/datasets/deeppcb/ with PCBData/ and trainval.txt/test.txt splits.
# Size: ~200 MB.

set -euo pipefail

DATASET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deeppcb"
REPO_URL="https://github.com/tangsanli5201/DeepPCB.git"

if [[ -d "$DATASET_DIR/.git" ]]; then
  echo "DeepPCB repo already cloned at $DATASET_DIR. Pulling latest..."
  git -C "$DATASET_DIR" pull --ff-only
  exit 0
fi

echo "Cloning DeepPCB..."
git clone --depth 1 "$REPO_URL" "$DATASET_DIR"

echo "Done. Contents:"
ls -la "$DATASET_DIR"
