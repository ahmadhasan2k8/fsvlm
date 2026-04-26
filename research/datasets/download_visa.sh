#!/usr/bin/env bash
# Download the VisA (Visual Anomaly) dataset from Amazon Research.
# Official source: https://github.com/amazon-science/spot-diff
# License: Creative Commons Attribution 4.0 (CC BY 4.0)
#
# Produces: research/datasets/visa/ with 12 object categories
# Size: ~15 GB tarball, ~16 GB extracted
#
# Idempotent: safe to re-run; uses wget -c to resume, skips extraction if already done.

set -euo pipefail

DATASET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/visa"
ARCHIVE_URL="https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
ARCHIVE_NAME="VisA_20220922.tar"

mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

if [[ -d "candle" && -d "capsules" && -d "cashew" ]]; then
  echo "VisA appears already extracted at $DATASET_DIR. Skipping."
  exit 0
fi

echo "Downloading VisA archive (~15 GB)..."
wget -c --progress=dot:giga "$ARCHIVE_URL" -O "$ARCHIVE_NAME"

echo "Extracting..."
tar -xf "$ARCHIVE_NAME"

echo "Done. Categories present:"
ls -d */ | sed 's|/$||'

echo
echo "To save disk, remove the archive:"
echo "  rm $DATASET_DIR/$ARCHIVE_NAME"
