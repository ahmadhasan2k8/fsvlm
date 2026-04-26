"""Verify fsvlm's data path can read a labeled folder dataset, without GPU or model load.

Runs four checks: FolderLabelReader, DataAgent.prepare, image loading, prompt rendering.
PASS → install is healthy. FAIL → run `fsvlm setup --check`.

Usage:
    python examples/quickstart/check_pipeline.py                    # uses /tmp/fsvlm-quickstart/
    python examples/quickstart/check_pipeline.py /path/to/your/data # uses your folder
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(data_path: Path) -> int:
    if not data_path.is_dir():
        print(f"FAIL — {data_path} is not a directory; run make_dataset.py first")
        return 1

    print("check_pipeline:")

    # 1. FolderLabelReader recognises the structure
    from fsvlm.readers.folder_reader import FolderLabelReader

    reader = FolderLabelReader()
    if not reader.supports(data_path):
        print(
            f"  FAIL — FolderLabelReader does not recognise {data_path} (expected good/ and defect/ subdirs)"
        )
        return 1
    samples = reader.read(data_path)
    good = sum(1 for s in samples if s.label == "good")
    defect = sum(1 for s in samples if s.label == "defect")
    print(f"  FolderLabelReader: {len(samples)} samples ({good} good / {defect} defect)")
    if good == 0 or defect == 0:
        print("  FAIL — both classes must have at least one sample")
        return 1

    # 2. DataAgent.prepare produces a valid train/val split
    from fsvlm.agents.data_agent import DataAgent
    from fsvlm.config import FSVLMConfig

    config = FSVLMConfig(base_dir=data_path / ".fsvlm-quickstart-cache")
    prepared = DataAgent(config).prepare(data_path)
    print(f"  DataAgent: train={len(prepared.train_samples)}, val={len(prepared.val_samples)}")
    if len(prepared.train_samples) + len(prepared.val_samples) != len(samples):
        print("  FAIL — train+val sample count does not match input")
        return 1

    # 3. Image loading + RGB conversion
    from fsvlm.utils.image import load_image, validate_image

    sample = samples[0].image_path
    img = load_image(sample, max_size=560)
    print(f"  Image loader: {img.size} {img.mode}")
    if img.mode != "RGB":
        print("  FAIL — image loader must return RGB")
        return 1
    if not validate_image(sample):
        print("  FAIL — validate_image rejected a known-good image")
        return 1

    # 4. Inspection prompt is well-formed
    from fsvlm.prompts.generic import INSPECTION_PROMPT

    print(f"  Inspection prompt: {len(INSPECTION_PROMPT)} chars, contains PASS/FAIL")
    if "PASS" not in INSPECTION_PROMPT or "FAIL" not in INSPECTION_PROMPT:
        print("  FAIL — inspection prompt missing PASS/FAIL anchor tokens")
        return 1

    print("PASS — pipeline is healthy")
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/fsvlm-quickstart")
    raise SystemExit(main(path))
