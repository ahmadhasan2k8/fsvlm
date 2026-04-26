"""VisA (Visual Anomaly) benchmark reader.

VisA layout::

    visa/
      split_csv/
        1cls.csv          # object,split,label,image,mask
        2cls_fewshot.csv
        2cls_highshot.csv
      <object>/
        Data/
          Images/
            Normal/*.JPG
            Anomaly/*.JPG
          Masks/
        image_anno.csv

Labels in the split CSV: ``normal`` / ``anomaly``. Split values: ``train`` / ``test``.
We map to FSVLM's binary framing: ``normal`` → ``good``, ``anomaly`` → ``defect``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from fsvlm.exceptions import DatasetError
from fsvlm.interfaces import LabelReader
from fsvlm.registry import label_readers
from fsvlm.types import LabeledSample


@label_readers.register("visa")
class VisAReader(LabelReader):
    """Reader for the Amazon VisA anomaly-detection dataset."""

    def supports(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        return (path / "split_csv" / "1cls.csv").is_file()

    def read(self, path: Path) -> list[LabeledSample]:
        return read_visa(path)


def read_visa(
    path: Path,
    objects: list[str] | None = None,
    split: str | None = None,
    protocol: str = "1cls",
) -> list[LabeledSample]:
    """Read VisA images using one of the official split CSVs.

    Args:
        path: VisA root (contains split_csv/ and per-object directories).
        objects: Optional allow-list of object names (e.g. ``["candle", "pcb1"]``).
        split: Optional filter: ``"train"`` or ``"test"``. Default ``None`` = both.
        protocol: Which CSV to read: ``"1cls"`` (default), ``"2cls_fewshot"``, or
            ``"2cls_highshot"``.

    Returns:
        LabeledSample list with ``good`` / ``defect`` labels; sorted by path.
    """
    csv_path = path / "split_csv" / f"{protocol}.csv"
    if not csv_path.is_file():
        raise DatasetError(
            f"Missing VisA split CSV: {csv_path}",
            suggestion="Re-run experiments/datasets/download_visa.sh.",
        )

    allow = set(objects) if objects else None
    samples: list[LabeledSample] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            obj = row["object"]
            row_split = row["split"]
            row_label = row["label"]
            rel_image = row["image"]

            if allow is not None and obj not in allow:
                continue
            if split is not None and row_split != split:
                continue

            image_path = path / rel_image
            if not image_path.is_file():
                # VisA ships file extensions in mixed case; try canonical variants
                for suffix in (".JPG", ".jpg", ".png", ".PNG"):
                    alt = image_path.with_suffix(suffix)
                    if alt.is_file():
                        image_path = alt
                        break
                else:
                    continue  # skip missing; log already emitted by downstream reporter

            label = "good" if row_label == "normal" else "defect"
            description = (
                "PASS\nItem appears normal; no anomalies visible."
                if label == "good"
                else f"FAIL\nAnomaly detected in {obj}."
            )
            samples.append(
                LabeledSample(
                    image_path=image_path,
                    label=label,
                    description=description,
                )
            )

    if not samples:
        raise DatasetError(
            f"No VisA samples matched the requested filters under {path}",
            suggestion="Check the objects/split filters and that the dataset is fully extracted.",
        )

    samples.sort(key=lambda s: s.image_path)
    return samples


def list_visa_objects(path: Path) -> list[str]:
    """Return the sorted list of VisA object categories present on disk."""
    root = path
    if not root.is_dir():
        raise DatasetError(f"VisA root not found: {root}")
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "Data" / "Images" / "Normal").is_dir()
    )
