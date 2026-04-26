"""DeepPCB benchmark reader.

DeepPCB format: `group<N>/<subdir>/<id>_test.jpg` + `<id>_temp.jpg` pairs plus
`group<N>/<subdir>_not/<id>.txt` bounding-box annotations. Split files
`trainval.txt` and `test.txt` list pairs, space-separated:

    group20085/20085/20085000.jpg group20085/20085_not/20085000.txt

For binary defect/good inspection framing used by FSVLM:
- Each test image (with bbox annotations) is labeled ``defect``
- Each template image (defect-free by definition) is labeled ``good``

Paired via the shared image id so downstream code can run a fair comparison.
"""

from __future__ import annotations

from pathlib import Path

from fsvlm.exceptions import DatasetError
from fsvlm.interfaces import LabelReader
from fsvlm.registry import label_readers
from fsvlm.types import LabeledSample

DEFECT_CLASSES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}


@label_readers.register("deeppcb")
class DeepPCBReader(LabelReader):
    """Reader for the DeepPCB industrial PCB-defect dataset."""

    def supports(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        pcb_data = path / "PCBData"
        return (
            pcb_data.is_dir() and (pcb_data / "trainval.txt").is_file() and (pcb_data / "test.txt").is_file()
        )

    def read(self, path: Path) -> list[LabeledSample]:
        return read_deeppcb(path)


def read_deeppcb(path: Path, split: str = "all") -> list[LabeledSample]:
    """Read DeepPCB as a list of LabeledSample pairs.

    Args:
        path: DeepPCB repo root (must contain PCBData/).
        split: ``"trainval"``, ``"test"``, or ``"all"`` (default).

    Returns:
        LabeledSample list: template images → ``good``, test images → ``defect``.
    """
    pcb_data = path / "PCBData"
    if not pcb_data.is_dir():
        raise DatasetError(
            f"Missing PCBData/ under {path}",
            suggestion="Clone https://github.com/tangsanli5201/DeepPCB to this path.",
        )

    if split == "all":
        splits = ["trainval.txt", "test.txt"]
    elif split == "trainval":
        splits = ["trainval.txt"]
    elif split == "test":
        splits = ["test.txt"]
    else:
        raise DatasetError(
            f"Unknown DeepPCB split: {split}",
            suggestion="Use 'trainval', 'test', or 'all'.",
        )

    samples: list[LabeledSample] = []
    for split_name in splits:
        split_file = pcb_data / split_name
        if not split_file.is_file():
            raise DatasetError(
                f"Missing DeepPCB split file: {split_file}",
                suggestion="Re-run experiments/datasets/download_deeppcb.sh.",
            )

        for raw_line in split_file.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue  # skip malformed lines
            img_rel, ann_rel = parts

            # The split file lists stem-like paths ("group20085/20085/20085000.jpg"),
            # but on disk each pair is "<id>_test.jpg" + "<id>_temp.jpg". Derive both.
            stem_ref = pcb_data / img_rel
            if "_test" in stem_ref.stem:
                test_img = stem_ref
                template_img = stem_ref.with_name(stem_ref.stem.replace("_test", "_temp") + stem_ref.suffix)
            else:
                test_img = stem_ref.with_name(f"{stem_ref.stem}_test{stem_ref.suffix}")
                template_img = stem_ref.with_name(f"{stem_ref.stem}_temp{stem_ref.suffix}")

            if not test_img.is_file():
                continue

            ann_file = pcb_data / ann_rel
            defect_descriptions = _parse_annotations(ann_file) if ann_file.is_file() else []
            description = _describe_defects(defect_descriptions) or (
                "FAIL\nPCB defect present; see bounding-box annotations."
            )

            samples.append(
                LabeledSample(
                    image_path=test_img,
                    label="defect",
                    description=description,
                )
            )
            if template_img.is_file():
                samples.append(
                    LabeledSample(
                        image_path=template_img,
                        label="good",
                        description="PASS\nDefect-free PCB template; no anomalies visible.",
                    )
                )

    if not samples:
        raise DatasetError(
            f"No DeepPCB samples parsed from {path}",
            suggestion="Verify the dataset downloaded cleanly and split files are non-empty.",
        )

    samples.sort(key=lambda s: s.image_path)
    return samples


def _parse_annotations(ann_file: Path) -> list[str]:
    """Return a list of defect class names from a DeepPCB annotation file."""
    names: list[str] = []
    try:
        for raw in ann_file.read_text().splitlines():
            parts = raw.strip().split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[4])
            except ValueError:
                continue
            name = DEFECT_CLASSES.get(class_id)
            if name:
                names.append(name)
    except OSError:
        return []
    return names


def _describe_defects(defect_names: list[str]) -> str:
    """Build a natural-language description from defect class names."""
    if not defect_names:
        return ""
    counts: dict[str, int] = {}
    for name in defect_names:
        counts[name] = counts.get(name, 0) + 1
    parts = [f"{count} {name}" for name, count in sorted(counts.items())]
    joined = ", ".join(parts)
    return f"FAIL\nPCB defects detected: {joined}."
