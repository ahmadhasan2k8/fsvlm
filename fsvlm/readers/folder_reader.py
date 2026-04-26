"""Folder-based label reader — reads good/ and defect/ subdirectories."""

from __future__ import annotations

from pathlib import Path

from fsvlm.exceptions import DatasetError
from fsvlm.interfaces import LabelReader
from fsvlm.registry import label_readers
from fsvlm.types import LabeledSample

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@label_readers.register("folder")
class FolderLabelReader(LabelReader):
    """Reads labeled images from good/ and defect/ subdirectories."""

    def supports(self, path: Path) -> bool:
        """Check if path is a directory with good/ or defect/ subdirs."""
        if not path.is_dir():
            return False
        return (path / "good").is_dir() or (path / "defect").is_dir()

    def read(self, path: Path) -> list[LabeledSample]:
        """Read labeled images from good/ and defect/ subdirectories."""
        return read_folder(path)


def read_folder(path: Path) -> list[LabeledSample]:
    """Read labeled images from a folder with good/ and defect/ subdirectories.

    Args:
        path: Root directory containing good/ and/or defect/ subdirectories.

    Returns:
        List of LabeledSample, sorted by path for reproducibility.

    Raises:
        DatasetError: If the directory doesn't exist or has no valid subdirectories.
    """
    if not path.is_dir():
        raise DatasetError(
            f"Image directory not found: {path}",
            suggestion="Provide a directory with good/ and defect/ subdirectories.",
        )

    good_dir = path / "good"
    defect_dir = path / "defect"

    has_good = good_dir.is_dir()
    has_defect = defect_dir.is_dir()

    if not has_good and not has_defect:
        raise DatasetError(
            f"No 'good/' or 'defect/' subdirectories found in {path}",
            suggestion="Organize images into good/ and defect/ subdirectories.",
        )

    samples: list[LabeledSample] = []

    if has_good:
        samples.extend(_read_label_dir(good_dir, "good"))

    if has_defect:
        samples.extend(_read_label_dir(defect_dir, "defect"))

    if not samples:
        raise DatasetError(
            f"No valid images found in {path}",
            suggestion=f"Add images ({', '.join(IMAGE_EXTENSIONS)}) to good/ and/or defect/.",
        )

    # Sort for reproducibility
    samples.sort(key=lambda s: s.image_path)
    return samples


def _read_label_dir(directory: Path, label: str) -> list[LabeledSample]:
    """Read all valid images from a single label directory."""
    samples = []
    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            samples.append(LabeledSample(image_path=file_path, label=label))
    return samples
