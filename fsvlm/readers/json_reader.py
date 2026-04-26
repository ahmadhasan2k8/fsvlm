"""JSON-based label reader.

Expected JSON format:
    [
        {"image_path": "images/good_001.png", "label": "good"},
        {"image_path": "images/defect_001.png", "label": "defect", "description": "Crack"}
    ]

Paths can be absolute or relative to the JSON file's parent directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from fsvlm.exceptions import DatasetError
from fsvlm.interfaces import LabelReader
from fsvlm.registry import label_readers
from fsvlm.types import LabeledSample


@label_readers.register("json")
class JSONLabelReader(LabelReader):
    """Reads labeled samples from a JSON file."""

    def supports(self, path: Path) -> bool:
        """Check if path is a JSON file."""
        return path.is_file() and path.suffix.lower() == ".json"

    def read(self, path: Path) -> list[LabeledSample]:
        """Read labeled samples from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            List of LabeledSample, sorted by image path.

        Raises:
            DatasetError: If the JSON is invalid or has no valid entries.
        """
        if not path.is_file():
            raise DatasetError(
                f"JSON file not found: {path}",
                suggestion="Provide a valid JSON file with array of {image_path, label} objects.",
            )

        base_dir = path.parent

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise DatasetError(
                f"Error reading JSON {path}: {e}",
                suggestion="Check JSON format and encoding.",
            ) from e

        if not isinstance(data, list):
            raise DatasetError(
                f"JSON must be an array of objects, got {type(data).__name__}",
                suggestion='Expected format: [{"image_path": "...", "label": "good"}, ...]',
            )

        samples: list[LabeledSample] = []

        for entry in data:
            if not isinstance(entry, dict):
                continue

            img_path_str = entry.get("image_path") or entry.get("image", "")
            label = entry.get("label", "")

            if not img_path_str or not label:
                continue

            img_path = Path(img_path_str)
            if not img_path.is_absolute():
                img_path = base_dir / img_path

            if not img_path.exists():
                continue

            label = label.lower()
            if label not in ("good", "defect"):
                continue

            samples.append(
                LabeledSample(
                    image_path=img_path,
                    label=label,
                    description=entry.get("description", ""),
                )
            )

        if not samples:
            raise DatasetError(
                f"No valid samples found in {path}",
                suggestion="JSON entries need: image_path (existing file), label (good/defect).",
            )

        samples.sort(key=lambda s: s.image_path)
        return samples
