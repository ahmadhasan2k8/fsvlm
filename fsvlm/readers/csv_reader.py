"""CSV-based label reader.

Expected CSV format:
    image_path,label,description
    images/good_001.png,good,
    images/defect_001.png,defect,Crack on surface

The 'description' column is optional. Paths can be absolute or relative
to the CSV file's parent directory.
"""

from __future__ import annotations

import csv
from pathlib import Path

from fsvlm.exceptions import DatasetError
from fsvlm.interfaces import LabelReader
from fsvlm.registry import label_readers
from fsvlm.types import LabeledSample


@label_readers.register("csv")
class CSVLabelReader(LabelReader):
    """Reads labeled samples from a CSV file."""

    def supports(self, path: Path) -> bool:
        """Check if path is a CSV file."""
        return path.is_file() and path.suffix.lower() == ".csv"

    def read(self, path: Path) -> list[LabeledSample]:
        """Read labeled samples from a CSV file.

        Args:
            path: Path to the CSV file.

        Returns:
            List of LabeledSample, sorted by image path.

        Raises:
            DatasetError: If the CSV is missing required columns or has no valid rows.
        """
        if not path.is_file():
            raise DatasetError(
                f"CSV file not found: {path}",
                suggestion="Provide a valid CSV file with columns: image_path, label",
            )

        base_dir = path.parent
        samples: list[LabeledSample] = []

        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                if reader.fieldnames is None:
                    raise DatasetError(
                        f"Empty CSV file: {path}",
                        suggestion="CSV must have a header row with: image_path, label",
                    )

                # Normalize field names (strip whitespace, lowercase)
                fields = {name.strip().lower() for name in reader.fieldnames}

                if "image_path" not in fields and "image" not in fields:
                    raise DatasetError(
                        f"CSV missing 'image_path' column. Found: {reader.fieldnames}",
                        suggestion="CSV must have columns: image_path, label",
                    )
                if "label" not in fields:
                    raise DatasetError(
                        f"CSV missing 'label' column. Found: {reader.fieldnames}",
                        suggestion="CSV must have columns: image_path, label",
                    )

                for row_num, row in enumerate(reader, start=2):
                    # Normalize keys
                    row = {k.strip().lower(): v.strip() if v else "" for k, v in row.items()}

                    img_path_str = row.get("image_path") or row.get("image", "")
                    label = row.get("label", "")
                    description = row.get("description", "")

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
                            description=description,
                        )
                    )

        except csv.Error as e:
            raise DatasetError(
                f"Error parsing CSV {path}: {e}",
                suggestion="Check CSV format and encoding (UTF-8 expected).",
            ) from e

        if not samples:
            raise DatasetError(
                f"No valid samples found in {path}",
                suggestion="CSV rows need: image_path (existing file), label (good/defect).",
            )

        samples.sort(key=lambda s: s.image_path)
        return samples
