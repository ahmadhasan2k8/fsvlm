"""Data Agent — reads images, validates, splits into train/val.

Uses registry-based reader detection: auto-detects folder, CSV, or JSON input.
"""

from __future__ import annotations

from pathlib import Path

from fsvlm.config import FSVLMConfig
from fsvlm.exceptions import DatasetError
from fsvlm.types import DatasetReport, LabeledSample, PreparedDataset


class DataAgent:
    """Prepares labeled image data for training.

    Auto-detects the input format (folder, CSV, JSON) via the reader registry,
    validates images, performs a stratified train/val split, and oversamples
    the minority class.

    Args:
        config: FSVLM configuration.
    """

    def __init__(self, config: FSVLMConfig) -> None:
        self._config = config

    def prepare(
        self,
        data_path: Path,
        test_split: float | None = None,
        seed: int | None = None,
    ) -> PreparedDataset:
        """Read, validate, split, and balance the dataset.

        Args:
            data_path: Directory (good/defect), CSV file, or JSON file.
            test_split: Fraction held out for validation (default from config).
            seed: Random seed for reproducibility.

        Returns:
            PreparedDataset ready for training.

        Raises:
            DatasetError: If the path is invalid or has no images.
        """

        split = test_split if test_split is not None else (1.0 - self._config.default_train_split)
        rng_seed = seed if seed is not None else self._config.default_seed

        # Auto-detect reader from registry
        samples = self._read_samples(data_path)
        good_count = sum(1 for s in samples if s.label == "good")
        defect_count = sum(1 for s in samples if s.label == "defect")

        if good_count == 0 and defect_count == 0:
            raise DatasetError(
                "No images found.",
                suggestion="Add images to good/ and defect/ subdirectories.",
            )

        # Validate images
        corrupt: list[str] = []
        valid_samples: list[LabeledSample] = []
        for s in samples:
            from fsvlm.utils.image import validate_image

            if validate_image(s.image_path):
                valid_samples.append(s)
            else:
                corrupt.append(str(s.image_path))

        if corrupt:
            from loguru import logger

            logger.warning(
                f"{len(corrupt)} images could not be loaded and were skipped: "
                f"{corrupt[:5]}{'...' if len(corrupt) > 5 else ''}"
            )

        if not valid_samples:
            raise DatasetError(
                "All images are corrupt or unreadable.",
                suggestion="Check image files and formats.",
            )

        # Collect image sizes
        sizes: list[tuple[int, int]] = []
        for s in valid_samples:
            try:
                from PIL import Image

                with Image.open(s.image_path) as img:
                    sizes.append(img.size)
            except Exception:
                sizes.append((0, 0))

        report = DatasetReport(
            total_images=len(valid_samples),
            good_count=sum(1 for s in valid_samples if s.label == "good"),
            defect_count=sum(1 for s in valid_samples if s.label == "defect"),
            corrupt_skipped=corrupt,
            image_sizes=sizes,
        )

        # Stratified split
        train_samples, val_samples = self._stratified_split(valid_samples, split, rng_seed)

        # Oversample minority class in training set
        train_samples = self._oversample_minority(train_samples, rng_seed)

        return PreparedDataset(
            train_samples=train_samples,
            val_samples=val_samples,
            report=report,
            seed=rng_seed,
        )

    def _stratified_split(
        self,
        samples: list[LabeledSample],
        test_fraction: float,
        seed: int,
    ) -> tuple[list[LabeledSample], list[LabeledSample]]:
        """Deterministic stratified train/val split.

        At very small N (per-class size ≤ 1) stratified val is impossible without
        emptying the training set. In that regime we return val == train so that
        downstream metrics code can run (training-set metrics are meaningless at
        tiny N; the real evaluation happens on the held-out TEST set separately).
        This unblocks Pass 3's tiger-analogy low-N cells (N=2, 3, 4).
        """
        import numpy as np

        good = [s for s in samples if s.label == "good"]
        defect = [s for s in samples if s.label == "defect"]

        rng = np.random.RandomState(seed)

        def _split_group(group: list[LabeledSample]) -> tuple[list[LabeledSample], list[LabeledSample]]:
            if len(group) <= 1:
                # Too small to split — use group as both train and val.
                return list(group), list(group)
            indices = np.arange(len(group))
            rng.shuffle(indices)
            n_val = max(1, int(len(group) * test_fraction))
            val_idx = set(indices[:n_val].tolist())
            train = [group[i] for i in range(len(group)) if i not in val_idx]
            val = [group[i] for i in val_idx]
            return train, val

        train_good, val_good = _split_group(good)
        train_defect, val_defect = _split_group(defect)

        return train_good + train_defect, val_good + val_defect

    def _oversample_minority(
        self,
        samples: list[LabeledSample],
        seed: int,
    ) -> list[LabeledSample]:
        """Oversample the minority class to match the majority class count."""
        import numpy as np

        good = [s for s in samples if s.label == "good"]
        defect = [s for s in samples if s.label == "defect"]

        if len(good) == 0 or len(defect) == 0:
            return samples

        if len(good) > len(defect):
            majority, minority = good, defect
        else:
            majority, minority = defect, good

        if len(majority) <= len(minority):
            return samples

        rng = np.random.RandomState(seed + 1)
        n_needed = len(majority) - len(minority)
        full_copies = n_needed // len(minority)
        remainder = n_needed % len(minority)

        oversampled = minority * (1 + full_copies)
        if remainder > 0:
            extra_idx = rng.choice(len(minority), remainder, replace=False)
            oversampled.extend(minority[i] for i in extra_idx)

        combined = majority + oversampled
        rng.shuffle(combined)
        return combined

    def _read_samples(self, data_path: Path) -> list[LabeledSample]:
        """Auto-detect reader and read samples from the given path."""
        # Import readers to populate registry
        from fsvlm.readers.csv_reader import CSVLabelReader  # noqa: F401
        from fsvlm.readers.folder_reader import FolderLabelReader  # noqa: F401
        from fsvlm.readers.json_reader import JSONLabelReader  # noqa: F401
        from fsvlm.registry import label_readers

        for reader_cls in label_readers.all_classes():
            reader = reader_cls()
            if reader.supports(data_path):
                return reader.read(data_path)

        # No reader matched — give a helpful error
        if data_path.is_file():
            suffix = data_path.suffix
            raise DatasetError(
                f"Unsupported file format: {suffix}",
                suggestion="Supported formats: folder (good/defect dirs), .csv, .json",
            )
        raise DatasetError(
            f"Cannot read data from: {data_path}",
            suggestion="Provide a folder with good/defect subdirs, or a CSV/JSON file.",
        )
