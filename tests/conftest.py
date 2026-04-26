"""Shared pytest fixtures for FSVLM tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fsvlm.config import FSVLMConfig
from fsvlm.types import LabeledSample, ValidationMetrics


@pytest.fixture
def tmp_image_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with good/ and defect/ subdirs containing test images."""
    good_dir = tmp_path / "good"
    defect_dir = tmp_path / "defect"
    good_dir.mkdir()
    defect_dir.mkdir()

    # Create small test images
    for i in range(10):
        img = Image.new("RGB", (64, 64), color=(0, 255, 0))
        img.save(good_dir / f"good_{i:03d}.png")

    for i in range(5):
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img.save(defect_dir / f"defect_{i:03d}.png")

    return tmp_path


@pytest.fixture
def sample_good_image(tmp_path: Path) -> Path:
    """Create a single test image."""
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (100, 100), color=(0, 255, 0))
    img.save(img_path)
    return img_path


@pytest.fixture
def labeled_samples(tmp_image_dir: Path) -> list[LabeledSample]:
    """Create LabeledSample objects from the temp image dir."""
    from fsvlm.readers.folder_reader import read_folder

    return read_folder(tmp_image_dir)


@pytest.fixture
def config(tmp_path: Path) -> FSVLMConfig:
    """Create a test config with temp directories."""
    return FSVLMConfig(base_dir=tmp_path / ".fsvlm")


@pytest.fixture
def sample_metrics() -> ValidationMetrics:
    """Sample validation metrics for testing."""
    return ValidationMetrics(
        accuracy=0.95,
        precision=0.92,
        recall=0.88,
        f1=0.90,
        auroc=0.97,
        optimal_threshold=0.45,
    )
