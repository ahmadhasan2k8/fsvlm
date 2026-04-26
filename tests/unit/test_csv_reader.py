"""Tests for fsvlm.readers.csv_reader."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fsvlm.exceptions import DatasetError
from fsvlm.readers.csv_reader import CSVLabelReader


@pytest.fixture
def csv_dataset(tmp_path: Path) -> Path:
    """Create a CSV dataset with images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for i in range(3):
        Image.new("RGB", (32, 32), color=(0, 255, 0)).save(img_dir / f"good_{i}.png")
    for i in range(2):
        Image.new("RGB", (32, 32), color=(255, 0, 0)).save(img_dir / f"defect_{i}.png")

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "image_path,label,description\n"
        "images/good_0.png,good,\n"
        "images/good_1.png,good,Clean surface\n"
        "images/good_2.png,good,\n"
        "images/defect_0.png,defect,Crack visible\n"
        "images/defect_1.png,defect,Scratch\n"
    )
    return csv_path


def test_supports():
    reader = CSVLabelReader()
    assert reader.supports(Path("data.csv")) is False  # doesn't exist
    assert reader.supports(Path("/tmp")) is False  # not a file


def test_supports_real_csv(csv_dataset: Path):
    reader = CSVLabelReader()
    assert reader.supports(csv_dataset) is True


def test_read_csv(csv_dataset: Path):
    reader = CSVLabelReader()
    samples = reader.read(csv_dataset)
    assert len(samples) == 5
    good = [s for s in samples if s.label == "good"]
    defect = [s for s in samples if s.label == "defect"]
    assert len(good) == 3
    assert len(defect) == 2


def test_read_csv_descriptions(csv_dataset: Path):
    reader = CSVLabelReader()
    samples = reader.read(csv_dataset)
    descs = [s.description for s in samples if s.description]
    assert len(descs) >= 2


def test_read_csv_missing_file():
    reader = CSVLabelReader()
    with pytest.raises(DatasetError, match="not found"):
        reader.read(Path("/nonexistent.csv"))


def test_read_csv_missing_columns(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("name,value\nfoo,bar\n")
    reader = CSVLabelReader()
    with pytest.raises(DatasetError, match="missing"):
        reader.read(csv_path)


def test_read_csv_empty(tmp_path: Path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("image_path,label\n")
    reader = CSVLabelReader()
    with pytest.raises(DatasetError, match="No valid"):
        reader.read(csv_path)


def test_read_csv_skips_missing_images(tmp_path: Path):
    csv_path = tmp_path / "partial.csv"
    csv_path.write_text("image_path,label\nmissing.png,good\n")
    reader = CSVLabelReader()
    with pytest.raises(DatasetError, match="No valid"):
        reader.read(csv_path)
