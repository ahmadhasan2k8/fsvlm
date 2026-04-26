"""Tests for the VisA benchmark reader."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from fsvlm.readers.visa_reader import (
    VisAReader,
    list_visa_objects,
    read_visa,
)


def _make_visa_fixture(tmp_path: Path) -> Path:
    """Build a minimal VisA-shaped directory with two objects and four images."""
    root = tmp_path / "visa"
    (root / "split_csv").mkdir(parents=True)

    for obj in ("candle", "pcb1"):
        for bucket in ("Normal", "Anomaly"):
            (root / obj / "Data" / "Images" / bucket).mkdir(parents=True)

    # Create image files
    images = {
        ("candle", "Normal", "0001.JPG"): "train",
        ("candle", "Anomaly", "0002.JPG"): "test",
        ("pcb1", "Normal", "0003.JPG"): "test",
        ("pcb1", "Anomaly", "0004.JPG"): "test",
    }
    rows = []
    for (obj, bucket, name), split in images.items():
        image_path = root / obj / "Data" / "Images" / bucket / name
        image_path.write_bytes(b"fake-jpeg")
        label = "normal" if bucket == "Normal" else "anomaly"
        rows.append(
            {
                "object": obj,
                "split": split,
                "label": label,
                "image": f"{obj}/Data/Images/{bucket}/{name}",
                "mask": "",
            }
        )

    csv_path = root / "split_csv" / "1cls.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["object", "split", "label", "image", "mask"])
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_visa_reader_supports_detects_root(tmp_path: Path) -> None:
    root = _make_visa_fixture(tmp_path)
    assert VisAReader().supports(root) is True


def test_visa_reader_supports_rejects_random_dir(tmp_path: Path) -> None:
    assert VisAReader().supports(tmp_path) is False


def test_read_visa_returns_all_rows(tmp_path: Path) -> None:
    root = _make_visa_fixture(tmp_path)
    samples = read_visa(root)
    labels = {s.label for s in samples}
    assert len(samples) == 4
    assert labels == {"good", "defect"}


def test_read_visa_filters_by_object(tmp_path: Path) -> None:
    root = _make_visa_fixture(tmp_path)
    samples = read_visa(root, objects=["candle"])
    assert all("candle" in str(s.image_path) for s in samples)
    assert len(samples) == 2


def test_read_visa_filters_by_split(tmp_path: Path) -> None:
    root = _make_visa_fixture(tmp_path)
    test_samples = read_visa(root, split="test")
    assert len(test_samples) == 3  # 1 candle-anomaly + 2 pcb1 rows are "test"


def test_read_visa_missing_csv_raises(tmp_path: Path) -> None:
    (tmp_path / "visa").mkdir()
    from fsvlm.exceptions import DatasetError

    with pytest.raises(DatasetError):
        read_visa(tmp_path / "visa")


def test_list_visa_objects(tmp_path: Path) -> None:
    root = _make_visa_fixture(tmp_path)
    assert list_visa_objects(root) == ["candle", "pcb1"]


def test_visa_descriptions_are_plain_english(tmp_path: Path) -> None:
    root = _make_visa_fixture(tmp_path)
    samples = read_visa(root)
    goods = [s for s in samples if s.label == "good"]
    defects = [s for s in samples if s.label == "defect"]
    assert all("no anomalies" in s.description.lower() for s in goods)
    assert all("anomaly detected" in s.description.lower() for s in defects)
