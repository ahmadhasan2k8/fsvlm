"""Tests for fsvlm.readers.json_reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from fsvlm.exceptions import DatasetError
from fsvlm.readers.json_reader import JSONLabelReader


@pytest.fixture
def json_dataset(tmp_path: Path) -> Path:
    """Create a JSON dataset with images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for i in range(3):
        Image.new("RGB", (32, 32)).save(img_dir / f"good_{i}.png")
    for i in range(2):
        Image.new("RGB", (32, 32)).save(img_dir / f"defect_{i}.png")

    data = [
        {"image_path": "images/good_0.png", "label": "good"},
        {"image_path": "images/good_1.png", "label": "good"},
        {"image_path": "images/good_2.png", "label": "good"},
        {"image_path": "images/defect_0.png", "label": "defect", "description": "Crack"},
        {"image_path": "images/defect_1.png", "label": "defect"},
    ]

    json_path = tmp_path / "labels.json"
    json_path.write_text(json.dumps(data))
    return json_path


def test_supports(json_dataset: Path):
    reader = JSONLabelReader()
    assert reader.supports(json_dataset) is True
    assert reader.supports(Path("/tmp")) is False


def test_read_json(json_dataset: Path):
    reader = JSONLabelReader()
    samples = reader.read(json_dataset)
    assert len(samples) == 5
    assert sum(1 for s in samples if s.label == "good") == 3
    assert sum(1 for s in samples if s.label == "defect") == 2


def test_read_json_descriptions(json_dataset: Path):
    reader = JSONLabelReader()
    samples = reader.read(json_dataset)
    crack = [s for s in samples if s.description == "Crack"]
    assert len(crack) == 1


def test_read_json_missing_file():
    reader = JSONLabelReader()
    with pytest.raises(DatasetError, match="not found"):
        reader.read(Path("/nonexistent.json"))


def test_read_json_invalid_format(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "an array"}')
    reader = JSONLabelReader()
    with pytest.raises(DatasetError, match="array"):
        reader.read(bad)


def test_read_json_corrupt(tmp_path: Path):
    bad = tmp_path / "corrupt.json"
    bad.write_text("{not valid json")
    reader = JSONLabelReader()
    with pytest.raises(DatasetError, match="Error reading"):
        reader.read(bad)
