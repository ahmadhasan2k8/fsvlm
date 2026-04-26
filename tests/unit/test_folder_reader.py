"""Tests for fsvlm.readers.folder_reader."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fsvlm.exceptions import DatasetError
from fsvlm.readers.folder_reader import read_folder


def test_read_folder(tmp_image_dir: Path):
    samples = read_folder(tmp_image_dir)
    assert len(samples) == 15  # 10 good + 5 defect
    good = [s for s in samples if s.label == "good"]
    defect = [s for s in samples if s.label == "defect"]
    assert len(good) == 10
    assert len(defect) == 5


def test_read_folder_sorted(tmp_image_dir: Path):
    samples = read_folder(tmp_image_dir)
    paths = [s.image_path for s in samples]
    assert paths == sorted(paths)


def test_read_folder_missing_dir():
    with pytest.raises(DatasetError, match="not found"):
        read_folder(Path("/nonexistent/path"))


def test_read_folder_no_subdirs(tmp_path: Path):
    with pytest.raises(DatasetError, match="No 'good/' or 'defect/'"):
        read_folder(tmp_path)


def test_read_folder_empty_subdirs(tmp_path: Path):
    (tmp_path / "good").mkdir()
    (tmp_path / "defect").mkdir()
    with pytest.raises(DatasetError, match="No valid images"):
        read_folder(tmp_path)


def test_read_folder_good_only(tmp_path: Path):
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    img = Image.new("RGB", (32, 32))
    img.save(good_dir / "test.png")

    samples = read_folder(tmp_path)
    assert len(samples) == 1
    assert samples[0].label == "good"


def test_read_folder_ignores_non_images(tmp_path: Path):
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    (good_dir / "notes.txt").write_text("not an image")
    img = Image.new("RGB", (32, 32))
    img.save(good_dir / "real.png")

    samples = read_folder(tmp_path)
    assert len(samples) == 1


def test_read_folder_multiple_formats(tmp_path: Path):
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    img = Image.new("RGB", (32, 32))
    img.save(good_dir / "a.png")
    img.save(good_dir / "b.jpg")
    img.save(good_dir / "c.bmp")

    samples = read_folder(tmp_path)
    assert len(samples) == 3
