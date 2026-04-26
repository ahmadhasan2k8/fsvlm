"""Tests for fsvlm.utils.image."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fsvlm.utils.image import load_image, validate_image


def test_load_image(sample_good_image: Path):
    img = load_image(sample_good_image)
    assert img.mode == "RGB"
    assert img.size == (100, 100)


def test_load_image_resize(tmp_path: Path):
    img_path = tmp_path / "big.png"
    Image.new("RGB", (1000, 800)).save(img_path)

    img = load_image(img_path, max_size=560)
    w, h = img.size
    assert max(w, h) <= 560


def test_load_image_small_stays(tmp_path: Path):
    img_path = tmp_path / "small.png"
    Image.new("RGB", (200, 150)).save(img_path)

    img = load_image(img_path, max_size=560)
    assert img.size == (200, 150)


def test_load_image_not_found():
    with pytest.raises(FileNotFoundError):
        load_image(Path("/nonexistent/image.png"))


def test_load_image_invalid(tmp_path: Path):
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    with pytest.raises(ValueError, match="Cannot open"):
        load_image(bad)


def test_validate_image_valid(sample_good_image: Path):
    assert validate_image(sample_good_image) is True


def test_validate_image_invalid(tmp_path: Path):
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    assert validate_image(bad) is False


def test_validate_image_missing(tmp_path: Path):
    assert validate_image(tmp_path / "missing.png") is False
