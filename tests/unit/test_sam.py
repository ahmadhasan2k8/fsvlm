"""Tests for fsvlm.models.sam — SAM wrapper and utilities."""

from __future__ import annotations

import numpy as np

from fsvlm.models.sam import (
    SAMSegmenter,
    _mask_to_rle,
    mask_to_location,
    rle_to_mask,
)


def test_mask_to_rle_roundtrip():
    """RLE encode → decode should produce identical mask."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True  # rectangle

    rle = _mask_to_rle(mask)
    recovered = rle_to_mask(rle)

    np.testing.assert_array_equal(mask, recovered)


def test_mask_to_rle_empty():
    """Empty mask should roundtrip correctly."""
    mask = np.zeros((5, 5), dtype=bool)
    rle = _mask_to_rle(mask)
    recovered = rle_to_mask(rle)
    np.testing.assert_array_equal(mask, recovered)


def test_mask_to_rle_full():
    """Fully-True mask should roundtrip correctly."""
    mask = np.ones((4, 4), dtype=bool)
    rle = _mask_to_rle(mask)
    recovered = rle_to_mask(rle)
    np.testing.assert_array_equal(mask, recovered)


def test_mask_to_rle_format():
    """RLE string should contain height,width header."""
    mask = np.zeros((8, 12), dtype=bool)
    mask[1, 1] = True
    rle = _mask_to_rle(mask)
    assert rle.startswith("8,12:")


def test_mask_to_location_center():
    """Mask in center should be described as 'center region'."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    assert mask_to_location(mask) == "center region"


def test_mask_to_location_upper_left():
    """Mask in upper-left should be described accordingly."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[5:15, 5:15] = True
    assert mask_to_location(mask) == "upper-left region"


def test_mask_to_location_lower_right():
    """Mask in lower-right should be described accordingly."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[80:95, 80:95] = True
    assert mask_to_location(mask) == "lower-right region"


def test_mask_to_location_empty():
    """Empty mask should return 'unknown'."""
    mask = np.zeros((50, 50), dtype=bool)
    assert mask_to_location(mask) == "unknown"


def test_sam_segmenter_defaults():
    """SAMSegmenter should have correct defaults."""
    seg = SAMSegmenter()
    assert seg.model_name == "facebook/sam2.1-hiera-tiny"
    assert seg.is_loaded is False


def test_sam_segmenter_custom_model():
    """SAMSegmenter should accept custom model name."""
    seg = SAMSegmenter(model_name="facebook/sam-vit-base", device="cpu")
    assert seg.model_name == "facebook/sam-vit-base"
    assert seg._device == "cpu"


def test_sam_segmenter_unload_without_load():
    """Unloading without loading should not raise."""
    seg = SAMSegmenter()
    seg.unload()  # should be a no-op
    assert seg.is_loaded is False


def test_rle_roundtrip_scattered():
    """Scattered True pixels should roundtrip correctly."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[0, 0] = True
    mask[5, 10] = True
    mask[19, 19] = True

    rle = _mask_to_rle(mask)
    recovered = rle_to_mask(rle)
    np.testing.assert_array_equal(mask, recovered)
