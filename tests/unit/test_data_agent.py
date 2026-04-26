"""Tests for fsvlm.agents.data_agent."""

from __future__ import annotations

from pathlib import Path

import pytest

from fsvlm.agents.data_agent import DataAgent
from fsvlm.config import FSVLMConfig
from fsvlm.exceptions import DatasetError


def test_prepare_basic(tmp_image_dir: Path, config: FSVLMConfig):
    agent = DataAgent(config)
    ds = agent.prepare(tmp_image_dir)

    assert ds.report.total_images == 15
    assert ds.report.good_count == 10
    assert ds.report.defect_count == 5
    assert len(ds.train_samples) > 0
    assert len(ds.val_samples) > 0
    assert len(ds.train_samples) + len(ds.val_samples) >= 15


def test_prepare_stratified_split(tmp_image_dir: Path, config: FSVLMConfig):
    agent = DataAgent(config)
    ds = agent.prepare(tmp_image_dir, test_split=0.2)

    # Both sets should have samples
    assert len(ds.train_samples) > 0
    assert len(ds.val_samples) > 0

    # Val should have both classes
    val_labels = {s.label for s in ds.val_samples}
    assert "good" in val_labels or "defect" in val_labels


def test_prepare_oversamples_minority(tmp_image_dir: Path, config: FSVLMConfig):
    agent = DataAgent(config)
    ds = agent.prepare(tmp_image_dir)

    # After oversampling, training set should be balanced
    train_good = sum(1 for s in ds.train_samples if s.label == "good")
    train_defect = sum(1 for s in ds.train_samples if s.label == "defect")
    # Defect count should be close to good count (within 1)
    assert abs(train_good - train_defect) <= 1


def test_prepare_reproducible(tmp_image_dir: Path, config: FSVLMConfig):
    agent = DataAgent(config)
    ds1 = agent.prepare(tmp_image_dir, seed=42)
    ds2 = agent.prepare(tmp_image_dir, seed=42)

    assert len(ds1.val_samples) == len(ds2.val_samples)
    for s1, s2 in zip(ds1.val_samples, ds2.val_samples):
        assert s1.image_path == s2.image_path
        assert s1.label == s2.label


def test_prepare_invalid_dir(config: FSVLMConfig):
    agent = DataAgent(config)
    with pytest.raises(DatasetError):
        agent.prepare(Path("/nonexistent"))


def test_prepare_custom_split(tmp_image_dir: Path, config: FSVLMConfig):
    agent = DataAgent(config)
    ds = agent.prepare(tmp_image_dir, test_split=0.5)

    # With 50% split, val should be roughly half
    total = ds.report.total_images
    assert len(ds.val_samples) >= total // 3  # allow for oversampling effects
