"""Tests for fsvlm.models.adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from fsvlm.exceptions import InvalidAdapterError
from fsvlm.models.adapter import load_adapter_metadata, save_adapter_metadata
from fsvlm.types import AdapterMetadata, ValidationMetrics


def test_save_load_roundtrip(tmp_path: Path):
    meta = AdapterMetadata(
        adapter_name="test-adapter",
        base_model="gemma-4-E4B-it",
        lora_rank=16,
        lora_alpha=16,
        training_images=500,
        training_epochs=3,
    )
    save_adapter_metadata(tmp_path, meta)
    loaded = load_adapter_metadata(tmp_path)

    assert loaded.adapter_name == "test-adapter"
    assert loaded.base_model == "gemma-4-E4B-it"
    assert loaded.lora_rank == 16
    assert loaded.schema_version == 1
    assert loaded.created_at != ""  # auto-set


def test_save_load_with_metrics(tmp_path: Path):
    metrics = ValidationMetrics(0.95, 0.9, 0.88, 0.89, 0.97, 0.42)
    meta = AdapterMetadata(
        adapter_name="with-metrics",
        base_model="gemma-4-E4B-it",
        validation_metrics=metrics,
    )
    save_adapter_metadata(tmp_path, meta)
    loaded = load_adapter_metadata(tmp_path)

    assert loaded.validation_metrics is not None
    assert loaded.validation_metrics.auroc == 0.97
    assert loaded.validation_metrics.optimal_threshold == 0.42


def test_load_missing_metadata(tmp_path: Path):
    with pytest.raises(InvalidAdapterError, match="Missing metadata"):
        load_adapter_metadata(tmp_path)


def test_load_corrupt_metadata(tmp_path: Path):
    (tmp_path / "fsvlm_metadata.json").write_text("not json{{{")
    with pytest.raises(InvalidAdapterError, match="Corrupt"):
        load_adapter_metadata(tmp_path)


def test_preserves_created_at(tmp_path: Path):
    meta = AdapterMetadata(
        adapter_name="test",
        base_model="test",
        created_at="2026-01-01T00:00:00+00:00",
    )
    save_adapter_metadata(tmp_path, meta)
    loaded = load_adapter_metadata(tmp_path)
    assert loaded.created_at == "2026-01-01T00:00:00+00:00"
