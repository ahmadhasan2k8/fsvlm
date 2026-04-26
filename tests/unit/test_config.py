"""Tests for fsvlm.config."""

from __future__ import annotations

from pathlib import Path

from fsvlm.config import FSVLMConfig, load_config


def test_default_config():
    c = load_config()
    assert c.default_model == "unsloth/gemma-4-E4B-it"
    assert c.default_seed == 3407
    assert c.default_train_split == 0.8
    assert c.min_precision == 0.75
    assert c.min_recall == 0.75


def test_derived_paths():
    c = FSVLMConfig(base_dir=Path("/tmp/test-fsvlm"))
    assert c.models_dir == Path("/tmp/test-fsvlm/models")
    assert c.adapters_dir == Path("/tmp/test-fsvlm/adapters")
    assert c.logs_dir == Path("/tmp/test-fsvlm/logs")


def test_ensure_dirs(tmp_path: Path):
    c = FSVLMConfig(base_dir=tmp_path / "dv")
    c.ensure_dirs()
    assert c.base_dir.exists()
    assert c.models_dir.exists()
    assert c.adapters_dir.exists()
    assert c.logs_dir.exists()


def test_env_override(monkeypatch: object):
    """Environment variables override defaults."""
    import os

    os.environ["FSVLM_DEFAULT_SEED"] = "42"
    try:
        c = FSVLMConfig()
        assert c.default_seed == 42
    finally:
        del os.environ["FSVLM_DEFAULT_SEED"]
