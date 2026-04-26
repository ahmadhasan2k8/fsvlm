"""Tests for fsvlm.models.hardware."""

from __future__ import annotations

from fsvlm.models.hardware import (
    KNOWN_MODELS,
    detect_gpu,
    get_model_by_name,
    recommend_model,
)
from fsvlm.types import GPUInfo


def test_detect_gpu():
    gpu = detect_gpu()
    # Should return a GPUInfo regardless of whether GPU is present
    assert isinstance(gpu, GPUInfo)
    assert isinstance(gpu.name, str)


def test_known_models_sorted_by_size():
    sizes = [m.vram_required_gb for m in KNOWN_MODELS]
    assert sizes == sorted(sizes), "KNOWN_MODELS should be sorted by VRAM requirement"


def test_recommend_model_no_gpu():
    gpu = GPUInfo(name="none", vram_total_gb=0, vram_free_gb=0, cuda_version="", is_available=False)
    rec = recommend_model(gpu)
    assert rec.size_label == "small"


def test_recommend_model_8gb():
    gpu = GPUInfo(name="test", vram_total_gb=8.0, vram_free_gb=8.0, cuda_version="12.0")
    rec = recommend_model(gpu)
    assert rec.name == "gemma-4-E4B-it"


def test_recommend_model_16gb():
    gpu = GPUInfo(name="test", vram_total_gb=16.0, vram_free_gb=16.0, cuda_version="12.0")
    rec = recommend_model(gpu)
    assert rec.name == "gemma-4-12B-it"


def test_get_model_by_name():
    m = get_model_by_name("gemma-4-E4B-it")
    assert m is not None
    assert m.hf_repo == "unsloth/gemma-4-E4B-it"


def test_get_model_by_hf_repo():
    m = get_model_by_name("unsloth/gemma-4-E2B-it")
    assert m is not None
    assert m.name == "gemma-4-E2B-it"


def test_get_model_unknown():
    m = get_model_by_name("nonexistent-model")
    assert m is None
