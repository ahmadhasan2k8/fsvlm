"""Tests for fsvlm.exceptions."""

from __future__ import annotations

from fsvlm.exceptions import (
    DatasetError,
    FSVLMError,
    InsufficientVRAMError,
    InvalidAdapterError,
    ModelNotFoundError,
    TrainingError,
)


def test_base_error():
    e = FSVLMError("something broke", suggestion="try again")
    assert "something broke" in str(e)
    assert "try again" in str(e)
    assert e.suggestion == "try again"


def test_base_error_no_suggestion():
    e = FSVLMError("just broke")
    assert "just broke" in str(e)
    assert e.suggestion == ""


def test_model_not_found():
    e = ModelNotFoundError("gemma-4-E4B-it")
    assert "gemma-4-E4B-it" in str(e)
    assert "fsvlm setup" in str(e)
    assert e.model_name == "gemma-4-E4B-it"


def test_insufficient_vram():
    e = InsufficientVRAMError(8.0, 16.0, "gemma-4-12B-it")
    assert "8.0GB" in str(e)
    assert "16GB" in str(e)
    assert e.available_gb == 8.0


def test_dataset_error():
    e = DatasetError("No images", suggestion="Add some images")
    assert isinstance(e, FSVLMError)


def test_invalid_adapter():
    e = InvalidAdapterError("/path/to/adapter", reason="corrupt weights")
    assert "corrupt weights" in str(e)
    assert "fsvlm train" in str(e)


def test_training_error():
    e = TrainingError("OOM", suggestion="Try smaller batch size")
    assert isinstance(e, FSVLMError)


def test_exception_hierarchy():
    assert issubclass(ModelNotFoundError, FSVLMError)
    assert issubclass(InsufficientVRAMError, FSVLMError)
    assert issubclass(DatasetError, FSVLMError)
    assert issubclass(InvalidAdapterError, FSVLMError)
    assert issubclass(TrainingError, FSVLMError)
    assert issubclass(FSVLMError, Exception)
