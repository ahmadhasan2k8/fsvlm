"""Custom exception hierarchy for FSVLM.

All exceptions inherit from FSVLMError. Each includes a `suggestion` field
with actionable guidance for the user. The Interface Layer catches these and
formats them for display.
"""

from __future__ import annotations


class FSVLMError(Exception):
    """Base exception for all FSVLM errors."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        self.suggestion = suggestion
        full = f"{message}\n  Suggestion: {suggestion}" if suggestion else message
        super().__init__(full)


class ModelNotFoundError(FSVLMError):
    """Raised when a requested model is not available locally."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Model '{model_name}' not found locally.",
            suggestion=f"Run: fsvlm setup --model {model_name}",
        )
        self.model_name = model_name


class InsufficientVRAMError(FSVLMError):
    """Raised when GPU VRAM is too low for the requested operation."""

    def __init__(self, available_gb: float, required_gb: float, model_name: str) -> None:
        super().__init__(
            f"Your GPU has {available_gb:.1f}GB VRAM. Model '{model_name}' requires ~{required_gb:.0f}GB.",
            suggestion="Try a smaller model: fsvlm setup --model small",
        )
        self.available_gb = available_gb
        self.required_gb = required_gb


class DatasetError(FSVLMError):
    """Raised for dataset-related issues (missing dirs, no images, corrupt files)."""


class InvalidAdapterError(FSVLMError):
    """Raised when an adapter cannot be loaded or is incompatible."""

    def __init__(self, adapter_path: str, reason: str = "") -> None:
        detail = f": {reason}" if reason else ""
        super().__init__(
            f"Invalid adapter at '{adapter_path}'{detail}.",
            suggestion="Re-train with: fsvlm train --images <path>",
        )


class TrainingError(FSVLMError):
    """Raised when training fails (OOM, NaN loss, etc.)."""
