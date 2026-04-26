"""Adapter metadata save/load — no signing yet (Phase 4)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fsvlm.exceptions import InvalidAdapterError
from fsvlm.types import AdapterMetadata, ValidationMetrics

METADATA_FILENAME = "fsvlm_metadata.json"
CURRENT_SCHEMA_VERSION = 1


def next_adapter_version(adapters_dir: Path, adapter_name: str = "") -> int:
    """Determine the next adapter version by scanning existing adapters.

    Args:
        adapters_dir: Root adapters directory.
        adapter_name: Base adapter name to match (empty matches all).

    Returns:
        Next version number (1 if no previous adapters found).
    """
    max_version = 0
    if not adapters_dir.exists():
        return 1

    for d in adapters_dir.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / METADATA_FILENAME
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
                if adapter_name and data.get("adapter_name", "") != adapter_name:
                    continue
                max_version = max(max_version, data.get("adapter_version", 1))
            except (json.JSONDecodeError, OSError):
                continue
        # Check one level deeper
        for child in d.iterdir():
            if child.is_dir():
                child_meta = child / METADATA_FILENAME
                if child_meta.exists():
                    try:
                        data = json.loads(child_meta.read_text())
                        if adapter_name and data.get("adapter_name", "") != adapter_name:
                            continue
                        max_version = max(max_version, data.get("adapter_version", 1))
                    except (json.JSONDecodeError, OSError):
                        continue

    return max_version + 1


def save_adapter_metadata(adapter_dir: Path, metadata: AdapterMetadata) -> Path:
    """Save adapter metadata alongside the adapter weights.

    Args:
        adapter_dir: Directory containing adapter weights.
        metadata: Metadata to save.

    Returns:
        Path to the saved metadata file.
    """
    if not metadata.created_at:
        metadata.created_at = datetime.now(timezone.utc).isoformat()

    meta_path = adapter_dir / METADATA_FILENAME
    data = asdict(metadata)
    meta_path.write_text(json.dumps(data, indent=2, default=str))
    return meta_path


def load_adapter_metadata(adapter_dir: Path) -> AdapterMetadata:
    """Load adapter metadata from a directory.

    Checks schema_version and migrates if needed.

    Args:
        adapter_dir: Directory containing adapter weights and metadata.

    Returns:
        Loaded AdapterMetadata.

    Raises:
        InvalidAdapterError: If metadata is missing or corrupt.
    """
    meta_path = adapter_dir / METADATA_FILENAME
    if not meta_path.exists():
        raise InvalidAdapterError(str(adapter_dir), reason="Missing metadata file")

    try:
        data = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise InvalidAdapterError(str(adapter_dir), reason=f"Corrupt metadata: {e}") from e

    schema_version = data.get("schema_version", 1)

    # Migration: add new fields with defaults for older schemas
    if schema_version < 1:
        raise InvalidAdapterError(str(adapter_dir), reason=f"Unknown schema version: {schema_version}")

    # Reconstruct ValidationMetrics if present
    val_metrics = data.get("validation_metrics")
    if val_metrics and isinstance(val_metrics, dict):
        val_metrics = ValidationMetrics(**val_metrics)
    else:
        val_metrics = None

    return AdapterMetadata(
        schema_version=data.get("schema_version", 1),
        adapter_name=data.get("adapter_name", ""),
        base_model=data.get("base_model", ""),
        lora_rank=data.get("lora_rank", 8),
        lora_alpha=data.get("lora_alpha", 8),
        training_images=data.get("training_images", 0),
        training_epochs=data.get("training_epochs", 0),
        validation_metrics=val_metrics,
        prompt_template=data.get("prompt_template", ""),
        created_at=data.get("created_at", ""),
        adapter_version=data.get("adapter_version", 1),
        parent_adapter=data.get("parent_adapter", ""),
    )
