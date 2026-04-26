"""Model download from HuggingFace Hub with progress display."""

from __future__ import annotations

from pathlib import Path

from fsvlm.exceptions import ModelNotFoundError
from fsvlm.types import ModelInfo


def is_model_cached(model: ModelInfo) -> bool:
    """Check if a model is already downloaded in the HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(model.hf_repo, "config.json")
        return result is not None and isinstance(result, str)
    except Exception:
        return False


def download_model(model: ModelInfo, show_progress: bool = True) -> Path:
    """Download a model from HuggingFace Hub.

    Args:
        model: ModelInfo with the HF repo to download.
        show_progress: Whether to show a rich progress bar.

    Returns:
        Path to the downloaded model snapshot directory.

    Raises:
        ModelNotFoundError: If the model cannot be found on HuggingFace.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ModelNotFoundError(model.hf_repo) from e

    try:
        from rich.console import Console

        console = Console()
        console.print(
            f"[bold]Downloading {model.name}[/bold] "
            f"({model.params_billions:.0f}B params, ~{model.vram_required_gb:.0f}GB VRAM required)"
        )

        path = snapshot_download(
            model.hf_repo,
            resume_download=True,
        )
        console.print(f"[green]Model downloaded to: {path}[/green]")
        return Path(path)

    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            raise ModelNotFoundError(model.hf_repo) from e
        raise
