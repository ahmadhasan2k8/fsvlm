"""Configuration management for FSVLM.

Priority: CLI flags > config.toml > environment variables > hardcoded defaults.
Config file lives at ~/.fsvlm/config.toml.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


def _default_base_dir() -> Path:
    return Path.home() / ".fsvlm"


class FSVLMConfig(BaseSettings):
    """Central configuration for FSVLM.

    All thresholds, defaults, and limits live here — no magic numbers in code.
    """

    model_config = {"env_prefix": "FSVLM_"}

    # Directories
    base_dir: Path = Field(default_factory=_default_base_dir)

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def adapters_dir(self) -> Path:
        return self.base_dir / "adapters"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def corrections_dir(self) -> Path:
        return self.base_dir / "corrections"

    # Model defaults
    default_model: str = "unsloth/gemma-4-E4B-it"
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    max_image_size: int = 560

    # Training defaults
    default_lora_rank: int = 8
    default_lora_alpha: int = 8
    default_learning_rate: float = 2e-4
    default_max_epochs: int = 3
    default_batch_size: int = 1
    default_gradient_accumulation: int = 8
    default_train_split: float = 0.8
    default_seed: int = 3407

    # Validation thresholds
    min_precision: float = 0.75
    min_recall: float = 0.75

    # Sweep
    sweep_enabled: bool = True

    # Feedback
    correction_retrain_threshold: int = 20  # retrain suggestion after N corrections

    # Watch mode
    watch_debounce_seconds: float = 2.0  # debounce rapid file arrivals

    # Server
    serve_host: str = "0.0.0.0"
    serve_port: int = 8080

    # UI
    ui_port: int = 7860

    # Telemetry
    telemetry_enabled: bool = False

    def ensure_dirs(self) -> None:
        """Create config directories if they don't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> FSVLMConfig:
    """Load config from environment and TOML file.

    TOML support deferred to when pydantic-settings adds toml source,
    or when we add tomli. For now, env vars and defaults.
    """
    return FSVLMConfig()
