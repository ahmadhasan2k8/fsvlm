"""Domain types for FSVLM.

Pure Python dataclasses — no external dependencies (no torch, no PIL, no sklearn).
All inter-agent communication uses these types. Dependencies flow inward only:
agents import types, types import nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LabeledSample:
    """A single labeled image for training or evaluation."""

    image_path: Path
    label: str  # "good" or "defect"
    description: str = ""  # optional natural-language description


@dataclass
class DatasetReport:
    """Summary of a prepared dataset."""

    total_images: int
    good_count: int
    defect_count: int
    corrupt_skipped: list[str] = field(default_factory=list)
    image_sizes: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class PreparedDataset:
    """Output of the Data Agent — ready for training."""

    train_samples: list[LabeledSample]
    val_samples: list[LabeledSample]
    report: DatasetReport
    seed: int = 3407


# ---------------------------------------------------------------------------
# Hardware types
# ---------------------------------------------------------------------------


@dataclass
class GPUInfo:
    """GPU hardware information."""

    name: str
    vram_total_gb: float
    vram_free_gb: float
    cuda_version: str
    compute_capability: tuple[int, int] = (0, 0)
    is_available: bool = True


@dataclass
class ModelInfo:
    """Information about a downloadable/local model."""

    name: str  # e.g. "gemma-4-E4B-it"
    hf_repo: str  # e.g. "unsloth/gemma-4-E4B-it"
    size_label: str  # "small", "medium", "large"
    params_billions: float
    vram_required_gb: float
    local_path: Path | None = None


# ---------------------------------------------------------------------------
# Training configuration types
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    rank: int = 8
    alpha: int = 8
    dropout: float = 0.0
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


@dataclass
class TrainingConfig:
    """Full training configuration."""

    model_name: str = "unsloth/gemma-4-E4B-it"
    load_in_4bit: bool = True
    max_seq_length: int = 1024

    lora: LoRAConfig = field(default_factory=LoRAConfig)

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    bf16: bool = True

    seed: int = 3407
    max_image_size: int = 560
    output_dir: Path = field(default_factory=lambda: Path("output"))

    inspection_prompt: str = (
        "You are a visual quality inspector. Examine this image. "
        "Respond with exactly PASS or FAIL on the first line. "
        "On the second line, describe what you see."
    )


# ---------------------------------------------------------------------------
# Sweep types
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """A single sweep candidate configuration."""

    rank: int = 16
    alpha: int = 16
    learning_rate: float = 2e-4
    max_epochs: int = 10


@dataclass
class SweepResult:
    """Result of a single sweep candidate."""

    config: SweepConfig
    metrics: ValidationMetrics | None = None
    train_loss: float = 0.0
    elapsed_seconds: float = 0.0
    selected: bool = False
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Training result types
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Output of the Training Agent."""

    adapter_path: Path
    config: TrainingConfig
    train_loss_history: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    sweep_results: list[SweepResult] = field(default_factory=list)
    sweep_reasoning: str = ""


# ---------------------------------------------------------------------------
# Validation types
# ---------------------------------------------------------------------------


@dataclass
class ConfusionMatrixData:
    """Confusion matrix storage."""

    matrix: list[list[int]]  # [[TN, FP], [FN, TP]] for binary
    labels: list[str] = field(default_factory=lambda: ["good", "defect"])


@dataclass
class FailureCase:
    """A single misclassified sample."""

    image_path: Path
    predicted: str
    actual: str
    confidence: float
    model_reasoning: str = ""


@dataclass
class ValidationMetrics:
    """Core evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    optimal_threshold: float = 0.5


@dataclass
class ValidationReport:
    """Full output of the Validation Agent."""

    metrics: ValidationMetrics
    confusion_matrix: ConfusionMatrixData
    failure_cases: list[FailureCase] = field(default_factory=list)
    num_test_samples: int = 0
    confidence_scores: list[float] = field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Inference types
# ---------------------------------------------------------------------------


@dataclass
class DefectFinding:
    """A single defect found in an image."""

    type: str  # e.g. "crack", "contamination"
    location: str  # natural language location
    severity: str  # "critical", "major", "minor", "cosmetic"
    confidence: float


@dataclass
class InspectionResult:
    """Output of the Inspector Agent for a single image."""

    image_path: Path
    pass_fail: bool  # True = PASS, False = FAIL
    confidence: float
    description: str
    defects: list[DefectFinding] = field(default_factory=list)
    model_name: str = ""
    adapter_name: str = ""
    adapter_version: int = 1
    inference_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Adapter metadata
# ---------------------------------------------------------------------------


@dataclass
class Correction:
    """A user correction for a previous inspection result."""

    image_path: Path
    predicted_label: str  # what the model said
    actual_label: str  # what the user corrected to
    confidence: float = 0.0  # model's confidence on the wrong prediction
    timestamp: str = ""  # ISO 8601
    adapter_name: str = ""
    adapter_version: int = 1
    notes: str = ""


# ---------------------------------------------------------------------------
# Adapter metadata
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Annotation types (SAM-assisted defect annotation)
# ---------------------------------------------------------------------------


@dataclass
class DefectAnnotation:
    """A single user-annotated defect region in an image.

    Created when user clicks on a defect and SAM segments it.
    """

    click_x: int  # user click x coordinate (pixels)
    click_y: int  # user click y coordinate (pixels)
    mask_rle: str = ""  # run-length encoded mask (compact storage)
    iou_score: float = 0.0  # SAM confidence for this mask
    user_description: str = ""  # user's free-text: "what's wrong here?"
    defect_type: str = ""  # LLM-classified type (filled by annotation agent)
    location_description: str = ""  # auto-generated: "upper-left region"


@dataclass
class AnnotatedImage:
    """An image with one or more user-annotated defect regions."""

    image_path: Path
    annotations: list[DefectAnnotation] = field(default_factory=list)
    is_good: bool = False  # True if user marked as "no defects"


@dataclass
class AnnotationSession:
    """A collection of annotated images ready for classification and training.

    The annotation agent processes this into a PreparedDataset.
    """

    images: list[AnnotatedImage] = field(default_factory=list)
    defect_taxonomy: dict[str, str] = field(default_factory=dict)  # type → description
    created_at: str = ""  # ISO 8601


@dataclass
class AdapterMetadata:
    """Metadata saved alongside a trained adapter.

    schema_version must be incremented when the format changes.
    Load code checks schema_version and migrates if needed.
    """

    schema_version: int = 1
    adapter_name: str = ""
    base_model: str = ""
    lora_rank: int = 8
    lora_alpha: int = 8
    training_images: int = 0
    training_epochs: int = 0
    validation_metrics: ValidationMetrics | None = None
    prompt_template: str = ""
    created_at: str = ""  # ISO 8601
    adapter_version: int = 1  # incremented on retrain
    parent_adapter: str = ""  # path to previous version (for retrain lineage)
