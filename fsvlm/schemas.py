"""Pydantic models for external serialization.

Used by JSON report export and future REST API. These mirror the domain
dataclasses in types.py but add Pydantic validation and JSON serialization.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ValidationMetricsSchema(BaseModel):
    """Serializable validation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    optimal_threshold: float = 0.5


class ConfusionMatrixSchema(BaseModel):
    """Serializable confusion matrix."""

    matrix: list[list[int]]
    labels: list[str] = Field(default_factory=lambda: ["good", "defect"])


class FailureCaseSchema(BaseModel):
    """Serializable failure case."""

    image_path: str
    predicted: str
    actual: str
    confidence: float
    model_reasoning: str = ""


class ValidationReportSchema(BaseModel):
    """Serializable validation report for JSON export."""

    metrics: ValidationMetricsSchema
    confusion_matrix: ConfusionMatrixSchema
    failure_cases: list[FailureCaseSchema] = Field(default_factory=list)
    num_test_samples: int = 0
    confidence_scores: list[float] = Field(default_factory=list)
    summary: str = ""


class InspectionResultSchema(BaseModel):
    """Serializable inspection result for API responses."""

    image_path: str
    pass_fail: str  # "PASS" or "FAIL"
    confidence: float
    description: str
    model_name: str = ""
    adapter_name: str = ""
    inference_time_ms: float = 0.0


class SweepConfigSchema(BaseModel):
    """A single sweep configuration."""

    rank: int
    alpha: int
    learning_rate: float
    max_epochs: int


class SweepResultSchema(BaseModel):
    """Result of a single sweep candidate."""

    config: SweepConfigSchema
    metrics: ValidationMetricsSchema
    train_loss: float = 0.0
    elapsed_seconds: float = 0.0
    selected: bool = False
    notes: list[str] = Field(default_factory=list)


class TrainingReportSchema(BaseModel):
    """Full training report with sweep results."""

    best_config: SweepConfigSchema | None = None
    all_results: list[SweepResultSchema] = Field(default_factory=list)
    reasoning: str = ""
