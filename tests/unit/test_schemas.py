"""Tests for fsvlm.schemas."""

from __future__ import annotations

from fsvlm.schemas import (
    ConfusionMatrixSchema,
    InspectionResultSchema,
    SweepConfigSchema,
    SweepResultSchema,
    TrainingReportSchema,
    ValidationMetricsSchema,
    ValidationReportSchema,
)


def test_validation_metrics_schema():
    m = ValidationMetricsSchema(accuracy=0.95, precision=0.9, recall=0.85, f1=0.87, auroc=0.96)
    data = m.model_dump()
    assert data["auroc"] == 0.96
    assert data["optimal_threshold"] == 0.5


def test_validation_report_schema():
    r = ValidationReportSchema(
        metrics=ValidationMetricsSchema(accuracy=0.9, precision=0.85, recall=0.8, f1=0.82, auroc=0.9),
        confusion_matrix=ConfusionMatrixSchema(matrix=[[80, 10], [5, 15]]),
        num_test_samples=110,
    )
    data = r.model_dump()
    assert data["num_test_samples"] == 110
    assert len(data["confusion_matrix"]["matrix"]) == 2


def test_inspection_result_schema():
    r = InspectionResultSchema(
        image_path="test.jpg",
        pass_fail="FAIL",
        confidence=0.95,
        description="Crack detected",
    )
    assert r.pass_fail == "FAIL"


def test_sweep_config_schema():
    sc = SweepConfigSchema(rank=32, alpha=32, learning_rate=2e-4, max_epochs=10)
    assert sc.rank == 32


def test_training_report_schema():
    tr = TrainingReportSchema(
        best_config=SweepConfigSchema(rank=32, alpha=32, learning_rate=2e-4, max_epochs=10),
        all_results=[
            SweepResultSchema(
                config=SweepConfigSchema(rank=16, alpha=16, learning_rate=2e-4, max_epochs=10),
                metrics=ValidationMetricsSchema(accuracy=0.9, precision=0.85, recall=0.8, f1=0.82, auroc=0.9),
                selected=False,
            ),
        ],
        reasoning="Selected rank=32",
    )
    assert tr.reasoning == "Selected rank=32"
