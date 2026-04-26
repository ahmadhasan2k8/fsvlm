"""Tests for fsvlm.types — domain dataclasses."""

from __future__ import annotations

from pathlib import Path

from fsvlm.types import (
    AdapterMetadata,
    AnnotatedImage,
    AnnotationSession,
    ConfusionMatrixData,
    Correction,
    DatasetReport,
    DefectAnnotation,
    DefectFinding,
    FailureCase,
    GPUInfo,
    InspectionResult,
    LabeledSample,
    LoRAConfig,
    ModelInfo,
    PreparedDataset,
    TrainingConfig,
    TrainingResult,
    ValidationMetrics,
    ValidationReport,
)


def test_labeled_sample_defaults():
    s = LabeledSample(image_path=Path("img.png"), label="good")
    assert s.label == "good"
    assert s.description == ""


def test_dataset_report():
    r = DatasetReport(total_images=100, good_count=80, defect_count=20)
    assert r.total_images == 100
    assert r.corrupt_skipped == []


def test_prepared_dataset():
    ds = PreparedDataset(
        train_samples=[LabeledSample(Path("a.png"), "good")],
        val_samples=[LabeledSample(Path("b.png"), "defect")],
        report=DatasetReport(total_images=2, good_count=1, defect_count=1),
    )
    assert len(ds.train_samples) == 1
    assert ds.seed == 3407


def test_gpu_info_defaults():
    g = GPUInfo(name="RTX 5080", vram_total_gb=16.0, vram_free_gb=14.0, cuda_version="12.8")
    assert g.is_available is True
    assert g.compute_capability == (0, 0)


def test_model_info():
    m = ModelInfo(
        name="test-model",
        hf_repo="org/test-model",
        size_label="small",
        params_billions=2.0,
        vram_required_gb=4.0,
    )
    assert m.local_path is None


def test_lora_config_defaults():
    lc = LoRAConfig()
    assert lc.rank == 8
    assert lc.alpha == 8
    assert lc.dropout == 0.0


def test_training_config_defaults():
    tc = TrainingConfig()
    assert tc.model_name == "unsloth/gemma-4-E4B-it"
    assert tc.lora.rank == 8
    assert tc.bf16 is True
    assert tc.seed == 3407


def test_training_result():
    tr = TrainingResult(
        adapter_path=Path("adapter"),
        config=TrainingConfig(),
    )
    assert tr.elapsed_seconds == 0.0
    assert tr.train_loss_history == []


def test_validation_metrics():
    vm = ValidationMetrics(accuracy=0.95, precision=0.9, recall=0.85, f1=0.87, auroc=0.96)
    assert vm.optimal_threshold == 0.5


def test_confusion_matrix_data():
    cm = ConfusionMatrixData(matrix=[[85, 5], [3, 12]])
    assert cm.labels == ["good", "defect"]


def test_failure_case():
    fc = FailureCase(
        image_path=Path("img.png"),
        predicted="good",
        actual="defect",
        confidence=0.8,
    )
    assert fc.model_reasoning == ""


def test_validation_report():
    vr = ValidationReport(
        metrics=ValidationMetrics(0.9, 0.85, 0.8, 0.82, 0.9),
        confusion_matrix=ConfusionMatrixData([[80, 10], [5, 15]]),
    )
    assert vr.num_test_samples == 0
    assert vr.summary == ""


def test_defect_finding():
    df = DefectFinding(type="crack", location="top-left", severity="major", confidence=0.9)
    assert df.severity == "major"


def test_inspection_result():
    ir = InspectionResult(
        image_path=Path("test.jpg"),
        pass_fail=False,
        confidence=0.92,
        description="Crack detected",
    )
    assert ir.adapter_version == 1
    assert ir.pass_fail is False


def test_adapter_metadata_schema_version():
    am = AdapterMetadata(adapter_name="test", base_model="gemma-4-E4B-it")
    assert am.schema_version == 1
    assert am.created_at == ""


def test_adapter_metadata_with_metrics():
    metrics = ValidationMetrics(0.95, 0.9, 0.88, 0.89, 0.97)
    am = AdapterMetadata(
        adapter_name="hazelnut-v1",
        base_model="gemma-4-E4B-it",
        validation_metrics=metrics,
    )
    assert am.validation_metrics is not None
    assert am.validation_metrics.auroc == 0.97


def test_adapter_metadata_versioning():
    am = AdapterMetadata(adapter_name="test", adapter_version=2, parent_adapter="/old/path")
    assert am.adapter_version == 2
    assert am.parent_adapter == "/old/path"


def test_correction_defaults():
    c = Correction(
        image_path=Path("img.png"),
        predicted_label="good",
        actual_label="defect",
    )
    assert c.confidence == 0.0
    assert c.timestamp == ""
    assert c.adapter_version == 1


def test_correction_full():
    c = Correction(
        image_path=Path("img.png"),
        predicted_label="defect",
        actual_label="good",
        confidence=0.85,
        timestamp="2026-04-06T12:00:00Z",
        adapter_name="hazelnut",
        adapter_version=2,
        notes="False alarm on surface texture",
    )
    assert c.actual_label == "good"
    assert c.notes == "False alarm on surface texture"


def test_defect_annotation_defaults():
    ann = DefectAnnotation(click_x=100, click_y=200)
    assert ann.mask_rle == ""
    assert ann.iou_score == 0.0
    assert ann.user_description == ""
    assert ann.defect_type == ""
    assert ann.location_description == ""


def test_annotated_image_with_annotations():
    ann = DefectAnnotation(click_x=10, click_y=20, user_description="crack")
    img = AnnotatedImage(
        image_path=Path("test.png"),
        annotations=[ann],
    )
    assert len(img.annotations) == 1
    assert img.is_good is False


def test_annotation_session_taxonomy():
    session = AnnotationSession(
        defect_taxonomy={"crack": "Surface crack", "stain": "Dark stain"},
    )
    assert len(session.defect_taxonomy) == 2
    assert session.images == []
