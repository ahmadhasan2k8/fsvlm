"""Tests for fsvlm.reports."""

from __future__ import annotations

import json
from pathlib import Path

from fsvlm.reports.html_report import HTMLReportGenerator
from fsvlm.reports.json_report import JSONReportGenerator
from fsvlm.types import (
    ConfusionMatrixData,
    FailureCase,
    ValidationMetrics,
    ValidationReport,
)


def _make_report() -> ValidationReport:
    return ValidationReport(
        metrics=ValidationMetrics(0.95, 0.92, 0.88, 0.90, 0.97, 0.42),
        confusion_matrix=ConfusionMatrixData([[80, 5], [3, 12]]),
        failure_cases=[
            FailureCase(Path("img1.png"), "good", "defect", 0.6, "Missed crack"),
            FailureCase(Path("img2.png"), "defect", "good", 0.7, "False alarm"),
        ],
        num_test_samples=100,
        confidence_scores=[0.1, 0.9, 0.5],
    )


def test_json_report(tmp_path: Path):
    gen = JSONReportGenerator()
    assert gen.file_extension() == ".json"

    report = _make_report()
    path = gen.generate(report, tmp_path / "report")
    assert path.exists()
    assert path.suffix == ".json"

    data = json.loads(path.read_text())
    assert data["metrics"]["auroc"] == 0.97
    assert data["num_test_samples"] == 100
    assert len(data["failure_cases"]) == 2


def test_html_report(tmp_path: Path):
    gen = HTMLReportGenerator()
    assert gen.file_extension() == ".html"

    report = _make_report()
    path = gen.generate(report, tmp_path / "report")
    assert path.exists()
    assert path.suffix == ".html"

    content = path.read_text()
    assert "97.0%" in content  # AUROC
    assert "90.0%" in content  # F1
    assert "MISSED" in content or "FALSE ALARM" in content
    assert "FSVLM" in content


def test_html_report_no_failures(tmp_path: Path):
    report = ValidationReport(
        metrics=ValidationMetrics(1.0, 1.0, 1.0, 1.0, 1.0),
        confusion_matrix=ConfusionMatrixData([[50, 0], [0, 10]]),
        num_test_samples=60,
    )
    gen = HTMLReportGenerator()
    path = gen.generate(report, tmp_path / "perfect")
    content = path.read_text()
    assert "100.0%" in content
    assert "MISSED" not in content
