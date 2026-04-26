"""Tests for fsvlm.agents.feedback_agent."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fsvlm.agents.feedback_agent import FeedbackAgent
from fsvlm.config import FSVLMConfig
from fsvlm.types import Correction


@pytest.fixture
def feedback_config(tmp_path: Path) -> FSVLMConfig:
    return FSVLMConfig(base_dir=tmp_path / ".fsvlm")


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "test.png"
    Image.new("RGB", (32, 32)).save(img_path)
    return img_path


def test_log_correction(feedback_config: FSVLMConfig, sample_image: Path):
    agent = FeedbackAgent(feedback_config)
    correction = Correction(
        image_path=sample_image,
        predicted_label="defect",
        actual_label="good",
        confidence=0.8,
    )
    count = agent.log_correction(correction)
    assert count == 1


def test_load_corrections(feedback_config: FSVLMConfig, sample_image: Path):
    agent = FeedbackAgent(feedback_config)
    for i in range(3):
        agent.log_correction(
            Correction(
                image_path=sample_image,
                predicted_label="good",
                actual_label="defect",
            )
        )

    corrections = agent.load_corrections()
    assert len(corrections) == 3
    assert corrections[0].actual_label == "defect"


def test_pending_count(feedback_config: FSVLMConfig, sample_image: Path):
    agent = FeedbackAgent(feedback_config)
    assert agent.pending_count() == 0

    agent.log_correction(
        Correction(
            image_path=sample_image,
            predicted_label="good",
            actual_label="defect",
        )
    )
    assert agent.pending_count() == 1


def test_should_retrain(feedback_config: FSVLMConfig, sample_image: Path):
    # Lower threshold for testing
    feedback_config.correction_retrain_threshold = 3
    agent = FeedbackAgent(feedback_config)

    assert not agent.should_retrain()

    for _ in range(3):
        agent.log_correction(
            Correction(
                image_path=sample_image,
                predicted_label="good",
                actual_label="defect",
            )
        )

    assert agent.should_retrain()


def test_corrections_as_samples(feedback_config: FSVLMConfig, sample_image: Path):
    agent = FeedbackAgent(feedback_config)
    agent.log_correction(
        Correction(
            image_path=sample_image,
            predicted_label="good",
            actual_label="defect",
            confidence=0.9,
        )
    )

    samples = agent.corrections_as_samples()
    assert len(samples) == 1
    assert samples[0].label == "defect"
    assert samples[0].image_path == sample_image


def test_corrections_as_samples_missing_image(feedback_config: FSVLMConfig):
    agent = FeedbackAgent(feedback_config)
    agent.log_correction(
        Correction(
            image_path=Path("/nonexistent/image.png"),
            predicted_label="good",
            actual_label="defect",
        )
    )

    samples = agent.corrections_as_samples()
    assert len(samples) == 0  # skipped because image missing


def test_clear_corrections(feedback_config: FSVLMConfig, sample_image: Path):
    agent = FeedbackAgent(feedback_config)
    agent.log_correction(
        Correction(
            image_path=sample_image,
            predicted_label="good",
            actual_label="defect",
        )
    )
    assert agent.pending_count() == 1

    agent.clear_corrections()
    assert agent.pending_count() == 0

    # Archived file should exist
    archives = list(feedback_config.corrections_dir.glob("corrections_*.jsonl"))
    assert len(archives) == 1


def test_summarize(feedback_config: FSVLMConfig, sample_image: Path):
    agent = FeedbackAgent(feedback_config)
    agent.log_correction(Correction(image_path=sample_image, predicted_label="good", actual_label="defect"))
    agent.log_correction(Correction(image_path=sample_image, predicted_label="good", actual_label="defect"))
    agent.log_correction(Correction(image_path=sample_image, predicted_label="defect", actual_label="good"))

    summary = agent.summarize()
    assert summary["good -> defect"] == 2
    assert summary["defect -> good"] == 1


def test_summarize_empty(feedback_config: FSVLMConfig):
    agent = FeedbackAgent(feedback_config)
    assert agent.summarize() == {}
