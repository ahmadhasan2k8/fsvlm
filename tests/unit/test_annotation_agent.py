"""Tests for fsvlm.agents.annotation_agent."""

from __future__ import annotations

from pathlib import Path

from fsvlm.agents.annotation_agent import AnnotationAgent
from fsvlm.config import FSVLMConfig
from fsvlm.types import (
    AnnotatedImage,
    AnnotationSession,
    DefectAnnotation,
)


def _make_session() -> AnnotationSession:
    """Create a test annotation session."""
    return AnnotationSession(
        images=[
            AnnotatedImage(
                image_path=Path("img1.png"),
                annotations=[
                    DefectAnnotation(
                        click_x=100,
                        click_y=50,
                        mask_rle="10,10:5 3",
                        iou_score=0.95,
                        user_description="there's a crack here",
                        location_description="upper-left region",
                    ),
                    DefectAnnotation(
                        click_x=200,
                        click_y=300,
                        mask_rle="10,10:80 5",
                        iou_score=0.88,
                        user_description="dark stain on surface",
                        location_description="lower-right region",
                    ),
                ],
            ),
            AnnotatedImage(
                image_path=Path("img2.png"),
                annotations=[
                    DefectAnnotation(
                        click_x=50,
                        click_y=50,
                        mask_rle="10,10:40 10",
                        iou_score=0.92,
                        user_description="scratch along the edge",
                        location_description="center region",
                    ),
                ],
            ),
            AnnotatedImage(
                image_path=Path("img3.png"),
                is_good=True,
            ),
        ],
    )


def test_annotations_to_samples():
    """Convert annotations to LabeledSamples."""
    config = FSVLMConfig()
    agent = AnnotationAgent(config)
    session = _make_session()

    # Pre-assign defect types (normally done by classify_annotations)
    session.images[0].annotations[0].defect_type = "crack"
    session.images[0].annotations[1].defect_type = "stain"
    session.images[1].annotations[0].defect_type = "scratch"

    samples = agent.annotations_to_samples(session)

    # Each annotation becomes its own sample: 2 from img1, 1 from img2, 1 good
    assert len(samples) == 4
    assert samples[0].label == "defect"
    assert "crack" in samples[0].description.lower()
    assert samples[0].image_path == Path("img1.png")
    assert samples[1].label == "defect"
    assert "stain" in samples[1].description.lower()
    assert samples[1].image_path == Path("img1.png")
    assert samples[2].label == "defect"
    assert "scratch" in samples[2].description.lower()
    assert samples[2].image_path == Path("img2.png")
    assert samples[3].label == "good"


def test_annotations_to_samples_no_annotations():
    """Images with no annotations and not marked good should be skipped."""
    config = FSVLMConfig()
    agent = AnnotationAgent(config)
    session = AnnotationSession(
        images=[AnnotatedImage(image_path=Path("img.png"), annotations=[])],
    )

    samples = agent.annotations_to_samples(session)
    assert len(samples) == 0


def test_annotations_to_samples_good_only():
    """Good-only session should produce good samples."""
    config = FSVLMConfig()
    agent = AnnotationAgent(config)
    session = AnnotationSession(
        images=[
            AnnotatedImage(image_path=Path("good1.png"), is_good=True),
            AnnotatedImage(image_path=Path("good2.png"), is_good=True),
        ],
    )

    samples = agent.annotations_to_samples(session)
    assert len(samples) == 2
    assert all(s.label == "good" for s in samples)


def test_fallback_taxonomy():
    """Fallback taxonomy should extract keywords from descriptions."""
    taxonomy = AnnotationAgent._fallback_taxonomy(
        [
            "there's a crack here",
            "scratch along the edge",
            "nothing obvious",
        ]
    )

    assert "crack" in taxonomy
    assert "scratch" in taxonomy


def test_fallback_taxonomy_no_keywords():
    """Fallback should return generic 'defect' when no keywords match."""
    taxonomy = AnnotationAgent._fallback_taxonomy(
        [
            "something weird here",
            "this looks wrong",
        ]
    )

    assert "defect" in taxonomy


def test_annotation_types():
    """Test annotation dataclass defaults."""
    ann = DefectAnnotation(click_x=10, click_y=20)
    assert ann.mask_rle == ""
    assert ann.iou_score == 0.0
    assert ann.user_description == ""
    assert ann.defect_type == ""
    assert ann.location_description == ""


def test_annotated_image_defaults():
    """Test annotated image dataclass defaults."""
    img = AnnotatedImage(image_path=Path("test.png"))
    assert img.annotations == []
    assert img.is_good is False


def test_annotation_session_defaults():
    """Test annotation session dataclass defaults."""
    session = AnnotationSession()
    assert session.images == []
    assert session.defect_taxonomy == {}
    assert session.created_at == ""
