"""Tests for InspectorSession and batch inspect."""

from __future__ import annotations

from pathlib import Path

from fsvlm.agents.inspector_agent import (
    IMAGE_EXTENSIONS,
    InspectorAgent,
    InspectorSession,
)
from fsvlm.config import FSVLMConfig


def test_image_extensions():
    assert ".png" in IMAGE_EXTENSIONS
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".jpeg" in IMAGE_EXTENSIONS
    assert ".txt" not in IMAGE_EXTENSIONS


def test_inspector_is_loaded():
    config = FSVLMConfig()
    inspector = InspectorAgent(config)
    assert inspector.is_loaded is False


def test_inspect_batch_without_model():
    config = FSVLMConfig()
    inspector = InspectorAgent(config)
    # inspect_batch calls inspect, which checks is_loaded
    results = inspector.inspect_batch([Path("a.png")])
    assert len(results) == 0  # all fail silently


def test_inspector_session_type():
    config = FSVLMConfig()
    session = InspectorSession(config, Path("/tmp/adapter"))
    assert hasattr(session, "__enter__")
    assert hasattr(session, "__exit__")
