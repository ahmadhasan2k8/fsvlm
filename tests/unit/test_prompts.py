"""Tests for fsvlm.prompts.generic."""

from __future__ import annotations

from fsvlm.prompts.generic import INSPECTION_PROMPT, build_training_conversation


def test_inspection_prompt_exists():
    assert "PASS" in INSPECTION_PROMPT or "inspector" in INSPECTION_PROMPT.lower()


def test_build_conversation_good():
    conv = build_training_conversation("good")
    assert len(conv) == 2
    assert conv[0]["role"] == "user"
    assert conv[1]["role"] == "assistant"

    # User message has image + text
    user_content = conv[0]["content"]
    assert any(p["type"] == "image" for p in user_content)
    assert any(p["type"] == "text" for p in user_content)

    # Assistant says PASS
    assistant_text = conv[1]["content"][0]["text"]
    assert assistant_text.startswith("PASS")


def test_build_conversation_defect():
    conv = build_training_conversation("defect")
    assistant_text = conv[1]["content"][0]["text"]
    assert assistant_text.startswith("FAIL")


def test_build_conversation_custom_description():
    conv = build_training_conversation("defect", description="Crack on left edge")
    assistant_text = conv[1]["content"][0]["text"]
    assert "FAIL" in assistant_text
    assert "Crack on left edge" in assistant_text


def test_build_conversation_custom_prompt():
    custom = "Check this PCB image."
    conv = build_training_conversation("good", prompt=custom)
    user_text = [p for p in conv[0]["content"] if p["type"] == "text"][0]["text"]
    assert user_text == custom
