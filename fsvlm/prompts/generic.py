"""Generic prompt templates for defect detection.

These are the templates validated in Phase 0 on MVTec AD hazelnut.
Vertical-specific templates (Phase 3+) will extend this.
"""

from __future__ import annotations

from fsvlm.prompts.verdict import resolve_inspection_prompt, verdict_tokens

# Canonical inspection prompt template. $pass_token / $fail_token are substituted
# per-backbone via resolve_inspection_prompt — Gemma/Qwen/Llama get PASS/FAIL;
# Pixtral (Mistral tokenizer) gets Pass/Fail.
INSPECTION_PROMPT = (
    "You are a visual quality inspector. Examine this image. "
    "Respond with exactly $pass_token or $fail_token on the first line. "
    "On the second line, describe what you see."
)


def build_training_conversation(
    label: str,
    prompt: str = INSPECTION_PROMPT,
    description: str = "",
    model_name: str | None = None,
) -> list[dict[str, object]]:
    """Build a VLM conversation pair for SFT training.

    Args:
        label: "good" or "defect".
        prompt: The inspection prompt to use.
        description: Optional description of what's in the image.
        model_name: HF model name; selects the verdict pair. None uses
            ``FSVLM_DEFAULT_MODEL`` env var or the default pair.

    Returns:
        List of message dicts in chat format (user + assistant).
    """
    pass_str, fail_str = verdict_tokens(model_name)
    pass_fail = pass_str if label == "good" else fail_str
    resolved_prompt = resolve_inspection_prompt(prompt, model_name)

    if description:
        response = f"{pass_fail}\n{description}"
    else:
        default_desc = (
            "The item appears to be in good condition with no visible defects."
            if label == "good"
            else "A defect has been detected in the item."
        )
        response = f"{pass_fail}\n{default_desc}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": resolved_prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": response},
            ],
        },
    ]
