"""Generic prompt templates for defect detection.

These are the templates validated in Phase 0 on MVTec AD hazelnut.
Vertical-specific templates (Phase 3+) will extend this.
"""

from __future__ import annotations

# The inspection prompt that achieved AUROC 0.952, F1 0.846 in Phase 0
INSPECTION_PROMPT = (
    "You are a visual quality inspector. Examine this image. "
    "Respond with exactly PASS or FAIL on the first line. "
    "On the second line, describe what you see."
)


def build_training_conversation(
    label: str,
    prompt: str = INSPECTION_PROMPT,
    description: str = "",
) -> list[dict[str, object]]:
    """Build a VLM conversation pair for SFT training.

    Args:
        label: "good" or "defect".
        prompt: The inspection prompt to use.
        description: Optional description of what's in the image.

    Returns:
        List of message dicts in chat format (user + assistant).
    """
    pass_fail = "PASS" if label == "good" else "FAIL"

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
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": response},
            ],
        },
    ]
