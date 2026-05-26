"""Per-backbone verdict tokens for single-token logit-ratio scoring.

Single-token PASS/FAIL logit-ratio scoring requires the verdict strings to each
tokenize to exactly one token under the loaded tokenizer. This holds for Gemma,
Qwen, and Llama tokenizers. The Mistral family (Pixtral-12B-2409) splits both
PASS and FAIL into two subwords, so the case-changed pair 'Pass' / 'Fail' is
used instead — both are single-token in the Mistral tokenizer and semantically
identical.

The verdict pair must be the SAME at every stage: training labels, inference
prompts, response classification, and logit-ratio scoring. Use ``verdict_tokens``
at every site that constructs or compares those strings; use
``verdict_token_ids`` at every site that does logit-position scoring.

Adding a new backbone
---------------------
1. Probe the tokenizer:
       from transformers import AutoTokenizer
       t = AutoTokenizer.from_pretrained("...")
       t.encode("PASS", add_special_tokens=False)  # check len == 1
2. If PASS/FAIL splits, find a single-token pair (e.g. 'Pass'/'Fail',
   'Yes'/'No') and add a model_name → pair entry below.
"""

from __future__ import annotations

import os
from string import Template

_DEFAULT: tuple[str, str] = ("PASS", "FAIL")

# Per-model overrides. Key is the HF model name passed to FastVisionModel.
_OVERRIDES: dict[str, tuple[str, str]] = {
    # Mistral tokenizer splits PASS/FAIL into 2 subwords each.
    # 'Pass' → 21889, 'Fail' → 48565 — both single-token.
    "unsloth/Pixtral-12B-2409": ("Pass", "Fail"),
    "unsloth/Pixtral-12B-2409-bnb-4bit": ("Pass", "Fail"),
    "mistralai/Pixtral-12B-2409": ("Pass", "Fail"),
}


def verdict_tokens(model_name: str | None = None) -> tuple[str, str]:
    """Return the (pass_token, fail_token) string pair for a model.

    Args:
        model_name: HF model name. If None, reads ``FSVLM_DEFAULT_MODEL`` from
            the environment, falling back to the default pair.

    Returns:
        Tuple ``(pass_token, fail_token)``.
    """
    if model_name is None:
        model_name = os.environ.get("FSVLM_DEFAULT_MODEL", "")
    return _OVERRIDES.get(model_name, _DEFAULT)


def verdict_token_ids(tokenizer, model_name: str | None = None) -> tuple[int, int]:
    """Encode the verdict pair and return their token IDs.

    Validates each verdict tokenizes to exactly one ID. Raises ``AssertionError``
    with guidance to update ``_OVERRIDES`` if either string is multi-subword.
    """
    pass_str, fail_str = verdict_tokens(model_name)
    _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    pass_ids = _tok.encode(pass_str, add_special_tokens=False)
    fail_ids = _tok.encode(fail_str, add_special_tokens=False)
    assert len(pass_ids) == 1, (
        f"Tokenizer splits '{pass_str}' into {len(pass_ids)} subwords {pass_ids}; "
        f"single-token logit scoring is invalid. Register a single-token verdict "
        f"pair for {model_name!r} in fsvlm/prompts/verdict.py:_OVERRIDES."
    )
    assert len(fail_ids) == 1, (
        f"Tokenizer splits '{fail_str}' into {len(fail_ids)} subwords {fail_ids}; "
        f"single-token logit scoring is invalid. Register a single-token verdict "
        f"pair for {model_name!r} in fsvlm/prompts/verdict.py:_OVERRIDES."
    )
    return pass_ids[0], fail_ids[0]


def resolve_inspection_prompt(template: str, model_name: str | None = None) -> str:
    """Substitute ``$pass_token`` / ``$fail_token`` placeholders in a prompt template.

    Templates without those placeholders are returned unchanged, so user-supplied
    raw prompts pass through cleanly. Use this at every site that passes a prompt
    string to the model so the prompt's instructed verdict matches the verdict
    pair used for training labels and logit scoring.
    """
    if "$pass_token" not in template and "$fail_token" not in template:
        return template
    pass_str, fail_str = verdict_tokens(model_name)
    return Template(template).safe_substitute(pass_token=pass_str, fail_token=fail_str)
