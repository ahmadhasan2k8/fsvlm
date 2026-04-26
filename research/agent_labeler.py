"""Agent labeler — generate per-image natural-language defect descriptions with Gemma 4.

Purpose
-------
For the Pass 2 ablation described in POSITIONING.md § "Research questions": compare training
labels of three quality tiers at matched N=30:

  (1) thin           — just "good" / "defect" (integer-label baseline)
  (2) metadata       — dataset-subtype string, e.g. "crack defect on the hazelnut" (Pass 1)
  (3) agent          — per-image VLM-generated description, e.g. "a dark crack runs diagonally
                       across the upper-right of the hazelnut shell" (this module)

Arm (3) simulates what a user+SAM workflow would produce: every training image carries its own
specific description generated from the image itself, not from its folder name.

Usage
-----
    # Generate and cache descriptions for the training pool of MVTec hazelnut:
    python experiments/agent_labeler.py mvtec hazelnut

    # Generate for all datasets/categories covered by the smoke pass:
    python experiments/agent_labeler.py --all

Cache
-----
Descriptions are cached at ``research/agent_labels.json`` keyed by a SHA-1 hash of the
image path + the model name, so re-runs are free and the cache is stable across sweeps.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / "research" / "agent_labels.json"

# Prompts per label class — tuned to produce one-sentence, location-aware descriptions
# that mirror what a user would type after clicking a SAM mask.
GOOD_PROMPT = (
    "Examine this {category_pretty}. Describe its normal appearance in one concise "
    "sentence (no defects to mention)."
)
DEFECT_PROMPT = (
    "Examine this {category_pretty}. There is a visible defect or anomaly somewhere in "
    "the image. Describe WHAT is wrong and WHERE (top, bottom, edge, center, etc.) in ONE "
    "concise sentence. Do NOT say 'PASS' or 'FAIL' — just describe the defect."
)


@dataclass
class LabelRequest:
    image_path: Path
    label: str         # "good" | "defect"
    category: str      # dataset-native category name (e.g. "hazelnut", "candle")
    dataset: str       # "mvtec" | "visa" | "deeppcb"


@dataclass
class LabelResult:
    image_path: Path
    label: str
    description: str
    model_name: str
    elapsed_ms: float


def _cache_key(image_path: Path, model_name: str) -> str:
    """Stable SHA-1 over absolute path + model. Safe across directory moves if path stays."""
    h = hashlib.sha1()
    h.update(str(image_path.resolve()).encode("utf-8"))
    h.update(b"\x00")
    h.update(model_name.encode("utf-8"))
    return h.hexdigest()


def load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except (json.JSONDecodeError, ValueError):
        return {}


def save_cache(cache: dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def generate_descriptions(
    requests: list[LabelRequest],
    model_name: str = "unsloth/gemma-4-E4B-it",
    max_new_tokens: int = 64,
    max_image_size: int = 560,
    progress_every: int = 20,
) -> list[LabelResult]:
    """Run Gemma 4 on each image to produce a single-sentence defect description.

    Uses the cache file at ``research/agent_labels.json``; only un-cached images hit the
    GPU. Returns LabelResult per request (preserving request order).
    """
    cache = load_cache()

    # Figure out what we actually need to compute
    pending: list[tuple[int, LabelRequest, str]] = []  # (index, request, cache_key)
    results: list[LabelResult | None] = [None] * len(requests)

    for i, req in enumerate(requests):
        key = _cache_key(req.image_path, model_name)
        if key in cache:
            results[i] = LabelResult(
                image_path=req.image_path,
                label=req.label,
                description=cache[key]["description"],
                model_name=cache[key]["model_name"],
                elapsed_ms=cache[key].get("elapsed_ms", 0.0),
            )
        else:
            pending.append((i, req, key))

    if not pending:
        print(f"[agent_labeler] All {len(requests)} descriptions cached — no GPU work needed.")
        return [r for r in results if r is not None]

    print(f"[agent_labeler] Cached: {len(requests) - len(pending)} / {len(requests)}. "
          f"Computing {len(pending)} via {model_name}...")

    # Lazy heavy imports — keep this module cheap to import for tests
    import torch
    import transformers.modeling_utils
    transformers.modeling_utils.caching_allocator_warmup = lambda *a, **kw: None

    from unsloth import FastVisionModel

    from fsvlm.utils.image import load_image

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        load_in_4bit=True,
        device_map="cuda:0",
    )
    FastVisionModel.for_inference(model)

    try:
        for n, (i, req, key) in enumerate(pending):
            pretty = req.category.replace("_", " ")
            prompt_template = DEFECT_PROMPT if req.label == "defect" else GOOD_PROMPT
            prompt = prompt_template.format(category_pretty=pretty)

            img = load_image(req.image_path, max_size=max_image_size)
            chat = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]
            prompt_text = tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False,
            )
            device = next(model.parameters()).device
            inputs = tokenizer(
                text=prompt_text, images=[img], return_tensors="pt", padding=True,
            )
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            t0 = time.time()
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                )
            elapsed_ms = (time.time() - t0) * 1000.0

            # Decode only the generated continuation
            prompt_len = inputs["input_ids"].shape[1]
            new_ids = gen_ids[0][prompt_len:]
            _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
            raw = _tok.decode(new_ids, skip_special_tokens=True).strip()
            description = _post_process(raw)

            results[i] = LabelResult(
                image_path=req.image_path,
                label=req.label,
                description=description,
                model_name=model_name,
                elapsed_ms=elapsed_ms,
            )
            cache[key] = {
                "description": description,
                "model_name": model_name,
                "elapsed_ms": elapsed_ms,
                "image_path": str(req.image_path),
                "label": req.label,
                "category": req.category,
                "dataset": req.dataset,
            }

            if (n + 1) % progress_every == 0 or (n + 1) == len(pending):
                print(f"  [{n+1}/{len(pending)}] {req.dataset}/{req.category}/{req.label} "
                      f"→ {description[:80]}…")
                # Incremental save so a mid-run crash doesn't lose progress
                save_cache(cache)
    finally:
        save_cache(cache)
        del model, tokenizer
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cleanup should never crash
            pass

    return [r for r in results if r is not None]


def _post_process(raw: str) -> str:
    """Clean up the raw model output — strip boilerplate, collapse whitespace, truncate."""
    text = raw.strip()
    # Strip common leading phrases that add no information
    for prefix in (
        "Here is",
        "Here's",
        "This image shows",
        "The image shows",
        "In this image,",
    ):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].lstrip(",: ").strip()
    # One sentence only — cut at the first period if multi-sentence
    if "." in text:
        first_period = text.index(".")
        if first_period < len(text) - 1:
            text = text[: first_period + 1]
    # Reasonable length cap
    return text[:400].strip() or "defect of unknown type visible in the image"


# ---------------------------------------------------------------------------
# CLI — generate descriptions for a specific (dataset, category) pair
# ---------------------------------------------------------------------------


def _requests_for_training_pool(dataset: str, category: str) -> list[LabelRequest]:
    """Reuse the sweep's DatasetAdapter to materialize the training pool."""
    from research.dataset_size_sweep import build_adapter
    adapter = build_adapter(dataset)
    good_pool, defect_pool = adapter.train_pool(category)
    reqs: list[LabelRequest] = []
    for ex in good_pool:
        reqs.append(LabelRequest(ex.image_path, "good", category, dataset))
    for ex in defect_pool:
        reqs.append(LabelRequest(ex.image_path, "defect", category, dataset))
    return reqs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", choices=["mvtec", "visa", "deeppcb"])
    parser.add_argument("category", nargs="?")
    parser.add_argument("--all", action="store_true",
                        help="Generate for every (dataset, category) covered by the current "
                             "smoke-pass configuration in queue.json.")
    parser.add_argument("--model", default="unsloth/gemma-4-E4B-it")
    args = parser.parse_args()

    if args.all:
        smoke_cats = [
            ("mvtec", "hazelnut"),
            ("mvtec", "bottle"),
            ("mvtec", "metal_nut"),
            ("visa", "candle"),
            ("visa", "pcb1"),
            ("visa", "fryum"),
            ("deeppcb", "pcb"),
        ]
        reqs: list[LabelRequest] = []
        for ds, cat in smoke_cats:
            reqs.extend(_requests_for_training_pool(ds, cat))
    else:
        if not args.dataset or not args.category:
            parser.error("provide DATASET CATEGORY, or use --all")
        reqs = _requests_for_training_pool(args.dataset, args.category)

    print(f"[agent_labeler] {len(reqs)} images to describe.")
    results = generate_descriptions(reqs, model_name=args.model)
    print(f"[agent_labeler] Produced {len(results)} descriptions. Cache: {CACHE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
