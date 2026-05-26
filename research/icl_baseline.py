"""ICL (in-context-learning) baseline — the ablation that answers the sharpest reviewer
critique: 'your fine-tuned N=2 result just shows contrastive matching works; you don't
need fine-tuning.' We measure Gemma 4 in ICL mode at matched N on the SAME test splits
Pass 3 used, then report the delta vs Pass 3's fine-tuned runs.

Per cell:
  - Sample N training examples (balanced good+defect) using the same deterministic
    sampler as the sweep.
  - Build a multi-image chat prompt: reference examples (labeled) + query image.
  - Feed through BASE (un-fine-tuned) Gemma 4 — no adapter, no weight updates.
  - Extract defect score via v0.1 logit-probability of PASS/FAIL at generation step 0.
  - Compute AUROC on the same held-out test split.

Writes rows with recipe_version='v0.4-icl-baseline' to dataset_size_results.json.
Pairs 1:1 with Pass 3 rows via identical (dataset, category, n, seed).

Usage:
    python experiments/icl_baseline.py \
        --categories hazelnut candle pcb \
        --n-shots 2 4 8 \
        --seeds 42 1337 7
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _system_prompt(category: str, pass_str: str, fail_str: str) -> str:
    pretty = category.replace("_", " ")
    return (
        f"You are an industrial quality-inspection assistant examining images of "
        f"{pretty}. Below are labeled reference examples. Then you will be asked to "
        f"classify a new image as {pass_str} (normal) or {fail_str} (defective). "
        f"Begin each answer with the single word {pass_str} or {fail_str}."
    )


def _build_icl_chat(
    reference_examples: list,
    query_image_path: Path,
    category: str,
    pass_str: str,
    fail_str: str,
):
    """Build a multi-image chat with reference examples and a query at the end.

    Gemma 4's processor accepts multiple image placeholders in a single chat turn;
    each {"type": "image"} consumes one image from the images list in order.
    """
    content = []
    content.append({"type": "text",
                    "text": _system_prompt(category, pass_str, fail_str) + "\n\nReference examples:"})
    for ex in reference_examples:
        content.append({"type": "image"})
        ref_label = pass_str if ex.label == "good" else fail_str
        content.append({"type": "text", "text": f"Label: {ref_label}"})
    content.append({"type": "text",
                    "text": f"\nNow classify this image. Answer with {pass_str} or {fail_str}, "
                            "then describe briefly:"})
    content.append({"type": "image"})
    return [{"role": "user", "content": content}]


def run_icl_cell(
    adapter,
    category: str,
    n_shots: int,
    seed: int,
    model,
    tokenizer,
    pass_id: int,
    fail_id: int,
    pass_str: str = "PASS",
    fail_str: str = "FAIL",
) -> dict:
    """Run one (category, n_shots, seed) ICL evaluation. Returns a RunRecord-shaped dict."""
    import torch

    from fsvlm.utils.image import load_image
    from research.dataset_size_sweep import (
        apply_label_source,
        sample_training_set,
    )
    from research.tiered_validation import _compute_metrics

    # Sample the reference set (same deterministic sampler as the sweep)
    good_pool, defect_pool = adapter.train_pool(category)
    good_pool, defect_pool = apply_label_source(good_pool, defect_pool, "metadata")
    references = sample_training_set(good_pool, defect_pool, n_shots, seed)

    test_set = adapter.test_set(category)
    test_labels = [t.label for t in test_set]

    print(f"  ICL: {adapter.name}/{category} N={n_shots} seed={seed}")
    print(f"  References: {sum(1 for r in references if r.label == 'good')} good + "
          f"{sum(1 for r in references if r.label == 'defect')} defect")
    print(f"  Test: {len(test_set)} images ({test_labels.count(0)} good / {test_labels.count(1)} defect)")

    # Pre-load reference images
    ref_images = [load_image(ex.image_path, max_size=560) for ex in references]

    scores = []
    t0 = time.time()
    for i, test_ex in enumerate(test_set):
        query_img = load_image(test_ex.image_path, max_size=560)
        chat = _build_icl_chat(references, test_ex.image_path, category, pass_str, fail_str)

        prompt_text = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False,
        )
        all_images = ref_images + [query_img]
        device = next(model.parameters()).device

        inputs = tokenizer(
            text=prompt_text, images=all_images,
            return_tensors="pt", padding=True,
        )
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=16, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )

        prob_fail = 0.5
        if hasattr(gen, "scores") and gen.scores:
            first_logits = gen.scores[0][0]
            p_logit = first_logits[pass_id].float()
            f_logit = first_logits[fail_id].float()
            probs = torch.softmax(torch.stack([p_logit, f_logit]), dim=0)
            prob_fail = probs[1].item()

        scores.append(prob_fail)
        if (i + 1) % 25 == 0 or i == len(test_set) - 1:
            print(f"    [{i+1}/{len(test_set)}]")

    elapsed = time.time() - t0
    metrics = _compute_metrics(scores, test_labels)

    return {
        "dataset": adapter.name,
        "category": category,
        "n_samples": n_shots,
        "seed": seed,
        "auroc": metrics["auroc"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "accuracy": metrics["accuracy"],
        "threshold": metrics["threshold"],
        "num_test": len(test_set),
        "elapsed_seconds": elapsed,
        "is_zero_shot": False,
        "notes": ["label_source=icl_reference", f"n_shots={n_shots}"],
        "git_hash": os.popen("git rev-parse HEAD").read().strip(),
        "git_short": os.popen("git rev-parse --short HEAD").read().strip(),
        "git_dirty": bool(os.popen("git status --porcelain").read().strip()),
        "recipe_version": "v0.4-icl-baseline",
        "status": "",
        "status_reason": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["mvtec", "visa", "deeppcb"])
    parser.add_argument("--categories", nargs="+",
                        default=["hazelnut", "candle", "pcb"],
                        help="Categories across all --datasets (filtered by adapter).")
    parser.add_argument("--n-shots", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 7])
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "research" / "dataset_size_results.json")
    parser.add_argument("--model", default="unsloth/gemma-4-E4B-it")
    args = parser.parse_args()

    # Lazy import
    import torch
    import transformers.modeling_utils
    transformers.modeling_utils.caching_allocator_warmup = lambda *a, **kw: None

    from unsloth import FastVisionModel

    from research.dataset_size_sweep import build_adapter

    print(f"[icl-baseline] Loading base model: {args.model}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=4096,  # larger for ICL with multiple images
        load_in_4bit=True,
        device_map="cuda:0",
    )
    FastVisionModel.for_inference(model)

    from fsvlm.prompts.verdict import verdict_token_ids, verdict_tokens

    pass_id, fail_id = verdict_token_ids(tokenizer, args.model)
    pass_str, fail_str = verdict_tokens(args.model)

    existing = []
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
        except (json.JSONDecodeError, ValueError):
            existing = []

    def already(dname, cat, n, seed):
        for r in existing:
            if (r.get("dataset") == dname and r.get("category") == cat
                    and r.get("n_samples") == n and r.get("seed") == seed
                    and r.get("recipe_version") == "v0.4-icl-baseline"):
                return True
        return False

    for dname in args.datasets:
        adapter = build_adapter(dname)
        cats = [c for c in adapter.categories() if c in args.categories]
        for cat in cats:
            for n in args.n_shots:
                for seed in args.seeds:
                    if already(dname, cat, n, seed):
                        print(f"  skip (done): {dname}/{cat} N={n} seed={seed}")
                        continue
                    try:
                        record = run_icl_cell(
                            adapter, cat, n, seed, model, tokenizer, pass_id, fail_id,
                            pass_str, fail_str,
                        )
                    except Exception as exc:
                        print(f"  FAILED: {dname}/{cat} N={n} seed={seed}: {exc}")
                        continue
                    existing.append(record)
                    args.output.write_text(json.dumps(existing, indent=2))
                    print(f"    AUROC={record['auroc']:.4f} F1={record['f1']:.4f} "
                          f"elapsed={record['elapsed_seconds']:.0f}s")

    del model, tokenizer
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"[icl-baseline] Complete. Results at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
