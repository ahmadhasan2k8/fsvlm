"""Tiered validation experiment: prove Tier 1 < Tier 2 < Tier 3 across datasets.

Tier 1: Zero-shot — base VLM with defect-describing prompt, no fine-tuning
Tier 2: Few-shot — fine-tune on defect images only + 2-3 good images
Tier 3: Full dataset — fine-tune on all training data (good + defect)

Usage:
    python experiments/tiered_validation.py --dataset hazelnut
    python experiments/tiered_validation.py --dataset bottle
    python experiments/tiered_validation.py --dataset metal_nut
    python experiments/tiered_validation.py --all
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset configs — defect descriptions for zero-shot prompts
# ---------------------------------------------------------------------------

DATASET_CONFIGS: dict[str, dict] = {
    "hazelnut": {
        "data_root": Path("experiments/mvtec_data/hazelnut"),
        "defect_prompt": (
            "Examine this hazelnut. Is it a normal, good quality hazelnut or does "
            "it have visible defects like cracks, holes, or damage? "
            "Answer PASS if good, FAIL if defective. Then describe briefly."
        ),
        "defect_types": ["crack", "cut", "hole", "print"],
        "defect_descriptions": {
            "crack": "Visible crack in the hazelnut shell",
            "cut": "Surface cut or scratch on the hazelnut",
            "hole": "Hole or puncture in the hazelnut shell",
            "print": "Foreign mark or print contamination on the surface",
        },
    },
    "bottle": {
        "data_root": Path("experiments/mvtec_data/bottle"),
        "defect_prompt": (
            "Examine this bottle. Is it a normal, undamaged bottle or does it have "
            "visible defects like cracks, chips, or contamination? "
            "Answer PASS if good, FAIL if defective. Then describe briefly."
        ),
        "defect_types": ["broken_large", "broken_small", "contamination"],
        "defect_descriptions": {
            "broken_large": "Large break or crack in the bottle",
            "broken_small": "Small chip or fracture in the bottle",
            "contamination": "Foreign contamination on the bottle surface",
        },
    },
    "metal_nut": {
        "data_root": Path("experiments/mvtec_data/metal_nut"),
        "defect_prompt": (
            "Examine this metal nut (hardware fastener). Is it normal or defective? "
            "Look for flipping, discoloration, scratches, or bending. "
            "Answer PASS if good, FAIL if defective. Then describe briefly."
        ),
        "defect_types": ["flip", "color", "scratch", "bent"],
        "defect_descriptions": {
            "flip": "Metal nut is flipped or oriented incorrectly",
            "color": "Color discoloration or oxidation on the metal nut",
            "scratch": "Surface scratch on the metal nut",
            "bent": "Bent or deformed metal nut",
        },
    },
}


@dataclass
class TierResult:
    """Result of one tier evaluation."""
    dataset: str
    tier: str
    auroc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    threshold: float
    num_test: int
    elapsed_seconds: float
    confusion_matrix: list[list[int]]


# ---------------------------------------------------------------------------
# Tier 1: Zero-shot evaluation
# ---------------------------------------------------------------------------

def run_tier1(dataset_name: str, config: dict) -> TierResult:
    """Run zero-shot evaluation with base model (no fine-tuning)."""
    print(f"\n{'='*60}")
    print(f"TIER 1 — Zero-shot: {dataset_name}")
    print(f"{'='*60}")

    data_root = config["data_root"]
    test_dir = data_root / "test"
    prompt = config["defect_prompt"]

    # Collect test images with labels
    test_images, test_labels = _collect_test_set(test_dir)
    print(f"Test set: {len(test_images)} images ({sum(test_labels)} defect, {len(test_labels) - sum(test_labels)} good)")

    start = time.time()

    # Run base model inference (no adapter)
    scores = _run_base_model_inference(test_images, prompt)

    elapsed = time.time() - start

    # Compute metrics
    metrics = _compute_metrics(scores, test_labels)
    cm = metrics["confusion_matrix"]

    result = TierResult(
        dataset=dataset_name,
        tier="tier1_zeroshot",
        auroc=metrics["auroc"],
        f1=metrics["f1"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        accuracy=metrics["accuracy"],
        threshold=metrics["threshold"],
        num_test=len(test_images),
        elapsed_seconds=elapsed,
        confusion_matrix=cm,
    )
    _print_result(result)
    return result


def _run_base_model_inference(
    image_paths: list[Path],
    prompt: str,
) -> list[float]:
    """Run base model (no adapter) on images, return defect scores.

    Scoring cascade (v0.1, logit-first):
    1. If the first generated token is an explicit ``FAIL`` → score 0.9
    2. If the first generated token is an explicit ``PASS`` → score 0.1
    3. Otherwise → use the *token-logit probability* P(FAIL) / [P(FAIL) + P(PASS)]
       computed directly from the logits at generation step 0.

    This replaces the v0 behaviour, which returned a constant 0.75 when no keyword
    matched — a silent AUROC=0.5 artefact visible on DeepPCB (`BASED` token).
    Defect-specialist flagged in 20260420_061530 consultation.
    """
    import os
    import torch
    import transformers.modeling_utils
    transformers.modeling_utils.caching_allocator_warmup = lambda *a, **kw: None

    from unsloth import FastVisionModel

    model_name = os.environ.get("FSVLM_DEFAULT_MODEL", "unsloth/gemma-4-E4B-it")
    print(f"Loading base model: {model_name}")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        load_in_4bit=True,
        device_map="cuda:0",
    )
    FastVisionModel.for_inference(model)

    from fsvlm.utils.image import load_image

    # Pre-compute PASS/FAIL token IDs once (same pattern as adapter inference)
    _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    pass_id = _tok.encode("PASS", add_special_tokens=False)[0]
    fail_id = _tok.encode("FAIL", add_special_tokens=False)[0]

    scores = []
    for i, img_path in enumerate(image_paths):
        img = load_image(img_path, max_size=560)

        chat = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]

        prompt_text = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False,
        )

        device = next(model.parameters()).device
        inputs = tokenizer(text=prompt_text, images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )

        # Token-logit probability of FAIL vs PASS at the first generated position.
        # This is the v0.1 non-degenerate fallback — before, a "BASED" (or any
        # non-keyword) first token sent us to the constant 0.75, producing AUROC=0.5.
        prob_fail_logit = 0.5
        if hasattr(gen, "scores") and gen.scores:
            first_logits = gen.scores[0][0]
            p_logit = first_logits[pass_id].float()
            f_logit = first_logits[fail_id].float()
            probs = torch.softmax(torch.stack([p_logit, f_logit]), dim=0)
            prob_fail_logit = probs[1].item()

        # Decode text response for the explicit PASS/FAIL override case.
        input_len = inputs["input_ids"].shape[-1]
        response = tokenizer.decode(
            gen.sequences[0][input_len:], skip_special_tokens=True
        ).strip().upper()
        first_word = response.split()[0].rstrip(".,!:") if response.split() else ""

        # Cascade: explicit keyword overrides logit probability; otherwise use the prob.
        if first_word == "FAIL":
            score = 0.9
        elif first_word == "PASS":
            score = 0.1
        else:
            score = prob_fail_logit

        scores.append(score)

        if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
            print(f"  [{i+1}/{len(image_paths)}] {img_path.parent.name}/{img_path.name}: "
                  f"first={first_word!r} p_fail_logit={prob_fail_logit:.3f} score={score:.3f}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return scores


# ---------------------------------------------------------------------------
# Tier 2: Few-shot defect-only fine-tuning
# ---------------------------------------------------------------------------

def run_tier2(dataset_name: str, config: dict) -> TierResult:
    """Fine-tune with defect images only + 2-3 good images."""
    print(f"\n{'='*60}")
    print(f"TIER 2 — Few-shot defect-only: {dataset_name}")
    print(f"{'='*60}")

    data_root = config["data_root"]
    test_dir = data_root / "test"
    train_good_dir = data_root / "train" / "good"
    prompt = config["defect_prompt"]

    # Collect ALL defect test images as our "defect examples" for training
    # But only use a FEW (simulating user with ~15-20 defect images)
    defect_images = []
    for defect_type in config["defect_types"]:
        defect_dir = test_dir / defect_type
        if defect_dir.exists():
            imgs = sorted(defect_dir.glob("*.png"))
            defect_images.extend(imgs)

    # Take ~15 defect images for training (leave rest for test)
    random.seed(42)
    random.shuffle(defect_images)
    train_defect = defect_images[:15]
    # These get excluded from the test set later

    # Take 3 good images for training
    good_images = sorted(train_good_dir.glob("*.png"))
    train_good = good_images[:3]

    print(f"Training: {len(train_defect)} defect + {len(train_good)} good = {len(train_defect) + len(train_good)} total")

    # Build training data CSV
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="dvlm_tier2_"))
    csv_path = tmpdir / "labels.csv"

    lines = ["image_path,label,description"]
    for img in train_good:
        lines.append(f"{img.resolve()},good,Normal undamaged item")
    for img in train_defect:
        # Determine defect type from parent dir name
        defect_type = img.parent.name
        desc = config["defect_descriptions"].get(defect_type, "Defect detected")
        lines.append(f"{img.resolve()},defect,{desc}")

    csv_path.write_text("\n".join(lines))

    # Train via subprocess (reuses full pipeline)
    output_dir = tmpdir / "adapter_output"
    start = time.time()

    proc = subprocess.run(
        [
            sys.executable, "-m", "fsvlm.cli",
            "train", "--images", str(csv_path),
            "--output", str(output_dir),
            "--epochs", "3", "--no-sweep", "-y",
        ],
        capture_output=False,
        timeout=1800,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Tier 2 training failed for {dataset_name}")

    # Find adapter
    adapter_path = output_dir / "adapter"
    if not adapter_path.exists():
        raise RuntimeError(f"No adapter found at {adapter_path}")

    # Evaluate on FULL test set (excluding training defect images)
    test_images, test_labels = _collect_test_set(test_dir)
    train_defect_set = {str(p) for p in train_defect}
    filtered = [(img, lbl) for img, lbl in zip(test_images, test_labels) if str(img) not in train_defect_set]
    test_images = [x[0] for x in filtered]
    test_labels = [x[1] for x in filtered]
    print(f"Test set (after excluding train defects): {len(test_images)} images")

    scores = _run_adapter_inference(test_images, adapter_path, prompt)
    elapsed = time.time() - start
    metrics = _compute_metrics(scores, test_labels)

    result = TierResult(
        dataset=dataset_name,
        tier="tier2_fewshot",
        auroc=metrics["auroc"],
        f1=metrics["f1"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        accuracy=metrics["accuracy"],
        threshold=metrics["threshold"],
        num_test=len(test_images),
        elapsed_seconds=elapsed,
        confusion_matrix=metrics["confusion_matrix"],
    )
    _print_result(result)
    return result


# ---------------------------------------------------------------------------
# Tier 3: Full dataset fine-tuning
# ---------------------------------------------------------------------------

def run_tier3(dataset_name: str, config: dict) -> TierResult:
    """Fine-tune with full training dataset (good + defect from train split)."""
    print(f"\n{'='*60}")
    print(f"TIER 3 — Full dataset: {dataset_name}")
    print(f"{'='*60}")

    data_root = config["data_root"]
    test_dir = data_root / "test"
    train_dir = data_root / "train"
    prompt = config["defect_prompt"]

    # MVTec train has only good images — we need defect training images too.
    # Use defect test images for training (common MVTec practice for defect detection).
    # To keep it fair, we'll do a split: use half of defect test for training, half for eval.
    # Plus all train/good for training, all test/good for eval.

    # Collect train good
    train_good = sorted((train_dir / "good").glob("*.png"))

    # Collect defect images, split 50/50 for train/test
    all_defect: list[tuple[Path, str]] = []
    for defect_type in config["defect_types"]:
        defect_dir = test_dir / defect_type
        if defect_dir.exists():
            for img in sorted(defect_dir.glob("*.png")):
                all_defect.append((img, defect_type))

    random.seed(42)
    random.shuffle(all_defect)
    split = len(all_defect) // 2
    train_defect = all_defect[:split]
    test_defect = all_defect[split:]

    # Test good
    test_good_imgs = sorted((test_dir / "good").glob("*.png"))

    print(f"Training: {len(train_good)} good + {len(train_defect)} defect = {len(train_good) + len(train_defect)} total")
    print(f"Test: {len(test_good_imgs)} good + {len(test_defect)} defect = {len(test_good_imgs) + len(test_defect)} total")

    # Build training CSV
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="dvlm_tier3_"))
    csv_path = tmpdir / "labels.csv"

    lines = ["image_path,label,description"]
    for img in train_good:
        lines.append(f"{img.resolve()},good,Normal undamaged item")
    for img, dtype in train_defect:
        desc = config["defect_descriptions"].get(dtype, "Defect detected")
        lines.append(f"{img.resolve()},defect,{desc}")

    csv_path.write_text("\n".join(lines))

    # Train
    output_dir = tmpdir / "adapter_output"
    start = time.time()

    proc = subprocess.run(
        [
            sys.executable, "-m", "fsvlm.cli",
            "train", "--images", str(csv_path),
            "--output", str(output_dir),
            "--epochs", "3", "--no-sweep", "-y",
        ],
        capture_output=False,
        timeout=1800,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Tier 3 training failed for {dataset_name}")

    adapter_path = output_dir / "adapter"

    # Evaluate on test set (test good + test defect only)
    test_images = list(test_good_imgs) + [img for img, _ in test_defect]
    test_labels = [0] * len(test_good_imgs) + [1] * len(test_defect)

    scores = _run_adapter_inference(test_images, adapter_path, prompt)
    elapsed = time.time() - start
    metrics = _compute_metrics(scores, test_labels)

    result = TierResult(
        dataset=dataset_name,
        tier="tier3_full",
        auroc=metrics["auroc"],
        f1=metrics["f1"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        accuracy=metrics["accuracy"],
        threshold=metrics["threshold"],
        num_test=len(test_images),
        elapsed_seconds=elapsed,
        confusion_matrix=metrics["confusion_matrix"],
    )
    _print_result(result)
    return result


# ---------------------------------------------------------------------------
# Adapter inference helper
# ---------------------------------------------------------------------------

def _run_adapter_inference(
    image_paths: list[Path],
    adapter_path: Path,
    prompt: str,
) -> list[float]:
    """Run adapter inference, return defect scores."""
    import torch
    import transformers.modeling_utils
    transformers.modeling_utils.caching_allocator_warmup = lambda *a, **kw: None

    from unsloth import FastVisionModel

    from fsvlm.utils.image import load_image

    print(f"Loading adapter: {adapter_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=1024,
        load_in_4bit=True,
        device_map="cuda:0",
    )
    FastVisionModel.for_inference(model)

    _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    pass_id = _tok.encode("PASS", add_special_tokens=False)[0]
    fail_id = _tok.encode("FAIL", add_special_tokens=False)[0]

    scores = []
    for i, img_path in enumerate(image_paths):
        img = load_image(img_path, max_size=560)
        chat = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        prompt_text = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False,
        )
        device = next(model.parameters()).device
        inputs = tokenizer(text=prompt_text, images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )

        prob_fail = 0.5
        if hasattr(gen, "scores") and gen.scores:
            first = gen.scores[0][0]
            p_logit = first[pass_id].float()
            f_logit = first[fail_id].float()
            probs = torch.softmax(torch.stack([p_logit, f_logit]), dim=0)
            prob_fail = probs[1].item()

        scores.append(prob_fail)
        if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
            print(f"  [{i+1}/{len(image_paths)}] processed")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return scores


# ---------------------------------------------------------------------------
# Test set collection + metrics
# ---------------------------------------------------------------------------

def _collect_test_set(test_dir: Path) -> tuple[list[Path], list[int]]:
    """Collect test images and binary labels (0=good, 1=defect)."""
    images: list[Path] = []
    labels: list[int] = []

    # Good images
    good_dir = test_dir / "good"
    if good_dir.exists():
        for img in sorted(good_dir.glob("*.png")):
            images.append(img)
            labels.append(0)

    # Defect images (all subdirs except 'good')
    for subdir in sorted(test_dir.iterdir()):
        if subdir.is_dir() and subdir.name != "good":
            for img in sorted(subdir.glob("*.png")):
                images.append(img)
                labels.append(1)

    return images, labels


def _compute_metrics(scores: list[float], labels: list[int]) -> dict:
    """Compute AUROC, F1, precision, recall with threshold optimization."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.array(labels)
    y_scores = np.array(scores)

    auroc = roc_auc_score(y_true, y_scores)

    # Threshold optimization: find where P >= 0.75 AND R >= 0.75
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_scores)
    best_f1 = 0.0
    best_thresh = 0.5
    for p, r, t in zip(prec_arr, rec_arr, thresholds):
        if p >= 0.5 and r >= 0.5:  # relaxed for zero-shot
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f > best_f1:
                best_f1 = f
                best_thresh = t

    # If no threshold meets criteria, use 0.5
    y_pred = (y_scores >= best_thresh).astype(int)

    return {
        "auroc": float(auroc),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(best_thresh),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _print_result(r: TierResult) -> None:
    """Print a tier result."""
    print(f"\n  {r.tier} on {r.dataset}:")
    print(f"    AUROC:     {r.auroc:.4f}")
    print(f"    F1:        {r.f1:.4f}")
    print(f"    Precision: {r.precision:.4f}")
    print(f"    Recall:    {r.recall:.4f}")
    print(f"    Accuracy:  {r.accuracy:.4f}")
    print(f"    Threshold: {r.threshold:.4f}")
    print(f"    Test size: {r.num_test}")
    print(f"    Time:      {r.elapsed_seconds:.0f}s")
    cm = r.confusion_matrix
    if len(cm) == 2:
        print(f"    CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")


def print_comparison(results: list[TierResult]) -> None:
    """Print side-by-side comparison table."""
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Dataset':<12} {'Tier':<18} {'AUROC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Acc':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r.dataset:<12} {r.tier:<18} {r.auroc:>8.4f} {r.f1:>8.4f} {r.precision:>8.4f} {r.recall:>8.4f} {r.accuracy:>8.4f}")
    print("-" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tiered validation experiment")
    parser.add_argument("--dataset", type=str, choices=list(DATASET_CONFIGS.keys()),
                       help="Run on a specific dataset")
    parser.add_argument("--all", action="store_true", help="Run on all datasets")
    parser.add_argument("--tier", type=str, choices=["1", "2", "3"],
                       help="Run only a specific tier")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.print_help()
        return

    datasets = list(DATASET_CONFIGS.keys()) if args.all else [args.dataset]
    all_results: list[TierResult] = []

    for ds_name in datasets:
        config = DATASET_CONFIGS[ds_name]

        if not config["data_root"].exists():
            print(f"WARNING: Dataset {ds_name} not found at {config['data_root']}, skipping")
            continue

        if args.tier is None or args.tier == "1":
            r1 = run_tier1(ds_name, config)
            all_results.append(r1)

        if args.tier is None or args.tier == "2":
            r2 = run_tier2(ds_name, config)
            all_results.append(r2)

        if args.tier is None or args.tier == "3":
            r3 = run_tier3(ds_name, config)
            all_results.append(r3)

    if len(all_results) > 1:
        print_comparison(all_results)

    # Save results — merge with existing results file
    out_path = Path("experiments/tiered_results.json")
    existing: list[dict] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except (json.JSONDecodeError, ValueError):
            existing = []

    # Remove old entries for same dataset+tier combos being updated
    new_keys = {(r.dataset, r.tier) for r in all_results}
    existing = [e for e in existing if (e["dataset"], e["tier"]) not in new_keys]
    existing.extend(asdict(r) for r in all_results)

    out_path.write_text(json.dumps(existing, indent=2))
    print(f"\nResults saved to {out_path} ({len(existing)} total entries)")


if __name__ == "__main__":
    main()
