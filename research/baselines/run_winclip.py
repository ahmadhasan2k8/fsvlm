"""WinCLIP+ K=2 matched-shot baseline.

For each (dataset, category), this script:

1. Selects the same test split that fsvlm's dataset_size_sweep uses (held-out anomaly
   halves on VisA; full test/ folder on MVTec-AD).
2. Samples K=2 reference normal images from the train pool with a fixed seed.
3. Runs anomalib's WinClip with k_shot=2, few_shot_source=<2 references>.
4. Computes image-level AUROC over the test split.

Output: research/baselines/winclip_k2_results.json (list of per-cat result rows).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.v2 import Compose, Normalize, Resize
from torchvision.transforms.v2.functional import InterpolationMode
from sklearn.metrics import roc_auc_score
from PIL import Image

# WinCLIP transform per anomalib's WinClip.configure_pre_processor (anomalib 2.3.x)
WINCLIP_TRANSFORM = Compose([
    Resize((240, 240), antialias=True, interpolation=InterpolationMode.BICUBIC),
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
              std=(0.26862954, 0.26130258, 0.27577711)),
])


def _load_image_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = TF.pil_to_tensor(img).float() / 255.0  # (3,H,W) in [0,1]
    return WINCLIP_TRANSFORM(t)  # (3,240,240) normalised

REPO_ROOT = Path(__file__).resolve().parents[2]
MVTEC_ROOT = REPO_ROOT / "research" / "mvtec_data"
VISA_ROOT = REPO_ROOT / "research" / "datasets" / "visa"


def _load_visa_csv() -> list[dict]:
    with (VISA_ROOT / "split_csv" / "1cls.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def visa_train_normals(category: str) -> list[Path]:
    rows = _load_visa_csv()
    return [
        VISA_ROOT / r["image"]
        for r in rows
        if r["object"] == category and r["split"] == "train" and r["label"] == "normal"
    ]


def visa_test_split(category: str) -> tuple[list[Path], list[Path]]:
    """Return (test_normals, test_anomalies) matching fsvlm's VisA convention.

    fsvlm uses the held-out half of the test-split anomaly rows (sorted by image
    path); the other half is borrowed for the train defect pool. We replicate that
    so WinCLIP's test set equals fsvlm's test set.
    """
    rows = _load_visa_csv()
    cat_rows = [r for r in rows if r["object"] == category and r["split"] == "test"]
    test_normals = [VISA_ROOT / r["image"] for r in cat_rows if r["label"] == "normal"]
    anomaly = sorted(
        [r for r in cat_rows if r["label"] == "anomaly"], key=lambda r: r["image"]
    )
    held_out = anomaly[len(anomaly) // 2:]
    test_anomalies = [VISA_ROOT / r["image"] for r in held_out]
    return test_normals, test_anomalies


def mvtec_train_normals(category: str) -> list[Path]:
    return sorted((MVTEC_ROOT / category / "train" / "good").glob("*.png"))


def mvtec_test_split(category: str) -> tuple[list[Path], list[Path]]:
    """Match the fsvlm `MVTecAdapter.test_set()` convention exactly.

    fsvlm's adapter holds out the second half of each defect subtype as the test
    pool (the first half is borrowed for the train pool, mirroring the VisA
    convention). Earlier versions of this function returned ALL defects per
    subtype, which made WinCLIP+ test on a ~1.6x larger anomaly pool than fsvlm
    did on the same cats — invalidating the matched-shot comparison on MVTec.
    """
    base = MVTEC_ROOT / category / "test"
    test_normals = sorted((base / "good").glob("*.png"))
    test_anomalies: list[Path] = []
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and sub.name != "good":
            imgs = sorted(sub.glob("*.png"))
            test_anomalies.extend(imgs[len(imgs) // 2:])
    return test_normals, test_anomalies


def select_references(pool: list[Path], k: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    # sort for determinism
    pool_sorted = sorted(pool)
    return rng.sample(pool_sorted, k)


def run_winclip_for_cat(
    dataset: str,
    category: str,
    k: int,
    seed: int,
    device: str = "cuda",
) -> dict:
    """Run WinCLIP+ K-shot on one category and return a result dict."""
    if dataset == "mvtec":
        train_pool = mvtec_train_normals(category)
        test_normals, test_anomalies = mvtec_test_split(category)
    elif dataset == "visa":
        train_pool = visa_train_normals(category)
        test_normals, test_anomalies = visa_test_split(category)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    if len(train_pool) < k:
        raise RuntimeError(f"{dataset}/{category}: only {len(train_pool)} train normals (<K={k})")
    if not test_anomalies or not test_normals:
        raise RuntimeError(f"{dataset}/{category}: empty test split")

    refs = select_references(train_pool, k, seed)

    # We bypass the lightning model's setup hook (broken on anomalib 2.3.2 due to
    # PreProcessor.test_transform not existing on the installed version). Instead,
    # we drive the inner WinClipModel directly with manually transformed images.
    from anomalib.models.image.winclip.torch_model import WinClipModel
    prompt_class = category.replace("_", " ")
    use_cuda = device == "cuda" and torch.cuda.is_available()

    inner = WinClipModel(scales=(2, 3), apply_transform=False)
    if use_cuda:
        inner = inner.cuda()
    ref_tensor = torch.stack([_load_image_tensor(p) for p in refs])
    if use_cuda:
        ref_tensor = ref_tensor.cuda()
    inner.setup(class_name=prompt_class, reference_images=ref_tensor)
    inner.eval()

    try:

        scores: list[float] = []
        labels: list[int] = []
        all_imgs = [(p, 0) for p in test_normals] + [(p, 1) for p in test_anomalies]
        with torch.no_grad():
            for img_path, label in all_imgs:
                tensor = _load_image_tensor(img_path).unsqueeze(0)
                if use_cuda:
                    tensor = tensor.cuda()
                out = inner(tensor)
                # WinClipModel.forward returns InferenceBatch(pred_score, anomaly_map)
                if hasattr(out, "pred_score") and out.pred_score is not None:
                    s = float(out.pred_score.detach().cpu().reshape(-1)[0].item())
                elif isinstance(out, tuple) and len(out) >= 1:
                    s = float(out[0].detach().cpu().reshape(-1)[0].item())
                else:
                    raise RuntimeError(f"WinClipModel forward returned no score: {type(out)}")
                scores.append(s)
                labels.append(label)

        auroc = float(roc_auc_score(labels, scores))
        return {
            "method": "winclip_plus",
            "dataset": dataset,
            "category": category,
            "k_shot": k,
            "seed": seed,
            "auroc": auroc,
            "n_test": len(all_imgs),
            "n_normal": len(test_normals),
            "n_anomaly": len(test_anomalies),
            "reference_images": [str(p.relative_to(REPO_ROOT)) for p in refs],
        }
    finally:
        if use_cuda:
            torch.cuda.empty_cache()


DEFAULT_CATS = [
    # Span of fsvlm Gemma ZS values (low → high) for matched-shot comparison.
    ("mvtec", "pill"),         # G_ZS=0.365, Q_ZS=0.756 — lowest Gemma ZS
    ("visa",  "chewinggum"),   # G_ZS=0.441, Q_ZS=0.540 — both low
    ("visa",  "pcb4"),         # G_ZS=0.510, Q_ZS=0.500 — both low, both saturate
    ("visa",  "macaroni2"),    # G_ZS=0.495, Q_ZS=0.500 — failure case both
    ("mvtec", "capsule"),      # G_ZS=0.471, Q_ZS=0.634 — mid
    ("mvtec", "zipper"),       # G_ZS=0.629, Q_ZS=0.713 — mid
    ("mvtec", "transistor"),   # G_ZS=0.712, Q_ZS=0.575 — high Gemma, mid Qwen
    ("mvtec", "wood"),         # G_ZS=0.985, Q_ZS=0.613 — saturated Gemma
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k-shot", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cats", nargs="*", default=None,
                   help='Pairs of "<dataset>/<category>" to override the default 8.')
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "research" / "baselines" / "winclip_k2_results.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.cats:
        cats = []
        for c in args.cats:
            d, cat = c.split("/", 1)
            cats.append((d, cat))
    else:
        cats = DEFAULT_CATS

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
        except Exception:
            existing = []

    done_keys = {(r["dataset"], r["category"], r["k_shot"], r["seed"]) for r in existing}

    for dataset, cat in cats:
        key = (dataset, cat, args.k_shot, args.seed)
        if key in done_keys:
            print(f"[skip] {dataset}/{cat} k={args.k_shot} seed={args.seed} already done")
            continue
        print(f"[run]  {dataset}/{cat} k={args.k_shot} seed={args.seed}", flush=True)
        t0 = time.time()
        try:
            row = run_winclip_for_cat(dataset, cat, args.k_shot, args.seed, args.device)
            row["elapsed_seconds"] = time.time() - t0
            print(f"       AUROC={row['auroc']:.3f} n_test={row['n_test']} elapsed={row['elapsed_seconds']:.1f}s",
                  flush=True)
            existing.append(row)
            args.output.write_text(json.dumps(existing, indent=2))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[fail] {dataset}/{cat}: {e}", flush=True)

    print(f"\nWrote {len(existing)} rows to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
