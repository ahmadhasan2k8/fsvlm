"""Recompute lift / saturation / matched-shot using consistent logit-only scoring.

Compares:
- v0.3-tier-a ZS (cascade-scored, n=0) — original buggy ZS reference
- v0.7-zs-logit-only ZS (logit-only-scored, n=0) — corrected ZS reference

The trained-path rows (v0.3-tier-a, v0.5-tier-a-qwen3) were already scored with
logit-only — they don't change. Only the ZS reference shifts.

Outputs the impact:
- Δ AUROC at ZS per cat (cascade → logit-only)
- Recomputed lift = N=2 (logit-only) - ZS (logit-only)
- Whether the saturation property survives the corrected baseline
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load() -> list[dict]:
    return json.loads((REPO_ROOT / "research" / "dataset_size_results.json").read_text())


def get_zs(rows: list[dict], recipe: str) -> dict:
    out = {}
    for r in rows:
        if r.get("recipe_version") != recipe or r.get("n_samples") != 0:
            continue
        if r.get("auroc") is None:
            continue
        out[(r["dataset"], r["category"])] = r["auroc"]
    return out


def get_n(rows: list[dict], recipe: str, n: int) -> dict:
    out = defaultdict(list)
    for r in rows:
        if r.get("recipe_version") != recipe or r.get("n_samples") != n:
            continue
        if r.get("auroc") is None:
            continue
        out[(r["dataset"], r["category"])].append(r["auroc"])
    return {k: statistics.mean(v) for k, v in out.items()}


def main() -> None:
    rows = load()

    # Original (buggy) ZS references
    gemma_zs_old = get_zs(rows, "v0.3-tier-a")
    qwen3_zs_old = get_zs(rows, "v0.5-tier-a-qwen3")

    # Corrected (logit-only) ZS references
    gemma_zs_new = get_zs(rows, "v0.7-zs-logit-only")
    qwen3_zs_new = get_zs(rows, "v0.7-zs-logit-only-qwen3")

    # N=2 / N=30 (these were always logit-only on the trained path)
    gemma_n2 = get_n(rows, "v0.3-tier-a", 2)
    gemma_n30 = get_n(rows, "v0.3-tier-a", 30)
    qwen3_n2 = get_n(rows, "v0.5-tier-a-qwen3", 2)
    qwen3_n30 = get_n(rows, "v0.5-tier-a-qwen3", 30)

    print("=" * 120)
    print(f"{'Corrected baseline: cascade-scored ZS vs logit-only ZS, and the impact on lift':^120}")
    print("=" * 120)

    for label, zs_old, zs_new, n2, n30 in [
        ("Gemma 4 E4B-it", gemma_zs_old, gemma_zs_new, gemma_n2, gemma_n30),
        ("Qwen3-VL-8B-Instruct", qwen3_zs_old, qwen3_zs_new, qwen3_n2, qwen3_n30),
    ]:
        print(f"\n=== {label} ===")
        print(f"{'cat':<22}{'ZS_old':>9}{'ZS_new':>9}{'ΔZS':>8}"
              f"{'N=2':>9}{'N=30':>9}{'lift_old':>10}{'lift_new':>10}{'collapsed?':>12}")
        print("-" * 120)
        cats_with_both = sorted(set(zs_old) & set(zs_new) & set(n2))
        if not cats_with_both:
            print("(no cats with both old + new ZS yet — re-run still in flight)")
            continue
        deltas_old = []
        deltas_new = []
        n_collapsed = 0
        for k in cats_with_both:
            zs_o = zs_old[k]
            zs_n = zs_new[k]
            d_zs = zs_n - zs_o
            n2v = n2[k]
            n30v = n30.get(k, n2v)
            lift_old = n2v - zs_o
            lift_new = n2v - zs_n
            collapsed = lift_new < 0.02 and lift_old > 0.10
            if collapsed: n_collapsed += 1
            deltas_old.append(lift_old)
            deltas_new.append(lift_new)
            cat_label = f"{k[0]}/{k[1]}"
            tag = "★ COLLAPSED" if collapsed else ""
            print(f"{cat_label:<22}{zs_o:>9.3f}{zs_n:>9.3f}{d_zs:>+8.3f}"
                  f"{n2v:>9.3f}{n30v:>9.3f}{lift_old:>+10.3f}{lift_new:>+10.3f}{tag:>12}")
        print(f"\nMean lift_old (cascade ZS): {statistics.mean(deltas_old):+.4f}")
        print(f"Mean lift_new (logit-only ZS): {statistics.mean(deltas_new):+.4f}")
        print(f"Cats where 'lift' collapsed to <0.02 with corrected ZS: {n_collapsed}/{len(cats_with_both)}")

    print()


if __name__ == "__main__":
    main()
