"""Round-3 reviewer priority-1 + priority-6: bootstrap CI on existing 3-seed N=2 data,
plus filter-threshold robustness curve.

This script answers two attacks from the round-3 review:

A1 (filter cutoff post-hoc): compute pass-rate at thresholds {0.02, 0.05, 0.075, 0.10, 0.15}.
A2 (single-seed N=2 has no CI): compute bootstrap 95% CI from existing 3 seeds. If
    lower-95%-CI ≥ 0.80 across (cat, model) pairs, the headline survives without 2 more seeds.
"""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_per_seed_aurocs() -> dict:
    rows = json.loads((REPO_ROOT / "research" / "dataset_size_results.json").read_text())
    out = defaultdict(lambda: defaultdict(list))  # recipe -> (d,c,n) -> [(seed, auroc)]
    for r in rows:
        rec = r.get("recipe_version")
        if rec not in {"v0.3-tier-a", "v0.5-tier-a-qwen3"}:
            continue
        if r.get("auroc") is None:
            continue
        out[rec][(r["dataset"], r["category"], r["n_samples"])].append(
            (r.get("seed"), r["auroc"])
        )
    return out


def percentile(xs: list[float], q: float) -> float:
    xs_sorted = sorted(xs)
    if not xs_sorted:
        return float("nan")
    k = (len(xs_sorted) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs_sorted[int(k)]
    d0 = xs_sorted[f] * (c - k)
    d1 = xs_sorted[c] * (k - f)
    return d0 + d1


def bootstrap_ratio_ci(
    n2_per_seed: list[float],
    n30_per_seed: list[float],
    zs: float,
    n_boot: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for the saturation ratio = (mean(N2) - ZS) / (mean(N30) - ZS).

    Returns (point_estimate, lower_95, upper_95). NaN values mean degenerate
    (lift_N30 too close to zero or negative).
    """
    rng = random.Random(seed)
    point = (statistics.mean(n2_per_seed) - zs) / (
        statistics.mean(n30_per_seed) - zs
    ) if abs(statistics.mean(n30_per_seed) - zs) > 1e-9 else float("nan")

    samples: list[float] = []
    for _ in range(n_boot):
        s2 = [rng.choice(n2_per_seed) for _ in range(len(n2_per_seed))]
        s30 = [rng.choice(n30_per_seed) for _ in range(len(n30_per_seed))]
        denom = statistics.mean(s30) - zs
        if abs(denom) < 1e-9:
            continue
        samples.append((statistics.mean(s2) - zs) / denom)
    if not samples:
        return point, float("nan"), float("nan")
    return point, percentile(samples, 0.025), percentile(samples, 0.975)


def main() -> None:
    data = load_per_seed_aurocs()

    # Build per (recipe, cat) -> {n: per-seed AUROC list}
    per_cat: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(dict)
    for rec, cells in data.items():
        for (d, c, n), seed_aurocs in cells.items():
            per_cat[(rec, d, c)][n] = [a for _, a in seed_aurocs]

    # ZS is single point per cat (n=0 typically with no seed); accept any n=0 row.
    zs_lookup: dict[tuple[str, str, str], float] = {}
    for (rec, d, c), ns in per_cat.items():
        if 0 in ns and ns[0]:
            zs_lookup[(rec, d, c)] = ns[0][0]

    # ---- A2: bootstrap CI on saturation ratio (existing 3 seeds) ----
    print("=" * 110)
    print("A2 (single-seed CI attack): bootstrap 95% CI on saturation ratio from existing 3 seeds")
    print("=" * 110)
    print(f"{'recipe':<22}{'cat':<22}{'ZS':>7}{'N2 mean':>10}{'N30 mean':>10}"
          f"{'lift_N30':>10}{'ratio':>9}{'CI_low':>9}{'CI_high':>10}{'pass≥.80':>10}{'CI_pass':>10}")
    print("-" * 110)

    rows_out = []
    for (rec, d, c), ns in sorted(per_cat.items()):
        if 0 not in ns or 2 not in ns or 30 not in ns:
            continue
        zs = zs_lookup[(rec, d, c)]
        n2 = ns[2]
        n30 = ns[30]
        lift_n30 = statistics.mean(n30) - zs
        if lift_n30 <= 0.05:
            continue  # qualifying filter
        point, lo, hi = bootstrap_ratio_ci(n2, n30, zs, n_boot=5000)
        passes = point >= 0.80
        ci_passes = lo >= 0.80
        rows_out.append({
            "recipe": rec, "dataset": d, "category": c,
            "zs": zs,
            "n2_per_seed": n2,
            "n30_per_seed": n30,
            "n2_mean": statistics.mean(n2),
            "n30_mean": statistics.mean(n30),
            "lift_n30": lift_n30,
            "ratio_point": point,
            "ratio_ci_low": lo,
            "ratio_ci_high": hi,
            "passes_080_point": passes,
            "passes_080_lower_ci": ci_passes,
        })
        print(f"{rec:<22}{f'{d}/{c}':<22}{zs:>7.3f}{statistics.mean(n2):>10.3f}{statistics.mean(n30):>10.3f}"
              f"{lift_n30:>10.3f}{point:>9.3f}{lo:>9.3f}{hi:>10.3f}{'✓' if passes else '✗':>10}{'✓' if ci_passes else '✗':>10}")

    print("-" * 110)
    by_recipe = defaultdict(list)
    for r in rows_out:
        by_recipe[r["recipe"]].append(r)
    for rec, rs in by_recipe.items():
        n_qual = len(rs)
        n_pass_point = sum(1 for r in rs if r["passes_080_point"])
        n_pass_lower = sum(1 for r in rs if r["passes_080_lower_ci"])
        print(f"\n{rec}: qualifying={n_qual}, point-passes={n_pass_point}, lower-CI-passes={n_pass_lower}")

    # Aggregated
    n_qual_all = len(rows_out)
    n_pass_point_all = sum(1 for r in rows_out if r["passes_080_point"])
    n_pass_lower_all = sum(1 for r in rows_out if r["passes_080_lower_ci"])
    print(f"\nAggregated: {n_qual_all} (cat, model) pairs.")
    print(f"  Point-estimate pass rate: {n_pass_point_all}/{n_qual_all} = {n_pass_point_all/n_qual_all:.1%}")
    print(f"  Lower-95%-CI pass rate:   {n_pass_lower_all}/{n_qual_all} = {n_pass_lower_all/n_qual_all:.1%}")

    # ---- A1: filter-threshold robustness curve ----
    print("\n" + "=" * 110)
    print("A1 (filter cutoff post-hoc): pass rate vs filter threshold")
    print("=" * 110)
    print(f"{'threshold':>10}{'recipe':>26}{'qualifying':>14}"
          f"{'pass_point_080':>18}{'pass_rate_point':>18}{'pass_lower_080':>18}{'pass_rate_lower':>18}")
    print("-" * 110)

    thresholds = [0.02, 0.05, 0.075, 0.10, 0.15]
    threshold_curve = []
    for thr in thresholds:
        for rec in ["v0.3-tier-a", "v0.5-tier-a-qwen3"]:
            qual = [r for r in rows_out if r["recipe"] == rec and r["lift_n30"] > thr]
            n_pass = sum(1 for r in qual if r["passes_080_point"])
            n_pass_low = sum(1 for r in qual if r["passes_080_lower_ci"])
            row = {
                "threshold": thr, "recipe": rec,
                "qualifying": len(qual), "pass_point_080": n_pass, "pass_lower_080": n_pass_low,
                "pass_rate_point": n_pass / len(qual) if qual else float("nan"),
                "pass_rate_lower": n_pass_low / len(qual) if qual else float("nan"),
            }
            threshold_curve.append(row)
            rate = f"{row['pass_rate_point']:.1%}" if qual else "N/A"
            rate_low = f"{row['pass_rate_lower']:.1%}" if qual else "N/A"
            print(f"{thr:>10.3f}{rec:>26}{len(qual):>14}{n_pass:>18}{rate:>18}{n_pass_low:>18}{rate_low:>18}")

    # Save outputs
    out_path = REPO_ROOT / "research" / "baselines" / "saturation_analysis.json"
    out_path.write_text(json.dumps({
        "n_seeds": 3,
        "filter_default_threshold": 0.05,
        "rows": rows_out,
        "threshold_curve": threshold_curve,
    }, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
