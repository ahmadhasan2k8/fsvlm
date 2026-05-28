"""Generate v15-specific figures highlighting the 3-backbone × 4-pattern decomposition,
the Pixtral systematic-weakness sanity check, the FT-vs-ICL comparison, and the
24-category buckets visualization.

Outputs to docs/figures/:
  - fig_v15_4pattern.png        — 4-cat × 3-backbone grid with seed dots
  - fig_v15_pixtral_weakness.png — 8 Pixtral cats vs Gemma baseline
  - fig_v15_ft_vs_icl.png        — 9-cat FT-at-N=2 vs ICL-at-K=2 with CIs
  - fig_v15_24cat_buckets.png    — Gemma 24-cat sorted, color-coded by bucket
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "research" / "dataset_size_results.json"
OUT = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_rows():
    return json.loads(DATA.read_text())


def cells(rows, recipe, dataset=None, category=None):
    out = []
    for r in rows:
        if r.get("recipe_version") != recipe:
            continue
        if dataset and r.get("dataset") != dataset:
            continue
        if category and r.get("category") != category:
            continue
        out.append(r)
    return out


def aurocs(rows, recipe, dataset, category, n=None):
    out = []
    for r in cells(rows, recipe, dataset, category):
        if n is not None and r.get("n_samples") != n:
            continue
        if r.get("auroc") is not None:
            out.append(r["auroc"])
    return out


# ---------------------------------------------------------------------------
# Figure 1: 4-pattern decomposition (3 backbones × 4 cats, seed dots overlaid)
# ---------------------------------------------------------------------------
def fig_4pattern(rows):
    cats = [
        ("pipe_fryum", "Gemma-specific weakness"),
        ("pcb1", "Backbone-stability gradient"),
        ("macaroni2", "Data-fundamental"),
        ("capsules", "Convergent mid-recovery"),
    ]
    backbones = [
        ("Gemma 4 E4B-it", "v0.4-longepoch-validation", "#1f77b4"),
        ("Qwen3-VL-8B-Instruct", "v0.4-longepoch-validation-qwen3", "#2ca02c"),
        ("Pixtral-12B-2409", "v0.4-longepoch-validation-pixtral", "#d62728"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    for ax, (cat, pattern) in zip(axes.flat, cats):
        means = []
        sds = []
        all_seeds = []
        labels = []
        colors = []
        for bb_name, recipe, color in backbones:
            seeds = aurocs(rows, recipe, "visa", cat, n=10)
            if not seeds:
                continue
            means.append(statistics.mean(seeds))
            sds.append(statistics.stdev(seeds) if len(seeds) > 1 else 0.0)
            all_seeds.append(seeds)
            labels.append(bb_name.split("-")[0].split(" ")[0])  # short name
            colors.append(color)
        x = np.arange(len(means))
        ax.bar(x, means, yerr=sds, capsize=4, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
        # Overlay seed dots
        for i, seeds in enumerate(all_seeds):
            jitter = np.random.RandomState(i).uniform(-0.08, 0.08, len(seeds))
            ax.scatter(np.full(len(seeds), i) + jitter, seeds,
                       color="black", s=18, zorder=10, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0.30, 1.00)
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_ylabel("AUROC")
        ax.set_title(f"visa/{cat} — {pattern}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("4-pattern decomposition of the catastrophic Gemma cluster\n"
                 "(50ep × N=10 × 5 seeds; bars = mean ± std; dots = individual seeds)",
                 fontsize=11)
    out = OUT / "fig_v15_4pattern.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  → {out.name}")


# ---------------------------------------------------------------------------
# Figure 2: Pixtral systematic-weakness check (8 cats Pixtral vs Gemma ZS)
# ---------------------------------------------------------------------------
def fig_pixtral_weakness(rows):
    # Pull all 27 Pixtral ZS cats + matching Gemma ZS (v0.7-zs-logit-only)
    pix_rows = [r for r in rows if r.get("recipe_version") == "v0.7-zs-logit-only-pixtral"]
    gem_rows = [r for r in rows if r.get("recipe_version") == "v0.7-zs-logit-only" and r.get("n_samples") == 0]
    gem_by_cat = {(r.get("dataset"), r.get("category")): r.get("auroc") for r in gem_rows}

    # Build list sorted by Gemma ZS desc for visual progression
    data = []
    for r in pix_rows:
        key = (r.get("dataset"), r.get("category"))
        if key in gem_by_cat:
            data.append((f"{key[0]}/{key[1]}", gem_by_cat[key], r.get("auroc")))
    data.sort(key=lambda x: -x[1])  # by Gemma ZS, descending

    labels = [d[0] for d in data]
    gemma_vals = [d[1] for d in data]
    pixtral_vals = [d[2] for d in data]
    deltas = np.array(pixtral_vals) - np.array(gemma_vals)

    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w/2, gemma_vals, w, label="Gemma 4 E4B-it ZS", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, pixtral_vals, w, label="Pixtral-12B-2409 ZS", color="#d62728", alpha=0.85)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, alpha=0.7, label="random AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Zero-shot AUROC")
    n_pix_below_random = sum(1 for v in pixtral_vals if v < 0.5)
    ax.set_title(
        f"Pixtral zero-shot vs Gemma zero-shot across all 27 MVTec AD + VisA cats — "
        f"mean Pixtral ZS = {np.mean(pixtral_vals):.3f}, mean Gemma ZS = {np.mean(gemma_vals):.3f}, "
        f"mean Δ = {np.mean(deltas):+.3f}; Pixtral below random on {n_pix_below_random}/27.",
        fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    out = OUT / "fig_v15_pixtral_weakness.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  → {out.name} (27-cat regen)")


# ---------------------------------------------------------------------------
# Figure 3: FT-at-N=2 vs ICL-at-K=2 across 9 cats
# ---------------------------------------------------------------------------
def fig_ft_vs_icl(rows):
    # ICL: v0.4-icl-baseline at K=2 across 9 cats (3 seeds each)
    # FT N=2: best matching FT recipe (v0.3-curve / v0.3-tier-a / v0.8-fixed-pipeline)
    ft_recipes_priority = ["v0.4-longepoch-validation",
                            "v0.8-fixed-pipeline",
                            "v0.3-tier-a",
                            "v0.3-curve"]

    icl_cats = sorted(set(
        (r.get("dataset"), r.get("category"))
        for r in rows
        if r.get("recipe_version") == "v0.4-icl-baseline" and r.get("n_samples") == 2
    ))

    data = []  # (label, icl_mean, icl_sd, ft_mean, ft_sd)
    for ds, c in icl_cats:
        icl_vals = aurocs(rows, "v0.4-icl-baseline", ds, c, n=2)
        if len(icl_vals) < 2:
            continue
        ft_vals = []
        for recipe in ft_recipes_priority:
            ft_vals = aurocs(rows, recipe, ds, c, n=2)
            if ft_vals:
                break
        if not ft_vals:
            continue
        data.append((
            f"{ds}/{c}",
            statistics.mean(icl_vals),
            statistics.stdev(icl_vals) if len(icl_vals) > 1 else 0.0,
            statistics.mean(ft_vals),
            statistics.stdev(ft_vals) if len(ft_vals) > 1 else 0.0,
        ))

    if not data:
        print("  (no FT/ICL overlap data found, skipping fig_v15_ft_vs_icl)")
        return

    # Sort by ICL mean for visual progression
    data.sort(key=lambda x: x[1])
    labels = [d[0] for d in data]
    icl_means = [d[1] for d in data]
    icl_sds = [d[2] for d in data]
    ft_means = [d[3] for d in data]
    ft_sds = [d[4] for d in data]
    deltas = [ft - icl for icl, ft in zip(icl_means, ft_means)]

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w/2, icl_means, w, yerr=icl_sds, capsize=4,
           label=f"ICL K=2 (3 seeds)", color="#ff7f0e", alpha=0.85)
    ax.bar(x + w/2, ft_means, w, yerr=ft_sds, capsize=4,
           label=f"FT N=2 (3 seeds)", color="#1f77b4", alpha=0.85)
    # Annotate delta on top of FT bar
    for i, d in enumerate(deltas):
        sign = "+" if d > 0 else ""
        color = "darkgreen" if d > 0.02 else ("darkred" if d < -0.02 else "grey")
        y = max(icl_means[i], ft_means[i]) + 0.04
        ax.text(x[i] + w/2, y, f"Δ{sign}{d:.02f}", ha="center", fontsize=8, color=color)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.4, 1.10)
    ax.set_ylabel("AUROC")
    mean_delta = np.mean(deltas)
    n_ft_wins = sum(1 for d in deltas if d > 0.02)
    n_icl_wins = sum(1 for d in deltas if d < -0.02)
    ax.set_title(f"FT-at-N=2 vs ICL-at-K=2 on Gemma 4 E4B-it — FT wins {n_ft_wins}/{len(data)}, "
                 f"ICL wins {n_icl_wins}/{len(data)}, mean Δ = {mean_delta:+.03f}",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    out = OUT / "fig_v15_ft_vs_icl.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  → {out.name}")


# ---------------------------------------------------------------------------
# Figure 4: 24-cat Gemma 50ep×N=10×5s sorted, color-coded by bucket
# ---------------------------------------------------------------------------
def fig_24cat_buckets(rows):
    by_cat = defaultdict(list)
    for r in rows:
        if r.get("recipe_version") != "v0.4-longepoch-validation":
            continue
        if r.get("n_samples") != 10:
            continue
        ds, c = r.get("dataset"), r.get("category")
        by_cat[(ds, c)].append(r.get("auroc"))
    rows_data = []
    for (ds, c), aurs in by_cat.items():
        if len(aurs) < 1:
            continue
        m = statistics.mean(aurs)
        sd = statistics.stdev(aurs) if len(aurs) > 1 else 0.0
        rows_data.append((f"{ds}/{c}", m, sd, len(aurs)))
    if not rows_data:
        print("  (no 24-cat Gemma 50ep×N=10 rows found, skipping fig_v15_24cat_buckets)")
        return
    rows_data.sort(key=lambda x: x[1], reverse=True)

    def bucket_color(m):
        if m >= 0.95: return "#2ca02c"  # green: saturated
        if m >= 0.85: return "#9467bd"  # purple: strong
        if m >= 0.70: return "#1f77b4"  # blue: mid
        if m >= 0.65: return "#ff7f0e"  # orange: borderline
        return "#d62728"                  # red: catastrophic

    labels = [d[0] for d in rows_data]
    means = [d[1] for d in rows_data]
    sds = [d[2] for d in rows_data]
    colors = [bucket_color(m) for m in means]

    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sds, capsize=2, color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.axhline(0.65, color="black", linestyle="--", linewidth=0.5, alpha=0.6,
               label="catastrophic threshold (0.65)")
    ax.axhline(0.95, color="black", linestyle=":", linewidth=0.5, alpha=0.6,
               label="saturation threshold (0.95)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("AUROC mean (50ep × N=10 × 5 seeds)")
    n_sat = sum(1 for m in means if m >= 0.95)
    n_strong = sum(1 for m in means if 0.85 <= m < 0.95)
    n_mid = sum(1 for m in means if 0.70 <= m < 0.85)
    n_borderline = sum(1 for m in means if 0.65 <= m < 0.70)
    n_cat = sum(1 for m in means if m < 0.65)
    ax.set_title("Gemma 4 E4B-it: 50ep × N=10 × 5 seeds across 24 MVTec AD + VisA cats — "
                 f"{n_sat} saturated, {n_strong} strong, {n_mid} mid, {n_borderline} borderline, {n_cat} catastrophic",
                 fontsize=10)
    # Bucket legend
    handles = [
        plt.Rectangle((0,0),1,1, color="#2ca02c", label=f"saturated ≥0.95 (n={n_sat})"),
        plt.Rectangle((0,0),1,1, color="#9467bd", label=f"strong 0.85–0.95 (n={n_strong})"),
        plt.Rectangle((0,0),1,1, color="#1f77b4", label=f"mid 0.70–0.85 (n={n_mid})"),
        plt.Rectangle((0,0),1,1, color="#ff7f0e", label=f"borderline 0.65–0.70 (n={n_borderline})"),
        plt.Rectangle((0,0),1,1, color="#d62728", label=f"catastrophic <0.65 (n={n_cat})"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, ncol=1, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)
    out = OUT / "fig_v15_24cat_buckets.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  → {out.name}")


# ---------------------------------------------------------------------------
# Figure 5 (regenerated): pcb1 rank reversal — now 3-backbone with Pixtral added
# ---------------------------------------------------------------------------
def fig_pcb1_rank_reversal_v15(rows):
    # ZS pcb1 (visa): Gemma 0.750, Qwen3 0.546, Pixtral 0.590
    # FT pcb1: Gemma 50ep×N=10×5s, Qwen3 50ep×N=10×5s, Pixtral 50ep×N=10×5s
    def get_ft(recipe):
        vals = aurocs(rows, recipe, "visa", "pcb1", n=10)
        if not vals:
            return None, None
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    gem_zs = 0.750
    qw_zs = 0.546
    px_zs = 0.590
    gem_ft, gem_sd = get_ft("v0.4-longepoch-validation")
    qw_ft, qw_sd = get_ft("v0.4-longepoch-validation-qwen3")
    px_ft, px_sd = get_ft("v0.4-longepoch-validation-pixtral")

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    x = np.arange(3)
    w = 0.32
    ax.bar(x - w/2, [gem_zs, qw_zs, px_zs], w,
           label="Zero-shot",
           color="#bdd7e7", edgecolor="black", linewidth=0.6)
    ax.bar(x + w/2, [gem_ft, qw_ft, px_ft], w,
           yerr=[gem_sd, qw_sd, px_sd], capsize=4,
           label="50ep × $N$=10 × 5 seeds",
           color="#3182bd", edgecolor="black", linewidth=0.6,
           error_kw={"ecolor": "black"})
    for i, (zs, ft) in enumerate(zip([gem_zs, qw_zs, px_zs], [gem_ft, qw_ft, px_ft])):
        ax.annotate(f"Δ {ft-zs:+.03f}", (i, max(zs, ft) + 0.04),
                    ha="center", fontsize=9,
                    color="darkgreen" if ft > zs else "darkred")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Gemma 4 E4B-it", "Qwen3-VL-8B-Instruct", "Pixtral-12B-2409"], fontsize=9)
    ax.set_ylim(0.40, 1.00)
    ax.set_ylabel("AUROC")
    ax.set_title("visa/pcb1 rank reversal — 3-backbone comparison\n"
                 "Gemma's lift is small; Qwen3 reliably recovers; Pixtral recovers unstably (σ=0.080)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    out = OUT / "fig_pcb1_rank_reversal.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  → {out.name} (3-backbone regen)")


# ---------------------------------------------------------------------------
# Figure 6 (regenerated): cross-family backbone — 3-backbone with Pixtral
# Pixtral coverage is partial; we show Pixtral data where it exists
# ---------------------------------------------------------------------------
def fig_cross_family_backbone_v15(rows):
    # Cats in original cross-family table; add Pixtral where we have it.
    # Pixtral has: pcb1 (ZS+FT 5s), capsules (ZS+FT 5s), transistor (ZS only),
    # chewinggum (ZS only), wood (ZS only).
    # Pixtral does NOT have: capsule (mvtec, FT only), bottle (FT)
    cats = [
        ("capsule",    "mvtec", "capsule",    0.859, 0.872, 0.968, 0.984, None, None),
        ("transistor", "mvtec", "transistor", 0.712, 0.757, 0.669, 0.879, 0.586, None),  # px ZS only
        ("bottle",     "mvtec", "bottle",     0.715, 0.789, 0.850, 0.930, None, None),
        ("capsules",   "visa",  "capsules",   0.633, 0.636, 0.663, 0.753, 0.395, None),  # px ZS+FT, FT computed
        ("pcb1",       "visa",  "pcb1",       0.750, 0.785, 0.546, 0.943, 0.590, None),  # px ZS+FT
        ("chewinggum", "visa",  "chewinggum", 0.978, 0.999, 0.981, 0.989, 0.436, None),
        ("wood",       "mvtec", "wood",       0.995, 0.995, 1.000, 1.000, 0.633, None),
    ]
    # Compute Pixtral FT where available
    pix_ft = {}
    for c in ["pcb1", "capsules"]:
        vals = aurocs(rows, "v0.4-longepoch-validation-pixtral", "visa", c, n=10)
        if vals:
            pix_ft[c] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)

    cat_labels = [c[0] for c in cats]
    n = len(cats)
    x = np.arange(n)
    width = 0.13

    zs_gem = [c[3] for c in cats]
    ft_gem = [c[4] for c in cats]
    zs_qw = [c[5] for c in cats]
    ft_qw = [c[6] for c in cats]
    zs_px = [c[7] for c in cats]  # may have None
    ft_px = []
    px_sd = []
    for c in cats:
        cat_name = c[2]
        if cat_name in pix_ft:
            ft_px.append(pix_ft[cat_name][0])
            px_sd.append(pix_ft[cat_name][1])
        else:
            ft_px.append(0)
            px_sd.append(0)

    fig, ax = plt.subplots(figsize=(12.5, 5.5), constrained_layout=True)

    # Gemma ZS + FT
    ax.bar(x - 2.5*width, zs_gem, width, label="Gemma ZS",
           color="#bdd7e7", edgecolor="black", linewidth=0.4)
    ax.bar(x - 1.5*width, ft_gem, width, label="Gemma 50ep×N=10",
           color="#3182bd", edgecolor="black", linewidth=0.4)
    # Qwen3 ZS + FT
    ax.bar(x - 0.5*width, zs_qw, width, label="Qwen3 ZS",
           color="#c7e9c0", edgecolor="black", linewidth=0.4)
    ax.bar(x + 0.5*width, ft_qw, width, label="Qwen3 50ep×N=10",
           color="#31a354", edgecolor="black", linewidth=0.4)
    # Pixtral ZS + FT (only where data exists)
    zs_px_plot = [v if v is not None else 0 for v in zs_px]
    ax.bar(x + 1.5*width, zs_px_plot, width, label="Pixtral ZS",
           color="#fcae91", edgecolor="black", linewidth=0.4)
    ax.bar(x + 2.5*width, ft_px, width, yerr=px_sd, capsize=2,
           label="Pixtral 50ep×N=10 (where covered)",
           color="#de2d26", edgecolor="black", linewidth=0.4,
           error_kw={"ecolor": "black"})
    # Mark missing Pixtral with '—'
    for i, (zs, ft) in enumerate(zip(zs_px, ft_px)):
        if zs is None:
            ax.text(x[i] + 1.5*width, 0.05, "—", ha="center", fontsize=10, color="grey")
        if ft == 0 and cats[i][2] not in pix_ft:
            ax.text(x[i] + 2.5*width, 0.05, "—", ha="center", fontsize=10, color="grey")

    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("AUROC")
    ax.set_title("Cross-family backbone characterization — Gemma + Qwen3 fully covered; "
                 "Pixtral targeted to 4 catastrophic cats (FT) and 4 saturated cats (ZS).\n"
                 "Pixtral ZS is uniformly below Gemma on the 5 cats with overlap.",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)
    out = OUT / "fig_cross_family_backbone.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  → {out.name} (3-backbone regen)")


if __name__ == "__main__":
    rows = load_rows()
    print(f"Loaded {len(rows)} rows from {DATA}")
    print("Generating v15 figures:")
    fig_24cat_buckets(rows)
    fig_4pattern(rows)
    fig_pixtral_weakness(rows)
    fig_ft_vs_icl(rows)
    print("Regenerating existing figures with 3-backbone data:")
    fig_pcb1_rank_reversal_v15(rows)
    fig_cross_family_backbone_v15(rows)
    print("Done.")


# ---------------------------------------------------------------------------
# Figure 9: Substitutability point — overlay (N=10, 50ep) on (N, 3ep) curves
# ---------------------------------------------------------------------------
def fig_substitutability_point(rows):
    """For cats with both an N-shot curve at 3 epochs and a 50ep × N=10 × 5-seed cell,
    plot the curve and mark the 50ep point separately. Visual evidence for the
    substitutability claim: if the 50ep point lands at or above the (N=full, 3ep)
    curve endpoint, the substitution holds for that cat.
    """
    # 7 cats with both kinds of data; deep-curve cats first
    cats = [
        ("mvtec", "capsule"),
        ("mvtec", "transistor"),
        ("mvtec", "wood"),
        ("visa", "chewinggum"),
        ("mvtec", "bottle"),
        ("visa", "capsules"),
        ("visa", "pcb1"),
    ]
    nshot_recipes = {"v0.8-fixed-pipeline", "v0.8-fixed-pipeline-full",
                     "v0.8-fixed-pipeline-multiseed", "v0.8-bisection", "v0.3-curve",
                     "v0.3-tier-a"}

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    axes_flat = axes.flat
    for ax, (ds, c) in zip(axes_flat, cats):
        # 3ep N-shot curve points
        ns_to_aurocs = defaultdict(list)
        for r in rows:
            if (r.get("recipe_version") in nshot_recipes
                and r.get("dataset") == ds and r.get("category") == c
                and r.get("auroc") is not None):
                ns_to_aurocs[r.get("n_samples")].append(r["auroc"])
        if not ns_to_aurocs:
            ax.set_visible(False)
            continue
        # Plot the curve at the N values where we have data; N=0 is ZS, N=-1 is N=full
        ns_sorted = sorted(n for n in ns_to_aurocs if n is not None and n != -1)
        xs = []
        ys = []
        yerr = []
        for n in ns_sorted:
            if n == 0:
                continue  # ZS handled separately
            aurs = ns_to_aurocs[n]
            xs.append(n)
            ys.append(statistics.mean(aurs))
            yerr.append(statistics.stdev(aurs) if len(aurs) > 1 else 0.0)
        ax.errorbar(xs, ys, yerr=yerr, marker="o", color="#1f77b4",
                    label="3-epoch N-shot curve", capsize=3, lw=1.5)
        # ZS reference line
        if 0 in ns_to_aurocs:
            zs = statistics.mean(ns_to_aurocs[0])
            ax.axhline(zs, color="grey", linestyle=":", lw=0.8, alpha=0.7,
                       label=f"ZS = {zs:.3f}")
        # N=full reference line
        if -1 in ns_to_aurocs:
            nf = statistics.mean(ns_to_aurocs[-1])
            ax.axhline(nf, color="#2ca02c", linestyle="--", lw=1, alpha=0.7,
                       label=f"N=full × 3ep = {nf:.3f}")
        # 50ep × N=10 multi-seed point
        ep50 = [r["auroc"] for r in rows
                if r.get("recipe_version") == "v0.4-longepoch-validation"
                and r.get("dataset") == ds and r.get("category") == c
                and r.get("n_samples") == 10 and r.get("auroc") is not None]
        if ep50:
            m = statistics.mean(ep50)
            sd = statistics.stdev(ep50) if len(ep50) > 1 else 0.0
            ax.errorbar([10], [m], yerr=[sd], marker="*", markersize=18,
                        color="#d62728", capsize=5, lw=2,
                        label=f"50ep × N=10 ({len(ep50)}s) = {m:.3f} ± {sd:.3f}")
        ax.set_xscale("log")
        ax.set_xlabel("N (training pool size, log scale)")
        ax.set_ylabel("AUROC")
        ax.set_title(f"{ds}/{c}", fontsize=10)
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.45, 1.02)
    # Hide the unused 8th panel
    if len(cats) < len(list(axes.flat)):
        axes.flat[-1].set_visible(False)
    fig.suptitle("Substitutability: the (N=10, 50ep) point (red ★) vs the 3-epoch N-shot curve (blue) and N=full reference (green dashed).\n"
                 "If the red star lands at or above the green line, 50ep × N=10 substitutes for N=full × 3ep on that cat.",
                 fontsize=10)
    out = OUT / "fig_v15_substitutability.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  → {out.name}")


if __name__ == "__main__":
    # Also call the new figure
    rows_for_sub = load_rows()
    fig_substitutability_point(rows_for_sub)


# ---------------------------------------------------------------------------
# Figure 10: 2D N × epochs heatmap for capsule (substitutability surface)
# ---------------------------------------------------------------------------
def fig_2d_substitutability_capsule(rows):
    """Heatmap of mean AUROC across (N, epochs) for capsule. Includes the
    existing 3ep × N=full × 5-seed reference value as an annotation.
    """
    import matplotlib.colors as mcolors
    sweep = [r for r in rows if r.get('recipe_version','').startswith('v0.5-2d-sweep')
             and r.get('category') == 'capsule']
    # Grid: epochs (3, 25, 50) × N (5, 10, 20, 40)
    epochs = [3, 25, 50]
    ns = [5, 10, 20, 40]
    grid = np.full((len(epochs), len(ns)), np.nan)
    sd_grid = np.full_like(grid, np.nan)
    for i, ep in enumerate(epochs):
        ep_tag = f"{ep}ep"
        for j, n in enumerate(ns):
            cells = [r['auroc'] for r in sweep
                     if r.get('recipe_version','').endswith(ep_tag)
                     and r.get('n_samples') == n
                     and r.get('auroc') is not None]
            if cells:
                grid[i, j] = statistics.mean(cells)
                if len(cells) > 1:
                    sd_grid[i, j] = statistics.stdev(cells)

    # Pull the reference N=full × 3ep value from the existing data
    ref_aurs = [r['auroc'] for r in rows
                if r.get('category') == 'capsule' and r.get('dataset') == 'mvtec'
                and r.get('n_samples') == -1
                and r.get('recipe_version') == 'v0.8-fixed-pipeline']
    nfull_3ep = statistics.mean(ref_aurs) if ref_aurs else None
    nfull_3ep_sd = statistics.stdev(ref_aurs) if len(ref_aurs) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    norm = mcolors.Normalize(vmin=0.80, vmax=0.92)
    im = ax.imshow(grid, cmap='viridis', norm=norm, aspect='auto', origin='lower')
    # Annotate cells with mean ± sd
    for i in range(len(epochs)):
        for j in range(len(ns)):
            if not np.isnan(grid[i, j]):
                txt = f"{grid[i, j]:.3f}"
                if not np.isnan(sd_grid[i, j]):
                    txt += f"\n± {sd_grid[i, j]:.3f}"
                ax.text(j, i, txt, ha='center', va='center',
                        color='white' if grid[i, j] < 0.86 else 'black',
                        fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_yticks(range(len(epochs)))
    ax.set_yticklabels([f"{ep}ep" for ep in epochs])
    ax.set_xlabel("Training-pool size N")
    ax.set_ylabel("Epoch budget")
    title = "Substitutability surface for mvtec/capsule"
    if nfull_3ep is not None:
        title += f"\n(reference: 3ep × N=full × 5 seeds = {nfull_3ep:.3f} ± {nfull_3ep_sd:.3f})"
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, label="AUROC mean (3 seeds)")
    if nfull_3ep is not None:
        cbar.ax.axhline(nfull_3ep, color='red', linestyle='--', linewidth=1.5)
        cbar.ax.text(1.3, nfull_3ep, f"  N=full×3ep ref ({nfull_3ep:.3f})",
                     transform=cbar.ax.get_yaxis_transform(), va='center', fontsize=8, color='red')
    out = OUT / "fig_v15_2d_capsule.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  → {out.name}")


def fig_2d_substitutability_4cat(rows):
    """2x2 panel of (N, epochs) AUROC surfaces for capsule, transistor, bottle, pcb1.

    Each panel: heatmap of mean AUROC across N x epochs grid with iso-AUROC contour
    overlay (interpolated) and per-cell mean +/- std annotation. The Phase 1
    3ep x N=full x 5-seed reference is plotted as a red dashed line on each panel's
    colorbar (best-eval-loss policy; 2D sweep is last-epoch -- not directly
    cross-comparable but informative for shape).
    """
    import matplotlib.colors as mcolors
    from scipy.interpolate import RectBivariateSpline

    cats = [
        ('capsule',    'mvtec', 'capsule (mvtec)',     (0.80, 0.92)),
        ('transistor', 'mvtec', 'transistor (mvtec)',  (0.65, 0.78)),
        ('bottle',     'mvtec', 'bottle (mvtec)',      (0.70, 0.88)),
        ('pcb1',       'visa',  'pcb1 (visa)',         (0.55, 0.78)),
    ]
    epochs = [3, 25, 50]
    ns = [5, 10, 20, 40]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle("Compute-data substitution surface (4 cats x 4 N x 3 epochs x 3 seeds, last-epoch policy)",
                 fontsize=12)

    for k, (cat, ds, label, vrange) in enumerate(cats):
        ax = axes[k // 2, k % 2]
        sweep = [r for r in rows if r.get('recipe_version','').startswith('v0.5-2d-sweep')
                 and r.get('category') == cat]
        grid = np.full((len(epochs), len(ns)), np.nan)
        sd_grid = np.full_like(grid, np.nan)
        for i, ep in enumerate(epochs):
            ep_tag = f"{ep}ep"
            for j, n in enumerate(ns):
                cells = [r['auroc'] for r in sweep
                         if r.get('recipe_version','').endswith(ep_tag)
                         and r.get('n_samples') == n
                         and r.get('auroc') is not None]
                if cells:
                    grid[i, j] = statistics.mean(cells)
                    if len(cells) > 1:
                        sd_grid[i, j] = statistics.stdev(cells)

        # Phase 1 N=full x 3ep reference
        ref_aurs = [r['auroc'] for r in rows
                    if r.get('category') == cat and r.get('dataset') == ds
                    and r.get('n_samples') == -1
                    and r.get('recipe_version') == 'v0.8-fixed-pipeline']
        nfull_3ep = statistics.mean(ref_aurs) if ref_aurs else None

        norm = mcolors.Normalize(vmin=vrange[0], vmax=vrange[1])
        im = ax.imshow(grid, cmap='viridis', norm=norm, aspect='auto', origin='lower')

        # Iso-AUROC contour overlay (interpolated to a finer grid for smoothness)
        x_orig = np.arange(len(ns))
        y_orig = np.arange(len(epochs))
        if not np.any(np.isnan(grid)):
            spline = RectBivariateSpline(y_orig, x_orig, grid, kx=min(2, len(epochs)-1), ky=min(3, len(ns)-1))
            x_fine = np.linspace(0, len(ns)-1, 60)
            y_fine = np.linspace(0, len(epochs)-1, 40)
            Z = spline(y_fine, x_fine)
            n_levels = 5
            levels = np.linspace(vrange[0]+0.01, vrange[1]-0.01, n_levels)
            cs = ax.contour(x_fine, y_fine, Z, levels=levels,
                            colors='white', linewidths=0.8, alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f', colors='white')

        # Cell annotations (mean +/- std)
        for i in range(len(epochs)):
            for j in range(len(ns)):
                if not np.isnan(grid[i, j]):
                    txt = f"{grid[i, j]:.3f}"
                    if not np.isnan(sd_grid[i, j]):
                        txt += f"\n+/-{sd_grid[i, j]:.3f}"
                    mid = (vrange[0] + vrange[1]) / 2
                    ax.text(j, i, txt, ha='center', va='center',
                            color='white' if grid[i, j] < mid else 'black',
                            fontsize=8, fontweight='bold')

        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f"N={n}" for n in ns], fontsize=9)
        ax.set_yticks(range(len(epochs)))
        ax.set_yticklabels([f"{ep}ep" for ep in epochs], fontsize=9)
        ax.set_xlabel("Training-pool size N", fontsize=9)
        ax.set_ylabel("Epoch budget", fontsize=9)

        sub = label
        if nfull_3ep is not None:
            sub += f"\n3ep x N=full ref (best-eval-loss): {nfull_3ep:.3f}"
        ax.set_title(sub, fontsize=10)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        if nfull_3ep is not None and vrange[0] <= nfull_3ep <= vrange[1]:
            cbar.ax.axhline(nfull_3ep, color='red', linestyle='--', linewidth=1.2)

    out = OUT / "fig_v15_2d_4cat.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  -> {out.name}")


if __name__ == "__main__":
    rows_for_2d = load_rows()
    fig_2d_substitutability_capsule(rows_for_2d)
    fig_2d_substitutability_4cat(rows_for_2d)


def fig_v15_teaser(rows):
    """Composite teaser for Section 1: capsule 2D substitutability surface (left) and
    4-pattern catastrophic-Gemma decomposition means (right). 1x2 panel intended as
    the first figure a reader sees -- visualizes both headline findings at a glance.
    """
    import matplotlib.colors as mcolors

    fig = plt.figure(figsize=(14, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3])

    # LEFT PANEL: capsule 2D substitutability heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    sweep = [r for r in rows if r.get('recipe_version','').startswith('v0.5-2d-sweep')
             and r.get('category') == 'capsule']
    epochs = [3, 25, 50]
    ns = [5, 10, 20, 40]
    grid = np.full((len(epochs), len(ns)), np.nan)
    sd_grid = np.full_like(grid, np.nan)
    for i, ep in enumerate(epochs):
        for j, n in enumerate(ns):
            cells = [r['auroc'] for r in sweep
                     if r.get('recipe_version','').endswith(f"{ep}ep")
                     and r.get('n_samples') == n]
            if cells:
                grid[i, j] = statistics.mean(cells)
                if len(cells) > 1:
                    sd_grid[i, j] = statistics.stdev(cells)

    norm = mcolors.Normalize(vmin=0.80, vmax=0.93)
    im = ax1.imshow(grid, cmap='viridis', norm=norm, aspect='auto', origin='lower')
    for i in range(len(epochs)):
        for j in range(len(ns)):
            if not np.isnan(grid[i, j]):
                txt = f"{grid[i, j]:.3f}"
                ax1.text(j, i, txt, ha='center', va='center',
                         color='white' if grid[i, j] < 0.865 else 'black',
                         fontsize=11, fontweight='bold')
    ax1.set_xticks(range(len(ns)))
    ax1.set_xticklabels([f"N={n}" for n in ns], fontsize=10)
    ax1.set_yticks(range(len(epochs)))
    ax1.set_yticklabels([f"{ep}ep" for ep in epochs], fontsize=10)
    ax1.set_xlabel("Training-pool size N", fontsize=10)
    ax1.set_ylabel("Epoch budget", fontsize=10)
    ax1.set_title("Substitutability surface (capsule, mvtec)\nmore compute trades for less data:\n50ep x N=40 = 0.918 reaches above 3ep x N=full",
                  fontsize=10)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="AUROC")

    # RIGHT PANEL: 4-pattern decomposition (means + std error bars only, compact)
    ax2 = fig.add_subplot(gs[0, 1])
    cats_with_labels = [
        ('visa/pipe_fryum', 'pipe_fryum',  'Gemma-specific\nweakness'),
        ('visa/pcb1',       'pcb1',        'Backbone-stability\ngradient'),
        ('visa/macaroni2',  'macaroni2',   'Data-fundamental'),
        ('visa/capsules',   'capsules-VisA','Convergent\nmid-recovery'),
    ]

    def cell_stats(cat, recipe):
        ds, c = cat.split('/')
        vs = [r['auroc'] for r in rows
              if r.get('dataset') == ds and r.get('category') == c
              and r.get('recipe_version') == recipe]
        return (statistics.mean(vs), statistics.stdev(vs) if len(vs) > 1 else 0.0, len(vs))

    g_means, g_sds = [], []
    q_means, q_sds = [], []
    p_means, p_sds = [], []
    for cat, _, _ in cats_with_labels:
        gm, gs_, _ = cell_stats(cat, 'v0.4-longepoch-validation')
        qm, qs_, _ = cell_stats(cat, 'v0.4-longepoch-validation-qwen3')
        pm, ps_, _ = cell_stats(cat, 'v0.4-longepoch-validation-pixtral')
        g_means.append(gm); g_sds.append(gs_)
        q_means.append(qm); q_sds.append(qs_)
        p_means.append(pm); p_sds.append(ps_)

    x = np.arange(len(cats_with_labels))
    width = 0.27
    ax2.bar(x - width, g_means, width, yerr=g_sds, capsize=4, label='Gemma 4 E4B-it',  color='#3470B5')
    ax2.bar(x,         q_means, width, yerr=q_sds, capsize=4, label='Qwen3-VL-8B',     color='#3CA452')
    ax2.bar(x + width, p_means, width, yerr=p_sds, capsize=4, label='Pixtral-12B',     color='#D45A4C')
    ax2.axhline(0.5, color='grey', linestyle=':', linewidth=0.8, alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{disp}\n{lbl}" for _, disp, lbl in cats_with_labels], fontsize=9)
    ax2.set_ylim(0.3, 1.0)
    ax2.set_ylabel("AUROC at 50ep x N=10 (5 seeds)", fontsize=10)
    ax2.set_title("4-pattern decomposition of catastrophic Gemma cluster\n(3 independent backbone families)",
                  fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle("Headline findings: compute-data substitution on the substitutability surface (left) and"
                 " 4-pattern catastrophic-Gemma decomposition (right)",
                 fontsize=11, y=1.04)
    out = OUT / "fig_v15_teaser.png"
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out.name}")


if __name__ == "__main__":
    rows_for_teaser = load_rows()
    fig_v15_teaser(rows_for_teaser)
