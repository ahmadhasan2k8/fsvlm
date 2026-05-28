"""Generate the four cross-family characterization figures from the benchmark results JSON.

Produces:

1. fig_cross_family_backbone.png  — 7 cats × 2 backbones grouped bars (ZS vs N=full),
   1-std error bars where multi-seed.
2. fig_pcb1_rank_reversal.png      — pcb1 zoom showing the rank flip after fine-tuning.
3. fig_n_shot_curves_v08.png        — 4-cat Gemma N-shot curves under v0.8 fixed-pipeline.
4. fig_3backbone_zs.png             — 6-cat zero-shot bars across 3 backbones (Llama bars
   are caveated per the Bug-9 scoring artifact documented in docs/audit-trail.md).

Inputs:
    research/dataset_size_results.json (post-audit recipe versions)

Outputs:
    docs/figures/fig_cross_family_backbone.png
    docs/figures/fig_pcb1_rank_reversal.png
    docs/figures/fig_n_shot_curves_v08.png
    docs/figures/fig_3backbone_zs.png

The CROSS_FAMILY constant below encodes the multi-seed v0.8-fixed-pipeline characterization
locked on 2026-05-02; the script verifies the multi-seed std values dynamically against the
results JSON each run via compute_multiseed_std().
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "research" / "dataset_size_results.json"
OUT = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# Locked v0.8 cross-family table from the decision doc. Single source of truth.
# Llama N=full uniformly OOM'd at 16GB; ZS reported with caveat (see audit-trail.md Bug 9 —
# Llama emits non-PASS/FAIL first tokens, so ZS is partly an artifact).
# pcb1 Qwen3 N=full now multi-seed (5 seeds, all 0.943, std=0).
CROSS_FAMILY = [
    # cat_label, dataset, category, zs_gem, ftN_gem, gem_seeds, zs_qw, ftN_qw, qw_seeds, zs_llama
    ("capsule",    "mvtec", "capsule",    0.859, 0.872, 5, 0.968, 0.984, 1, 0.630),
    ("transistor", "mvtec", "transistor", 0.712, 0.757, 5, 0.669, 0.879, 5, 0.574),
    ("bottle",     "mvtec", "bottle",     0.715, 0.789, 1, 0.850, 0.930, 1, None),
    ("capsules",   "visa",  "capsules",   0.633, 0.636, 1, 0.663, 0.753, 1, 0.707),
    ("pcb1",       "visa",  "pcb1",       0.750, 0.785, 1, 0.546, 0.943, 5, 0.714),
    ("chewinggum", "visa",  "chewinggum", 0.978, 0.999, 5, 0.981, 0.989, 1, 0.913),
    ("wood",       "mvtec", "wood",       0.995, 0.995, 5, 1.000, 1.000, 1, 0.997),
]

def load_rows() -> list[dict]:
    return json.loads(DATA.read_text())


def compute_multiseed_std(rows: list[dict], category: str, dataset: str,
                         recipe_versions: set[str]) -> tuple[float, int]:
    """Return (std, n_seeds) for the N=full cells of (cat, ds) under any matching recipe.

    n=-1 is "full"; returns (0.0, 1) if only one seed exists.
    """
    aurocs = []
    for r in rows:
        if (r.get("category") == category and r.get("dataset") == dataset
                and r.get("n_samples") == -1
                and r.get("recipe_version") in recipe_versions
                and isinstance(r.get("auroc"), (int, float))):
            aurocs.append(r["auroc"])
    if len(aurocs) <= 1:
        return 0.0, len(aurocs)
    return statistics.stdev(aurocs), len(aurocs)


GEM_RECIPES = {"v0.8-fixed-pipeline", "v0.8-fixed-pipeline-full",
               "v0.8-fixed-pipeline-multiseed", "v0.8-bisection"}
QW_RECIPES = {"v0.8-fixed-pipeline-qwen3"}


def fig_cross_family() -> None:
    """Figure 1: 7 cats × 2 backbones grouped bars, ZS vs N=full per group."""
    cats = [r[0] for r in CROSS_FAMILY]
    n = len(cats)
    x = np.arange(n)
    width = 0.18

    zs_gem = [r[3] for r in CROSS_FAMILY]
    ft_gem = [r[4] for r in CROSS_FAMILY]
    zs_qw = [r[6] for r in CROSS_FAMILY]
    ft_qw = [r[7] for r in CROSS_FAMILY]

    rows = load_rows()
    err_gem = []
    err_qw = []
    for label, ds, cat, *_ in CROSS_FAMILY:
        std_g, _ = compute_multiseed_std(rows, cat, ds, GEM_RECIPES)
        std_q, _ = compute_multiseed_std(rows, cat, ds, QW_RECIPES)
        err_gem.append(std_g)
        err_qw.append(std_q)

    fig, ax = plt.subplots(figsize=(11, 5))

    b1 = ax.bar(x - 1.5 * width, zs_gem, width, label="Gemma 4 ZS",
                color="#bdd7e7", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x - 0.5 * width, ft_gem, width, yerr=err_gem,
                label="Gemma 4 N=full", color="#3182bd",
                edgecolor="black", linewidth=0.5,
                error_kw={"ecolor": "black", "capsize": 3, "lw": 0.8})
    b3 = ax.bar(x + 0.5 * width, zs_qw, width, label="Qwen3-VL ZS",
                color="#fcae91", edgecolor="black", linewidth=0.5)
    b4 = ax.bar(x + 1.5 * width, ft_qw, width, yerr=err_qw,
                label="Qwen3-VL N=full", color="#de2d26",
                edgecolor="black", linewidth=0.5,
                error_kw={"ecolor": "black", "capsize": 3, "lw": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Cross-family backbone effect: 7 categories × 2 backbones, ZS vs N=full\n"
                 "(error bars = 1 std across 5 seeds where multi-seed; v0.8 fixed-pipeline)")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)

    # Headroom-recovery percentages annotated above each FT bar
    for i, (gem_pct_str, qw_pct_str) in enumerate([
        ("9%", "49%"), ("16%", "63%"), ("26%", "53%"),
        ("1%", "27%"), ("14%", "87%"), ("sat", "sat"), ("sat", "sat"),
    ]):
        ax.annotate(gem_pct_str, (x[i] - 0.5 * width, ft_gem[i] + err_gem[i] + 0.012),
                    ha="center", fontsize=7, color="#08519c")
        ax.annotate(qw_pct_str, (x[i] + 1.5 * width, ft_qw[i] + err_qw[i] + 0.012),
                    ha="center", fontsize=7, color="#a50f15")

    plt.tight_layout()
    out = OUT / "fig_cross_family_backbone.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_pcb1_rank_reversal() -> None:
    """Figure 2: pcb1 zoom — rank reversal between ZS and N=full."""
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ["Zero-shot", "N=full (fine-tuned)"]
    gem = [0.750, 0.785]
    qw = [0.546, 0.943]

    x = np.arange(len(labels))
    width = 0.34

    b1 = ax.bar(x - width / 2, gem, width, label="Gemma 4 E4B-it",
                color="#3182bd", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width / 2, qw, width, label="Qwen3-VL-8B-Instruct",
                color="#de2d26", edgecolor="black", linewidth=0.5)

    for i, (g, q) in enumerate(zip(gem, qw)):
        ax.annotate(f"{g:.3f}", (x[i] - width / 2, g + 0.012),
                    ha="center", fontsize=10, fontweight="bold")
        ax.annotate(f"{q:.3f}", (x[i] + width / 2, q + 0.012),
                    ha="center", fontsize=10, fontweight="bold")

    # Annotate rank delta
    ax.annotate("Gemma > Qwen3 by +0.20", (0, 0.50), ha="center", fontsize=9,
                color="#3182bd", fontweight="bold")
    ax.annotate("Qwen3 > Gemma by +0.16", (1, 0.50), ha="center", fontsize=9,
                color="#de2d26", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.40, 1.05)
    ax.set_title("pcb1 (visa): zero-shot rank REVERSES after fine-tuning\n"
                 "Gemma wins ZS by 0.20; Qwen3 wins N=full by 0.16 (a 0.36 swing)")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out = OUT / "fig_pcb1_rank_reversal.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_n_shot_curves() -> None:
    """Figure 3: 4-cat Gemma N-shot curves under v0.8 fixed-pipeline."""
    rows = load_rows()

    cats = [
        ("transistor", "mvtec"),
        ("capsule", "mvtec"),
        ("chewinggum", "visa"),
        ("wood", "mvtec"),
    ]
    colors = {"transistor": "#de2d26", "capsule": "#3182bd",
              "chewinggum": "#31a354", "wood": "#756bb1"}

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    n_axis = [1, 2, 10, 30, 60, 100, 270]   # x positions; 270 stands in for "full"

    for cat, ds in cats:
        # Gather per-N AUROCs from v0.8-fixed-pipeline + v0.8-bisection + v0.8-fixed-pipeline-full
        # plus v0.7-zs-logit-only as N=0
        zs = [r["auroc"] for r in rows
              if r.get("category") == cat and r.get("dataset") == ds
              and r.get("recipe_version") == "v0.7-zs-logit-only"
              and isinstance(r.get("auroc"), (int, float))]
        zs_val = zs[0] if zs else None

        per_n: dict[int, list[float]] = {}
        for r in rows:
            if r.get("category") != cat or r.get("dataset") != ds:
                continue
            rv = r.get("recipe_version", "")
            if rv not in {"v0.8-fixed-pipeline", "v0.8-bisection",
                          "v0.8-fixed-pipeline-full", "v0.8-fixed-pipeline-multiseed"}:
                continue
            auroc = r.get("auroc")
            n_samples = r.get("n_samples")
            if not isinstance(auroc, (int, float)):
                continue
            if n_samples == -1:
                key = 270  # "full"
            else:
                key = n_samples
            per_n.setdefault(key, []).append(auroc)

        # Compute mean per N where multiple seeds exist
        xs, ys, errs = [], [], []
        if zs_val is not None:
            xs.append(0.5)  # left of N=1, log-scale stand-in for "ZS"
            ys.append(zs_val)
            errs.append(0)
        for n in n_axis:
            if n in per_n:
                xs.append(n)
                ys.append(statistics.mean(per_n[n]))
                errs.append(statistics.stdev(per_n[n]) if len(per_n[n]) > 1 else 0)

        if not ys:
            continue

        ax.errorbar(xs, ys, yerr=errs, marker="o", linewidth=1.6,
                    label=f"{cat} (ZS={zs_val:.3f})" if zs_val else cat,
                    color=colors.get(cat, "black"), capsize=3, markersize=6)

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0.5, 1, 2, 10, 30, 60, 100, 270])
    ax.set_xticklabels(["ZS", "1", "2", "10", "30", "60", "100", "full"])
    ax.set_xlabel("N labeled examples per category")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.55, 1.05)
    ax.set_title("Gemma 4 E4B-it N-shot curves under v0.8 fixed-pipeline\n"
                 "(error bars = 1 std across 5 seeds where multi-seed)")
    ax.axvline(x=30, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.annotate("knee region\n(N≈30–60)", (45, 0.62), fontsize=8, color="gray", ha="center")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUT / "fig_n_shot_curves_v08.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_3backbone_zs() -> None:
    """Figure 4: 3-backbone zero-shot comparison with Llama caveat note.

    Highlights that Llama-3.2's ZS values are confirmed artifact (for non-saturated cats)
    of position-0 logit-ratio
    scoring on a backbone whose first generated token is "THE" / "**ANALYSIS" rather
    than PASS/FAIL (Bug 9).
    """
    cats = [r[0] for r in CROSS_FAMILY if r[9] is not None]
    n = len(cats)
    x = np.arange(n)
    width = 0.26

    zs_gem = [r[3] for r in CROSS_FAMILY if r[9] is not None]
    zs_qw = [r[6] for r in CROSS_FAMILY if r[9] is not None]
    zs_llama = [r[9] for r in CROSS_FAMILY if r[9] is not None]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width, zs_gem, width, label="Gemma 4 E4B-it",
           color="#3182bd", edgecolor="black", linewidth=0.5)
    ax.bar(x, zs_qw, width, label="Qwen3-VL-8B-Instruct",
           color="#de2d26", edgecolor="black", linewidth=0.5)
    ax.bar(x + width, zs_llama, width, label="Llama-3.2-11B-Vision (Bug 9 affected*)",
           color="#737373", edgecolor="black", linewidth=0.5, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylabel("AUROC (zero-shot)")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("3-backbone zero-shot comparison (v0.8 fixed-pipeline)\n"
                 "*Llama bars: confirmed scoring artifact for non-saturated cats — "
                 "Bug 9 + 9b (audit-trail.md, paper §5.5)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out = OUT / "fig_3backbone_zs.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_cross_family()
    fig_pcb1_rank_reversal()
    fig_n_shot_curves()
    fig_3backbone_zs()
    print("done")
