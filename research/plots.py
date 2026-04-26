"""Generate paper-ready figures and tables from the append-only result log.

Reads `dataset_size_results.json` (or any file with the same row schema) and produces:
  - AUROC-vs-N curves per (dataset, category), one panel per category, log-x axis
  - Per-cell summary table in CSV, Markdown, and LaTeX

Used by the /plot skill. The implementation is intentionally minimal — pandas + matplotlib —
so it works in any environment where those two libraries install.

Usage:
    python research/plots.py --results research/dataset_size_results.json \
        --output docs/figures/ --figures auroc_vs_n per_cell_table

Add new figure kinds by writing a function and registering it in FIGURE_DISPATCH.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path


def _load_rows(results_path: Path, recipe_filter: str | None) -> list[dict]:
    rows = json.loads(results_path.read_text())
    if recipe_filter:
        rows = [r for r in rows if r.get("recipe_version") == recipe_filter]
    rows = [r for r in rows if r.get("auroc") is not None]
    return rows


def _aggregate_by_cell(rows: list[dict]) -> dict[tuple, dict]:
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r.get("dataset"), r.get("category"), r.get("n_samples"),
               r.get("recipe_version", "v0"))
        by_cell[key].append(r)
    out = {}
    for key, cell_rows in by_cell.items():
        aurocs = [r["auroc"] for r in cell_rows]
        f1s = [r.get("f1", 0.0) for r in cell_rows]
        out[key] = {
            "n_runs": len(cell_rows),
            "mean_auroc": statistics.fmean(aurocs),
            "std_auroc": statistics.pstdev(aurocs) if len(aurocs) > 1 else 0.0,
            "mean_f1": statistics.fmean(f1s),
            "std_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
        }
    return out


def figure_auroc_vs_n(rows: list[dict], output_dir: Path, fmt: str) -> list[dict]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return [{"name": "auroc_vs_n", "error": "matplotlib not installed; pip install matplotlib"}]

    by_cat: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ds, cat, n = r.get("dataset"), r.get("category"), r.get("n_samples")
        if ds and cat and n is not None:
            by_cat[(ds, cat)][n].append(r["auroc"])

    out_records = []
    for (ds, cat), n_to_aurocs in sorted(by_cat.items()):
        ns = sorted(n_to_aurocs.keys())
        means = [statistics.fmean(n_to_aurocs[n]) for n in ns]
        stds = [statistics.pstdev(n_to_aurocs[n]) if len(n_to_aurocs[n]) > 1 else 0.0 for n in ns]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(ns, means, yerr=stds, marker="o", capsize=4)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("N (labeled examples)")
        ax.set_ylabel("AUROC")
        ax.set_title(f"{ds} / {cat}")
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3)
        path = output_dir / f"auroc_vs_n_{ds}_{cat}.{fmt}"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        out_records.append({"name": f"auroc_vs_n_{ds}_{cat}", "path": str(path),
                            "size_bytes": path.stat().st_size, "n_cells_plotted": len(ns)})
    return out_records


def figure_per_cell_table(rows: list[dict], output_dir: Path, fmt: str) -> list[dict]:
    cells = _aggregate_by_cell(rows)
    if not cells:
        return [{"name": "per_cell_table", "error": "no rows"}]
    out_records = []
    # CSV
    csv_path = output_dir / "per_cell_table.csv"
    with csv_path.open("w") as f:
        f.write("dataset,category,n_samples,recipe_version,n_runs,mean_auroc,std_auroc,mean_f1,std_f1\n")
        for (ds, cat, n, rv), agg in sorted(cells.items()):
            f.write(f"{ds},{cat},{n},{rv},{agg['n_runs']},{agg['mean_auroc']:.4f},"
                    f"{agg['std_auroc']:.4f},{agg['mean_f1']:.4f},{agg['std_f1']:.4f}\n")
    out_records.append({"name": "per_cell_table_csv", "path": str(csv_path),
                        "size_bytes": csv_path.stat().st_size, "n_rows": len(cells)})

    # Markdown
    md_path = output_dir / "per_cell_table.md"
    with md_path.open("w") as f:
        f.write("| dataset | category | N | recipe | n | AUROC ± std | F1 ± std |\n")
        f.write("|---|---|---:|---|---:|---:|---:|\n")
        for (ds, cat, n, rv), agg in sorted(cells.items()):
            f.write(f"| {ds} | {cat} | {n} | {rv} | {agg['n_runs']} | "
                    f"{agg['mean_auroc']:.3f} ± {agg['std_auroc']:.3f} | "
                    f"{agg['mean_f1']:.3f} ± {agg['std_f1']:.3f} |\n")
    out_records.append({"name": "per_cell_table_md", "path": str(md_path),
                        "size_bytes": md_path.stat().st_size, "n_rows": len(cells)})

    # LaTeX (booktabs)
    tex_path = output_dir / "per_cell_table.tex"
    underscore_escape = "\\_"
    pm = "\\pm"
    line_end = "\\\\"
    nl = "\n"
    midrule_block = "\n\\midrule\n"
    with tex_path.open("w") as f:
        f.write("\\begin{tabular}{llrlrrr}\n\\toprule\n")
        f.write(
            f"dataset & category & $N$ & recipe & $n$ & AUROC ${pm}$ std & F1 ${pm}$ std {line_end}"
            f"{midrule_block}"
        )
        for (ds, cat, n, rv), agg in sorted(cells.items()):
            cat_esc = cat.replace("_", underscore_escape)
            rv_esc = rv.replace("_", underscore_escape)
            f.write(
                f"{ds} & {cat_esc} & {n} & {rv_esc} & {agg['n_runs']} & "
                f"${agg['mean_auroc']:.3f} {pm} {agg['std_auroc']:.3f}$ & "
                f"${agg['mean_f1']:.3f} {pm} {agg['std_f1']:.3f}$ {line_end}{nl}"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")
    out_records.append({"name": "per_cell_table_tex", "path": str(tex_path),
                        "size_bytes": tex_path.stat().st_size, "n_rows": len(cells)})
    return out_records


FIGURE_DISPATCH: dict[str, Callable] = {
    "auroc_vs_n": figure_auroc_vs_n,
    "per_cell_table": figure_per_cell_table,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="research/dataset_size_results.json")
    p.add_argument("--output", default="docs/figures/")
    p.add_argument("--figures", nargs="+", default=["auroc_vs_n", "per_cell_table"])
    p.add_argument("--recipe", default=None)
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.is_file():
        print(f"results file not found: {results_path}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(results_path, args.recipe)
    if not rows:
        print(f"no rows after recipe filter '{args.recipe}'")
        return 1

    figures = []
    for fname in args.figures:
        fn = FIGURE_DISPATCH.get(fname)
        if fn is None:
            print(f"unknown figure: {fname}; available: {list(FIGURE_DISPATCH)}")
            continue
        figures.extend(fn(rows, output_dir, args.format))

    manifest_path = output_dir / "manifest.json"
    git_hash = os.popen("git rev-parse HEAD 2>/dev/null").read().strip() or "unknown"
    manifest_path.write_text(json.dumps({
        "results_source": str(results_path),
        "filter_recipe": args.recipe,
        "git_hash": git_hash,
        "figures": figures,
    }, indent=2))
    print(f"wrote {len(figures)} artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
