"""Verdict closer — the Aether "GitHub trick" for FSVLM.

For a given set of runs (typically one pass in queue.json), compare the results to a baseline
recipe and assign each row a status:

- **new_baseline**: This recipe/config is the new best; anchor future comparisons here.
- **keep**: Real improvement (>= ABS_LIFT on AUROC or F1 vs. baseline, above NOISE threshold).
- **noop**: All metrics within ±NOOP_TOL of baseline — no behavioral change. Anti-goodharting.
- **discard**: Worse than baseline beyond NOISE. The recipe change should be rolled back.

This is pure computation on the results file. It does NOT edit the working tree or run git — the
calling skill (`/autoresearch`) decides whether to act on the verdict (e.g., `git checkout`
recipe files on discard).

Usage:
    python experiments/verdict.py \
        --results research/dataset_size_results.json \
        --pass-id pass1-smoke \
        --write  # mutate rows in-place; otherwise dry-run to stdout
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# Decision thresholds. Tune here, not inline in code.
ABS_LIFT = 0.03     # AUROC or F1 change this large counts as real improvement
NOOP_TOL = 0.02     # metrics changing by less than this are noise / no-op
NOISE = 0.02        # below this, "worse" doesn't count as discard (stays keep)


def aggregate_by_cell(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, dict]:
    """Group rows by a tuple of keys and compute mean ± stdev of core metrics."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out: dict[tuple, dict] = {}
    for cell, cell_rows in groups.items():
        aurocs = [r["auroc"] for r in cell_rows]
        f1s = [r["f1"] for r in cell_rows]
        out[cell] = {
            "n_runs": len(cell_rows),
            "mean_auroc": statistics.fmean(aurocs),
            "std_auroc": statistics.pstdev(aurocs) if len(aurocs) > 1 else 0.0,
            "mean_f1": statistics.fmean(f1s),
            "std_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
            "rows": cell_rows,
        }
    return out


def classify(curr: dict, baseline: dict | None) -> tuple[str, str]:
    """Return (status, reason) comparing curr summary to baseline summary."""
    if baseline is None:
        return "new_baseline", "no prior baseline; this cell becomes the anchor"

    auroc_delta = curr["mean_auroc"] - baseline["mean_auroc"]
    f1_delta = curr["mean_f1"] - baseline["mean_f1"]

    best_delta = max(auroc_delta, f1_delta)
    worst_delta = min(auroc_delta, f1_delta)

    # Anti-goodharting: every metric within ±NOOP_TOL → no real change
    if abs(auroc_delta) < NOOP_TOL and abs(f1_delta) < NOOP_TOL:
        return "noop", (
            f"within noise: ΔAUROC={auroc_delta:+.3f} ΔF1={f1_delta:+.3f} "
            f"(tol=±{NOOP_TOL})"
        )

    # Real lift on at least one metric
    if best_delta >= ABS_LIFT:
        return "keep", (
            f"lift: ΔAUROC={auroc_delta:+.3f} ΔF1={f1_delta:+.3f} "
            f"(threshold=+{ABS_LIFT})"
        )

    # Regression beyond noise → discard
    if worst_delta <= -NOISE:
        return "discard", (
            f"regression: ΔAUROC={auroc_delta:+.3f} ΔF1={f1_delta:+.3f} "
            f"(noise=±{NOISE})"
        )

    # Marginal win inside noise band
    return "keep", (
        f"marginal: ΔAUROC={auroc_delta:+.3f} ΔF1={f1_delta:+.3f}"
    )


def compute_verdicts(
    rows: list[dict],
    pass_rows: list[dict],
) -> dict[tuple[str, str, int], tuple[str, str]]:
    """For every (dataset, category, n_samples) cell in pass_rows, return (status, reason).

    Baseline per cell = earlier rows for the same (dataset, category, n_samples) with the
    lowest n_samples if no direct match (i.e. compare N=30 vs zero-shot baseline).
    """
    pass_cells = aggregate_by_cell(pass_rows, ("dataset", "category", "n_samples"))
    all_cells = aggregate_by_cell(rows, ("dataset", "category", "n_samples"))

    verdicts: dict[tuple[str, str, int], tuple[str, str]] = {}
    for cell, summary in pass_cells.items():
        dataset, category, n_samples = cell
        # Baseline = zero-shot (n=0) for the same dataset/category, if present
        zero_cell = (dataset, category, 0)
        baseline = all_cells.get(zero_cell) if zero_cell != cell else None
        status, reason = classify(summary, baseline)
        verdicts[cell] = (status, reason)
    return verdicts


def apply_verdicts(rows: list[dict], verdicts: dict) -> int:
    """Mutate rows in-place — write status + status_reason. Return rows changed."""
    changed = 0
    for r in rows:
        cell = (r["dataset"], r["category"], r["n_samples"])
        if cell in verdicts and not r.get("status"):
            status, reason = verdicts[cell]
            r["status"] = status
            r["status_reason"] = reason
            changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("research/dataset_size_results.json"))
    parser.add_argument("--pass-id", default=None,
                        help="Filter to rows whose recipe_version matches a queue pass. "
                             "If omitted, classify ALL rows against zero-shot baselines.")
    parser.add_argument("--write", action="store_true",
                        help="Mutate --results in place. Without this flag, dry-run to stdout.")
    args = parser.parse_args()

    rows = json.loads(args.results.read_text())
    if not isinstance(rows, list) or not rows:
        print(f"No rows in {args.results}")
        return 1

    if args.pass_id:
        pass_rows = [r for r in rows if r.get("recipe_version") == args.pass_id
                     or r.get("notes", []) and args.pass_id in r.get("notes", [])]
    else:
        pass_rows = [r for r in rows if r.get("n_samples", 0) > 0]  # skip zero-shot baselines

    if not pass_rows:
        print(f"No rows matched pass-id={args.pass_id}")
        return 1

    verdicts = compute_verdicts(rows, pass_rows)
    print(json.dumps(
        [{"cell": list(k), "status": v[0], "reason": v[1]} for k, v in verdicts.items()],
        indent=2,
    ))

    if args.write:
        changed = apply_verdicts(rows, verdicts)
        args.results.write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {changed} row(s) with status.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
