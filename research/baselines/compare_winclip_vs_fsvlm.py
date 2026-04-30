"""Build the WinCLIP+ K=2 vs fsvlm N=2 matched-shot comparison table.

Inputs:
- research/baselines/winclip_k2_results.json (this script's sibling)
- research/dataset_size_results.json (fsvlm sweep)

Output: research/baselines/comparison_table.json + a printed table for paper.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_winclip() -> dict:
    rows = json.loads((REPO_ROOT / "research" / "baselines" / "winclip_k2_results.json").read_text())
    return {(r["dataset"], r["category"]): r for r in rows}


def load_fsvlm() -> dict:
    rows = json.loads((REPO_ROOT / "research" / "dataset_size_results.json").read_text())
    out = defaultdict(lambda: defaultdict(dict))  # recipe -> (d,c) -> n -> [auroc...]
    for r in rows:
        rec = r.get("recipe_version")
        if rec not in {"v0.3-tier-a", "v0.5-tier-a-qwen3"}:
            continue
        if r.get("auroc") is None:
            continue
        key = (r["dataset"], r["category"])
        n = r["n_samples"]
        out[rec][key].setdefault(n, []).append(r["auroc"])
    # mean across seeds
    means = {}
    for rec, cats in out.items():
        means[rec] = {}
        for k, ns in cats.items():
            means[rec][k] = {n: statistics.mean(v) for n, v in ns.items()}
    return means


def main() -> None:
    winclip = load_winclip()
    fsvlm = load_fsvlm()
    gemma = fsvlm.get("v0.3-tier-a", {})
    qwen = fsvlm.get("v0.5-tier-a-qwen3", {})

    print(f"\n{'='*108}")
    print(f"{'WinCLIP+ K=2 vs fsvlm N=2 matched-shot comparison':^108}")
    print(f"{'='*108}")
    header = (
        f"{'cat':<22}{'G_ZS':>8}{'Q_ZS':>8}"
        f"{'WinCLIP_K2':>12}"
        f"{'Gemma_N2':>10}{'Qwen3_N2':>10}"
        f"{'Gemma_N30':>11}{'Qwen3_N30':>11}"
        f"{'WinCLIP-G_N2':>14}{'WinCLIP-Q_N2':>14}"
    )
    print(header)
    print("-" * len(header))

    rows_out = []
    for (d, c), w in sorted(winclip.items(), key=lambda kv: gemma.get(kv[0], {}).get(0, 0)):
        g = gemma.get((d, c), {})
        q = qwen.get((d, c), {})
        g_zs, g_n2, g_n30 = g.get(0), g.get(2), g.get(30)
        q_zs, q_n2, q_n30 = q.get(0), q.get(2), q.get(30)
        win = w["auroc"]
        diff_g = (win - g_n2) if g_n2 is not None else None
        diff_q = (win - q_n2) if q_n2 is not None else None

        def f(x, prec=3):
            return f"{x:.{prec}f}" if x is not None else "—"
        def fs(x, prec=3):
            return f"{x:+.{prec}f}" if x is not None else "—"

        print(f"{f'{d}/{c}':<22}{f(g_zs):>8}{f(q_zs):>8}"
              f"{f(win):>12}{f(g_n2):>10}{f(q_n2):>10}"
              f"{f(g_n30):>11}{f(q_n30):>11}"
              f"{fs(diff_g):>14}{fs(diff_q):>14}")
        rows_out.append({
            "dataset": d, "category": c,
            "winclip_k2_auroc": win,
            "gemma_zs": g_zs, "gemma_n2": g_n2, "gemma_n30": g_n30,
            "qwen3_zs": q_zs, "qwen3_n2": q_n2, "qwen3_n30": q_n30,
            "winclip_minus_gemma_n2": diff_g,
            "winclip_minus_qwen3_n2": diff_q,
        })

    # Aggregate stats
    print("\n" + "-" * len(header))
    g_diffs = [r["winclip_minus_gemma_n2"] for r in rows_out if r["winclip_minus_gemma_n2"] is not None]
    q_diffs = [r["winclip_minus_qwen3_n2"] for r in rows_out if r["winclip_minus_qwen3_n2"] is not None]
    print(f"\nMean WinCLIP+ K=2 - Gemma N=2: {statistics.mean(g_diffs):+.3f}  "
          f"(median {statistics.median(g_diffs):+.3f}, n={len(g_diffs)})")
    print(f"Mean WinCLIP+ K=2 - Qwen3 N=2: {statistics.mean(q_diffs):+.3f}  "
          f"(median {statistics.median(q_diffs):+.3f}, n={len(q_diffs)})")
    win_count_g = sum(1 for x in g_diffs if x > 0)
    win_count_q = sum(1 for x in q_diffs if x > 0)
    print(f"WinCLIP+ K=2 beats Gemma N=2: {win_count_g}/{len(g_diffs)} cats")
    print(f"WinCLIP+ K=2 beats Qwen3 N=2: {win_count_q}/{len(q_diffs)} cats")

    out_path = REPO_ROOT / "research" / "baselines" / "comparison_table.json"
    out_path.write_text(json.dumps({
        "rows": rows_out,
        "summary": {
            "n": len(rows_out),
            "winclip_minus_gemma_n2_mean": statistics.mean(g_diffs) if g_diffs else None,
            "winclip_minus_gemma_n2_median": statistics.median(g_diffs) if g_diffs else None,
            "winclip_minus_qwen3_n2_mean": statistics.mean(q_diffs) if q_diffs else None,
            "winclip_minus_qwen3_n2_median": statistics.median(q_diffs) if q_diffs else None,
            "winclip_beats_gemma_n2": f"{win_count_g}/{len(g_diffs)}",
            "winclip_beats_qwen3_n2": f"{win_count_q}/{len(q_diffs)}",
        },
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
