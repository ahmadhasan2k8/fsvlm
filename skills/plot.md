---
name: plot
description: |
  Turn the results log into figures and tables ready for a paper, blog post, or README. Reads
  rows from `dataset_size_results.json` (or any compatible append-only log) and produces:
  AUROC-vs-N curves per category, extractor-comparison bar chart, FT-vs-ICL delta chart, and
  per-cell summary tables in CSV / Markdown / LaTeX.
  TRIGGER when: user has accumulated sweep rows and wants publication-ready visuals.
  SKIP when: user just wants metrics text (use /verdict's stdout output).
inputs:
  - results_path (path, default: research/dataset_size_results.json)
  - output_dir (path, default: docs/figures/)
  - figures (list, default: [auroc_vs_n, extractor_comparison, ft_vs_icl, per_cell_table])
  - filter_recipe (string, optional) — restrict to one recipe_version cohort
  - format (string, default: png) — png | pdf | svg for figures; tables always CSV+Markdown
eval_artifact: ${output_dir}/manifest.json
pass_criteria:
  - file exists at eval_artifact
  - JSON lists every requested figure with its file path and a non-zero file size
  - every listed file actually exists on disk
  - per_cell_table.csv has one row per (dataset, category, n_samples, recipe_version) cell
escalation: |
  If pass_criteria fails:
    - missing figures: matplotlib import error or no rows match the filter; check stderr
    - empty CSV: filter excluded all rows; relax filter_recipe
    - all values identical in a chart: the source rows are identical (extractor degenerate or
      sweep produced one cell only); re-run /sweep
---

# Skill: plot

## Purpose

The "I want to publish this" skill. Reads the row schema directly — works for any (dataset,
backend, category) tuple in the log without code changes.

## Procedure

```bash
python research/plots.py \
  --results "$results_path" \
  --output "$output_dir" \
  --figures "${figures[@]}" \
  ${filter_recipe:+--recipe "$filter_recipe"} \
  --format "$format"
```

Each figure is one matplotlib script call. The driver writes:

```
$output_dir/
  manifest.json              # what was generated, with file sizes
  auroc_vs_n_<dataset>_<category>.png   # one panel per category, log-x axis
  extractor_comparison.png   # bar chart, v0 vs v0.1 per category
  ft_vs_icl.png              # FT − ICL delta, error bars from seed stdev
  per_cell_table.csv         # one row per cell, mean ± stdev
  per_cell_table.md          # same data, Markdown table
  per_cell_table.tex         # same data, LaTeX booktabs (no figure floats)
```

`manifest.json`:
```json
{
  "results_source": "research/dataset_size_results.json",
  "filter_recipe": null,
  "generated_at": "2026-04-25T21:00:00Z",
  "git_hash": "...",
  "figures": [
    {"name": "auroc_vs_n_mvtec_hazelnut", "path": "docs/figures/auroc_vs_n_mvtec_hazelnut.png",
     "size_bytes": 28471, "n_cells_plotted": 9},
    ...
  ]
}
```

## Self-evaluation

PASS if `manifest.json` lists every requested figure, each path exists, file sizes are
non-zero, and the table has the expected row count.

## Failure modes

- **Empty figures**: filter_recipe excluded all rows. Drop the filter or pick a recipe that
  appears in the log.
- **`matplotlib` import fails**: install via `pip install matplotlib pandas`.
- **Plots look weird (collapsed lines, NaN bars)**: source rows have NaN metrics; usually a
  /sweep cell that crashed silently. Re-run /verdict to mark them `discard` and re-plot.

## Adapting to your runtime

- **Claude Code**: drop into `~/.claude/skills/`
- **OpenAI Agents SDK**: register `plot(results_path, output_dir, figures, filter_recipe, format)`
- **Plain shell**: invoke the Procedure block directly

## Pairs with

- **/sweep** + **/verdict** — produces the rows; /plot turns them into visuals
- **/autoresearch** — calls /plot at end-of-pass to build a per-pass figure pack
