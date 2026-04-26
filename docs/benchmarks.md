# Benchmarks

This page documents fsvlm's benchmark methodology, the respectable-dataset coverage target,
current coverage, headline results, and — crucially — the cases where fsvlm is **not** the
right choice. See [POSITIONING.md](../POSITIONING.md) for the non-negotiable methodology
commitments.

## Research question

*How few natural-language-labeled images are enough to fine-tune a modern VLM into a useful
defect detector on respected industrial anomaly benchmarks?*

## Methodology

- **Base model**: `unsloth/gemma-4-E4B-it` (Google Gemma 4 E4B-it, Unsloth 4-bit quantized).
  Loaded as `Gemma4ForConditionalGeneration`.
- **Training**: QLoRA via [unsloth](https://github.com/unslothai/unsloth). Credit to them for
  every VRAM / training-speed win; those are not our contribution.
- **Evaluation**: per-run we report AUROC, F1, precision, recall, accuracy, threshold, and
  num_test. AUROC is the headline — threshold-independent, robust to class imbalance.
- **Seeds**: ≥3 per data point where possible. Current smoke pass uses 2 seeds; the full
  launch sweep uses 3.
- **Test splits**: fixed held-out, deterministic. For MVTec we split each category's test set
  50/50 between the training pool and the held-out evaluation set so train-pool defects never
  leak into evaluation. VisA uses the official `split_csv/1cls.csv` protocol with anomaly-test
  also split 50/50.
- **Reproducibility**: every run writes a row to `research/dataset_size_results.json` with
  `git_hash`, `git_short`, `git_dirty`, `recipe_version`, and `status` fields for the Aether-
  style provenance chain. Every dataset has a single-command download script in
  `research/datasets/`.

## The respectable-dataset suite

Targets per [POSITIONING.md § "Benchmark datasets"](../POSITIONING.md#benchmark-datasets--the-respectable-suite).

### Tier A (foundational, must-have)

| Dataset | Categories | v0.1 coverage |
|---------|-----------:|--------------:|
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | 15 | 1 full N-curve + 5 cells at N=30 |
| [VisA](https://github.com/amazon-science/spot-diff) | 12 | 1 full N-curve + 2 cells at N=30 |

### Tier B (harder / modern, ≥2 required for launch credibility)

| Dataset | Categories | v0.1 coverage |
|---------|-----------:|--------------:|
| [MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco) | 5 | 0 (v0.2 target) |
| [Real-IAD](https://realiad4ad.github.io/) | 30 | 0 (v0.2 target) |

### Tier C (vertical credibility, ≥1 required)

| Dataset | Categories | v0.1 coverage |
|---------|-----------:|--------------:|
| [DeepPCB](https://github.com/tangsanli5201/DeepPCB) | 1 (pcb) | 1 full N-curve |

*This table updates as coverage advances. `research/dataset_size_results.json` is the source
of truth.*

## Baselines (planned for v0.2)

Required comparisons on the same fixed test splits:

- **Zero-shot base VLM** — Gemma 4 E4B-it without fine-tuning (Tier 1 reference, shipped in v0.1)
- **Zero-shot API models** — GPT-4o, Claude, Gemini (opt-in, for reference only)
- **YOLOv11-nano** — fine-tuned on the same N examples
- **[Anomalib PatchCore](https://github.com/openvinotoolkit/anomalib)** — classical SOTA on MVTec
- **Raw HF Transformers + PEFT** (no unsloth, no fsvlm) — isolates unsloth's and our contributions

Anomalib PatchCore is the first v0.2 target; the rest are tracked GitHub issues. Baseline runner
scripts will land in `research/baselines/`.

## Headline results — v0.1 demonstration

Three categories swept end-to-end at N ∈ {0, 2, 3, 5, 10, 20, 30, 40, 60, 100} labeled examples,
3 seeds per cell, on a single RTX 5080 Laptop 16 GB GPU. Gemma 4 E4B-it via QLoRA, v0.1 score-
extractor cascade. Numbers are mean across seeds (stdev in `research/dataset_size_results.json`):

| Category | Zero-shot AUROC | N=2 AUROC | N=30 AUROC | Curve shape |
|----------|----------------:|----------:|-----------:|-------------|
| MVTec hazelnut | 0.821 | **0.941** | 0.942 | knee at N=2; flat through N=60 |
| VisA candle    | 0.680 | **0.953** | 0.953 | knee at N=2; flat through N=60 |
| DeepPCB pcb    | 0.840 | 0.835 | 0.832 | "delayed knee" — at-ceiling at ZS, lift only at N≥60 |

The N=2 knee is the headline finding for distinctive defects. See the README figures and
[docs/research-log.md](research-log.md) for the full analysis, including the score-extractor
audit and the FT-vs-ICL ablation.

The remaining 24 MVTec + VisA categories have zero-shot and N=30 points from earlier sweeps but
no full N-curve yet — they're the v0.2 coverage target.

## When fsvlm is *not* the right choice

Honest failure-mode section, per Karpathy's "null results are logged just as honestly as
positive ones" discipline.

*Populated as the full sweep completes and per-category failure-mode review identifies the
losers. Expected candidates based on prior intuition:*

- **Periodic textures** (MVTec grid, carpet under certain patterns) — classical methods like
  Anomalib PatchCore typically win here because Fourier/texture cues beat semantic reasoning.
- **Sub-pixel defects** under low-resolution capture — VLMs need enough pixels for the defect
  to be visible; classical CNNs can sometimes learn from micro-texture cues the VLM tokenizer
  discards.
- **Ultra-imbalanced production data** (<1% defect rate with thousands of clean images) — the
  few-shot advantage collapses when full-pool classical methods have abundance on their side.

These will be quantified in the launch table with side-by-side numbers, not just hand-waved.

## Reproducing every number on this page

```bash
bash research/datasets/download_visa.sh
bash research/datasets/download_deeppcb.sh
# (MVTec AD: download from mvtec.com into research/mvtec_data/ — requires email registration)

bash research/run_sweep.sh \
  --datasets mvtec visa deeppcb \
  --categories hazelnut bottle metal_nut candle pcb1 fryum pcb \
  --n-values 0 30 \
  --seeds 42 1337 \
  --epochs 3

# Then classify each row as keep / discard / noop / new_baseline:
python research/verdict.py \
  --results research/dataset_size_results.json \
  --write
```

## Machine-readable raw data

`research/dataset_size_results.json` is the append-only ground truth. Every row is one
(dataset, category, n_samples, seed) run, carrying metrics plus `git_hash` / `recipe_version` /
`status` provenance.
