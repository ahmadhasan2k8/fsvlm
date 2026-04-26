# fsvlm — Project Framing & Strategy

fsvlm is an open-source, **model-agnostic** framework for benchmarking and fine-tuning generative
vision-language models in the few-shot regime on a single consumer GPU. The motivating research
question — *how few natural-language-labeled images are enough to fine-tune a modern VLM into a
useful detector?* — drives the choice of opinionated defaults, but the framework itself is
general: **any** supported VLM × **any** labeled-image dataset → adapter + tiered evaluation +
provenance-stamped result rows.

v0.1 ships with a Google Gemma 4 E4B-it backend and demonstration readers for **MVTec AD**,
**VisA**, and **DeepPCB** — the industrial-anomaly domain we used to develop and stress-test
the framework. New backends (Qwen-VL, LLaVA, Phi-Vision, future Gemmas) and new datasets land
as one new file each through the `ModelBackend` and `LabelReader` abstract base classes; no
existing code changes. Apache 2.0 — fully open, commercial use allowed, citation appreciated
for academic work.

---

## What this project IS

A model-agnostic, dataset-agnostic, single-GPU, local-first framework with five extension
points (`ModelBackend`, `LabelReader`, `ScoreExtractor`, `TrainingBackend`, `ReportGenerator`).
The opinionated defaults that ship in v0.1:

1. **SAM 2 assisted interactive labeling** — click on a defect, SAM 2 proposes the mask, you
   describe it in natural language. An annotation agent classifies free-form descriptions into
   a consistent defect taxonomy.
2. **End-to-end QLoRA fine-tuning pipeline** built on **unsloth** with defaults tuned for
   10–100 labeled examples on a single consumer GPU.
3. **Tiered self-evaluation** — every run reports zero-shot, few-shot, and full-train metrics
   on the same fixed held-out split so fine-tuning lift is always measured against its own
   baseline.
4. **Score-extractor cascade (v0.1)** — token-logit probability extractor exposed as a
   first-class methodology axis (most prior generative-VLM work treats extractor design as an
   implementation detail).
5. **Auto-research sweep** across LoRA ranks / learning rates with early termination and
   overfit detection.
6. **Append-only result log** with git SHA + recipe-version provenance on every row;
   `verdict.py` classifies each row as `keep` / `discard` / `noop` / `new_baseline` with
   anti-goodharting tolerances.
7. **Pre-registered defect taxonomy** for every demonstration dataset, frozen with an ISO
   timestamp before sweep results are observed.
8. **Edge deployment path** — ONNX / quantized export for running tuned adapters on-device. *(planned)*

## What this project IS NOT

- Not a model. There is no fsvlm checkpoint to download, no fsvlm architecture to cite.
- Not a SaaS, not a cloud service, not a marketplace.
- Not competing with unsloth — **built on** unsloth. Training-speed/VRAM wins are unsloth's, not ours.
- Not a cluster framework. Multi-GPU and distributed training are out of scope for v0.1.
- Not targeting non-technical end users — comfort with a Python shell is assumed.
- Not used in, trained on, nor targeting any of the following verticals: aerial imagery, drone
  inspection, wind turbines, solar panels, transmission lines, power distribution equipment, or
  other utility/energy infrastructure. Hard project-level scoping decision.

---

## Unsloth credit (non-negotiable)

fsvlm is built on top of [unsloth](https://github.com/unslothai/unsloth). The fast, memory-efficient
fine-tuning kernels are theirs. We credit them clearly in:

- README first paragraph
- Every blog post
- Every benchmark that touches training speed or VRAM
- `pyproject.toml` dependencies

**Do not claim unsloth's wins.** Training speed and VRAM reduction are their contribution. Our
contribution lives in the layers above: labeling UX, few-shot recipes, self-evaluation protocol,
agent orchestration, edge deployment, and the end-to-end workflow.

---

## Research questions and benchmarks

We benchmark the things we add on top of unsloth. Training-speed benchmarks would be stolen valor.

### Q1 — How many labels is enough? *(headline question)*

Prove or disprove whether natural-language-labeled small datasets (20–40 images) are enough for
a useful defect detector on modern industrial anomaly benchmarks.

**Current preliminary evidence** (monotonic tier improvement, `research/tiered_results.json`):

| Dataset   | Zero-shot AUROC | Few-shot AUROC | Full-train AUROC |
|-----------|-----------------|----------------|------------------|
| hazelnut  | 0.919           | 0.962          | 0.974            |
| bottle    | 0.740           | 0.787          | 0.850            |
| metal_nut | 0.500           | 0.784          | 0.857            |

**Required next experiment — dataset-size sweep**: for each category in the benchmark suite
below, train with **N ∈ {10, 20, 30, 40, 60, 100, 200, full}** labeled examples (held-out test
set fixed, ≥3 seeds per N). Plot AUROC / F1 as a function of N. Honest reporting: include every
category, including ones where few-shot does not work.

### Q2 — Does fine-tuning actually help, monotonically, across datasets?

The tiered protocol (zero-shot → few-shot → full) must show monotonic AUROC lift. If it doesn't,
the framework is broken or the dataset is saturating. Either is worth knowing.

### Q3 — Where does fsvlm break?

Honest failure-mode analysis: categories and defect types where the framework underperforms
classical methods (Anomalib PatchCore) or a tuned CNN (YOLOv11). Publish them in
`docs/benchmarks.md § "When fsvlm is not the right choice"`. Null results matter.

---

## Benchmark datasets — the respectable suite

MVTec hazelnut/bottle/metal_nut alone is not a credible study. Use the following datasets in
tiers, and publish numbers on each before any public release.

### Tier A — foundational (must-have, non-negotiable)

- **MVTec AD** — 15 categories, the universal reference. Full suite, all 15 categories.
  License: non-commercial research (fine for benchmarking).
- **VisA** (Amazon) — 12 categories, more realistic defects, less saturated than MVTec.
  License: Creative Commons.

### Tier B — harder / modern (at least 2 required for credibility)

- **MVTec LOCO AD** — **logical** + structural anomalies. Logical anomalies (missing part,
  wrong arrangement, extra component) should be where VLMs outperform classical CV because they
  require reasoning, not just texture analysis. Natural test for the VLM thesis.
- **Real-IAD** (2024) — ~150k images, 30 categories, multi-view. The new large-scale reference.
  Running the full thing is expensive; pick a representative subset (5–8 categories) and be
  explicit about the selection.
- **MVTec 3D-AD** — point cloud + RGB. *Optional unless we support 3D; likely skip for v1.*

### Tier C — breadth (pick 1–2 to show cross-domain generalization)

- **DeepPCB** — PCB solder/trace defects. Electronics.
- **KolektorSDD2** — metal surface cracks. General manufacturing.
- **Severstal Steel Defect Detection** (Kaggle) — steel surface, large and well-known.
- **MPDD** — metal parts defect detection. Honest industrial mix.

### Datasets we explicitly avoid

- **Synthetic-only datasets** — reviewers discount them for real-world claims.
- **Tiny datasets with known SOTA saturation** — if AUROC is already 99.9%, the benchmark is
  uninformative.
- **Any aerial, drone, wind, solar, or utility-infrastructure dataset** — out of project scope.
- **Medical imaging** — regulated domain, liability, different evaluation norms, not in scope.

### Methodology commitments (non-negotiable)

- Fixed held-out test splits — use the dataset's official train/test where provided.
- ≥3 seeds per data point with mean ± stdev reported.
- Reproducibility: every dataset has a download script + a single command to reproduce the
  reported numbers (`python benchmarks/run_<dataset>.py --seed 42 --n 30`).
- Honest failure-mode callouts: if fsvlm loses on category X, say so and explain why.
- No cherry-picking. If we report N numbers on MVTec AD, we report all 15 categories.

### Baselines required in the launch benchmark table

- Zero-shot base VLM (Gemma 4 / Qwen3-VL) — our own Tier 1
- Zero-shot API models (GPT-4o / Claude / Gemini) — opt-in, for reference
- YOLOv11-nano fine-tuned on the same N examples
- Anomalib PatchCore (SOTA classical on MVTec)
- HF Transformers + PEFT (no unsloth, no us) — isolates unsloth's and our contributions

---

## Self-evaluating framework

fsvlm evaluates itself honestly at every step. These are not incidental plumbing — they are
the features that make the methodology credible:

- **Tiered validation** — `research/tiered_validation.py` runs three tiers (zero-shot,
  few-shot defect-only, full dataset) on the same fixed held-out split and produces a
  comparison report. The proof-of-lift protocol.
- **Auto-research sweep** — the TrainingAgent sweeps LoRA configs, picks best by F1, flags
  overfitting when `train_accuracy - val_accuracy > 0.1`.
- **Append-only result log with provenance** — every (dataset, category, N, seed) call appends
  one row to `research/dataset_size_results.json` with `git_hash`, `recipe_version`, and
  `status` (after `verdict.py` classifies it as `keep` / `discard` / `noop` / `new_baseline`).
- **Pre-registered defect taxonomy** — committed to the repo as JSON with a frozen ISO
  timestamp. Editing the taxonomy after results are observed invalidates the rows it explains.
- **ValidationAgent** — generates confusion matrix, failure gallery, confidence histogram, and
  a plain-English self-analysis of VLM mistakes on its own predictions.

---

## Launch-readiness checklist

Items that convert the codebase into a credible open-source research release:

- [ ] README with 30-second quickstart, GIF of SAM annotation UI, 3-line CLI demo, unsloth credit in para 1
- [ ] Reproducible benchmarks: single-command-per-dataset scripts
- [ ] Baselines table vs. YOLOv11 + Anomalib + API-model zero-shot + raw HF+PEFT
- [ ] Dataset-size sweep covering the full Tier A + Tier B + at least one Tier C
- [ ] Edge deployment path: ONNX export → INT8 quantization → on-device inference benchmark
- [ ] Blog post: "How few natural-language labels are enough to fine-tune a VLM for defect detection?"
- [ ] Honest "When fsvlm is not the right choice" section in `docs/benchmarks.md`
- [ ] One meaningful PR to unsloth upstream as goodwill
- [ ] Reach out to unsloth maintainers before launch to confirm positioning as complementary
- [ ] Active maintenance commitment for at least 12 months post-release

---

## Failure modes to avoid

1. **Scope inflation.** Small research project → library → platform → marketplace. Resist this
   every time it re-emerges. If the scope feels "too constraining," that's the signal you're in
   the right place.
2. **Skipping benchmarks.** Code quality does not substitute for evidence. Without rigorous
   benchmarks, the research doesn't exist.
3. **Under-crediting unsloth.** Misattributed credit in the ML OSS community is a serious norm
   violation.
4. **Benchmark dishonesty.** Never cherry-pick, never hide failure modes, always report seeds and
   stdev. The ML community has long memories.
5. **Building for the wrong user.** This is a research tool for people comfortable in a Python
   shell. Not a consumer product.

---

## Decision date

By **2027-04-30**, fsvlm should have either:

- Public research release with full benchmark suite published and reproducible, OR
- Explicit decision to pause / sunset the project based on negative results.

Open-ended projects drift. A kill criterion keeps the research honest.
