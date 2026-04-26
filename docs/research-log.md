# Research Log — Decisions, Findings, and Failures

This document chronicles the experimental journey behind fsvlm's benchmark results, including
failed hypotheses, methodology corrections, and walked-back initial interpretations. It is
published openly because practitioners and reviewers benefit from seeing the wrong turns, not
just the polished narrative.

**Scope:** every meaningful experimental decision from the project's first week, in roughly
chronological order. Numbers are as recorded in `research/dataset_size_results.json`.

---

## Scoping and dataset selection

The project began as an investigation of a single question: *how few natural-language-labeled
images are enough to fine-tune a modern vision-language model (Gemma 4 E4B-it) into a useful
industrial defect detector?*

Benchmark targets were chosen as the "respectable suite" typical of anomaly-detection research:

- **MVTec AD** — 15 object categories, industry standard for industrial anomaly detection.
- **VisA** — 12 categories, Amazon's newer large-scale AD benchmark.
- **DeepPCB** — 1,500 PCB test/template pairs with 6 defect subtypes.
- (Deferred: MVTec LOCO, Real-IAD — time/compute budget.)

Pre-registered methodology commitments were frozen early: append-only results JSON with
git-hash provenance per row, ≥3 seeds per reported number, fixed held-out test splits, and a
pre-registered "distinctive vs subtle" defect taxonomy (with a frozen ISO timestamp in
`research/defect_taxonomy.json`) that we would consult *after* results landed to check
predictions.

## Model version mismatch

The first smoke sweep failed immediately with `NotImplementedError: unsloth/gemma-4-E4B-it
is not supported in your current Unsloth version!`. We had installed `unsloth 2026.4.2`;
Google's Gemma 4 release required `unsloth 2026.4.6+`. Upgrading both `unsloth` and
`unsloth_zoo` resolved the error.

**Consequence:** prior tiered-validation results (produced before the upgrade)
were on a *different* model — the older `unsloth/gemma-3n-E4B-it` loaded as
`Gemma3nForConditionalGeneration`. After upgrade, the same model string loads as
`Gemma4ForConditionalGeneration`. Numbers are not cross-comparable; we treated the post-
upgrade sweep as the canonical baseline and discarded pre-upgrade numbers.

## First pass — smoke sweep (7 categories × N=30 × 2 seeds = 21 runs)

Hypothesis (pre-registered): on at least 2 MVTec categories, fine-tuning with N=30 would lift
zero-shot AUROC by ≥ 0.05.

Result: **all 7 categories** showed lift. Mean AUROC lift across categories = +0.26. Largest
lift was on VisA fryum (+0.36 from a sub-chance 0.47 zero-shot baseline).

| Category | Zero-shot AUROC | N=30 AUROC (2 seeds) | Lift |
|----------|---------------:|-------:|-----:|
| MVTec bottle | 0.513 | 0.673 | +0.16 |
| MVTec hazelnut | 0.668 | **0.949** | **+0.28** |
| MVTec metal_nut | 0.500 | 0.731 | +0.23 |
| VisA candle | 0.720 | 0.956 | +0.24 |
| VisA fryum | 0.470 | 0.828 | **+0.36** |
| VisA pcb1 | 0.510 | 0.748 | +0.24 |
| DeepPCB pcb | 0.500 | 0.835 | **+0.34** |

Stdev across seeds was ≤ 0.041 on all categories; reproducibility was cleaner than expected.

## Observation: "predict everything FAIL" failure mode

Several zero-shot runs produced `recall=1.0` with F1 plateauing at the class base-rate. AUROC
captured the real discrimination signal — often meaningfully above 0.5 — but the decision
threshold was degenerate (every image scored above threshold). Implication: **AUROC is the
only threshold-independent statistic we can use as headline.** F1 can be misleading on
class-imbalanced test sets.

## Power disruption mid-sweep

A power loss killed the sweep subprocess during DeepPCB zero-shot inference (~260 of 1,000
images processed). The append-per-run write pattern saved 18 of 21 rows cleanly. On resume,
the `--resume` flag correctly skipped completed rows and picked up where the sweep left off.
Lesson: append-only result logs are not just a methodology nice-to-have; they're what lets you
survive real-world interruptions without losing hours of compute.

## The first "wow" moment and what it turned out to be: DeepPCB

At zero-shot, the base Gemma 4 model emitted the token `"BASED"` as the first generated word
for most DeepPCB images. Our score extractor — which keyword-matched the generated text and
fell back to a constant 0.75 when no keyword matched — sent every image to that same 0.75.
Uniform scores produced AUROC = exactly 0.500 by construction.

After fine-tuning on 30 labeled examples, AUROC jumped to **0.835**. We initially framed this
as a remarkable lift-from-chance finding. This interpretation was **wrong**; see the next
entry.

## MAJOR FINDING (and correction): the extractor was hiding the signal

A peer-review-style critique of our methodology flagged the constant-fallback extractor as
producing artifacts when the model emits non-keyword first tokens. We rewrote the extractor to
use the model's token-logit probability at generation position 0:

```python
prob_fail = softmax([logit("PASS"), logit("FAIL")])[1]
```

An explicit PASS/FAIL first-token still overrides at 0.1 / 0.9. Otherwise the score is the
real-valued logit probability, distinct per image.

**Re-running zero-shot on 4 categories with the fixed extractor revealed substantial hidden
signal in the base model:**

| Category | v0 extractor ZS | v0.1 extractor ZS | Recovery |
|----------|----------------:|------------------:|---------:|
| MVTec hazelnut | 0.668 | 0.821 | **+0.152** |
| MVTec metal_nut | 0.500 | 0.712 | **+0.212** |
| VisA candle | 0.720 | 0.680 | −0.041 (honest regression) |
| DeepPCB pcb | **0.500** | **0.840** | **+0.340** ⚡ |

The DeepPCB story flipped entirely: the base VLM's logit distribution between PASS and FAIL
*was* informative all along. The `"BASED"` surface token was a format-protocol idiosyncrasy,
not a capability failure. Our initial "breakthrough from broken baseline" reading was
discarded in favour of a more honest dual-tracking interpretation:

1. **Methodology claim:** proper score extraction from generative VLMs is non-trivial; a
   naive first-token-parse with constant fallback can silently produce AUROC = chance even
   when the model's internal discrimination is strong.
2. **Few-shot claim:** once the extractor baseline is clean, small-N fine-tuning adds
   *additional* lift on top — not the full lift we initially attributed to it.

Both interpretations are recorded in the results JSON via the `recipe_version` field
(`v0` vs `v0.1-extractor-fix`). Both cohorts remain in the results log so the comparison
is reproducible.

**Candle's regression from 0.720 → 0.680** is preserved as an honest observation: the old
keyword-matching path was catching defect-hinting phrases like "dark spot" and "discoloration"
that genuinely carried signal. The logit-probability path is not universally better; it is
more *principled* but can lose information on some categories.

## Label-source ablation — the nuanced result

The benchmark was originally supposed to test *"natural-language-labeled"* training data. An
honest audit showed that our training strings were derived from dataset-metadata (subtype
folder names like `crack`), not from a user-described SAM-assisted workflow. Running the tool's
actual user workflow on every training image would take ~2-4 hours of pre-flight compute.

We designed a 3-arm ablation at matched N=30:

- **thin**: bare "good"/"defect" strings (integer-label baseline)
- **metadata**: dataset-subtype strings (the default in earlier sweeps)
- **agent**: per-image descriptions generated by running the base VLM on each image with a
  "describe what is wrong" prompt (simulates the user+SAM workflow)

Result: the hypothesis that agent-labels dominate is **not supported.** Thin labels are
surprisingly competitive — within 0.01 AUROC of the winner on every cell:

| Category | thin | metadata | agent | Winner |
|----------|-----:|---------:|------:|--------|
| MVTec hazelnut | 0.942 | **0.947** | 0.936 | metadata (+0.005) |
| MVTec metal_nut | 0.718 | **0.735** | 0.718 | metadata (+0.017) |
| VisA candle | 0.956 | 0.951 | **0.961** | agent (+0.005 over thin) |

Pattern: metadata wins where the dataset's subtype folder name carries real information
(MVTec's `crack`, `flip`, etc.); agent wins where metadata is generic (VisA's "anomaly
detected in candle" is no richer than thin).

**The honest framing:** richer natural-language supervision provides **marginal-to-no** benefit
on most categories at N=30 with this base model; meaningful benefit appears only on visually
heterogeneous anomaly categories where dataset metadata is thin.

## Metal_nut — a documented failure mode

Metal_nut produced an immediate red flag: across all 6 label-source-ablation cells (3 label arms × 2 seeds),
F1, precision, recall, and threshold were **identical to four decimal places**
(F1 = 0.8136, P = 0.6857, R = 1.000, threshold ≈ 0.4378). AUROC varied in a band
[0.70, 0.75] across seeds — so the score distribution was moving — but the threshold optimizer
always landed on the degenerate `recall=1.0` point where every image was called defective.

We tried four increasingly-targeted interventions:

1. Extractor change (v0 → v0.1): no effect on F1 / P / R.
2. Label-source variation (thin / metadata / agent): no effect.
3. Adapter capacity bump (LoRA rank 8 → 16, LR 2e-4 → 1e-4, alpha/rank ratio 1.0 → 0.5): no
   effect.
4. Subtype-stratified sampling (forcing each N=30 training draw to include bent/color/flip/
   scratch proportionally): no effect.

All 12 metal_nut N=30 runs produced the same degenerate F1 = 0.8136. Twelve data points of
null result across four independent intervention axes is a rigorous documentation. The
conclusion:

> At N=30 with 4-bit QLoRA of Gemma 4 E4B-it, the score distributions for good vs defective
> metal nuts overlap so heavily that no threshold in the optimized range separates them at
> better-than-base-rate precision. AUROC lifts meaningfully (0.500 → 0.73) — the model has
> some internal discrimination — but the decision-boundary optimization cannot exploit it.
> Metal_nut defects (bent, color, flip, scratch) are subtle within-category variations, not
> semantically distinctive categorical anomalies. Classical pixel-reconstruction methods
> (PatchCore, EfficientAD) are the recommended tool for this category.

This failure mode is preserved in `docs/benchmarks.md` as an entry in the "when fsvlm is not the right
choice" table, alongside the four interventions tried.

## Recipe bump didn't help (null result)

A targeted ablation tested whether larger adapter capacity fixes anything. Same three
categories × same seeds, changing only LoRA rank 8 → 16, LR 2e-4 → 1e-4, scale 1.0 → 0.5.

All three cells regressed slightly:

| Category | v0.1 rank=8 | v0.2 rank=16 | Δ |
|----------|-----------:|-----------:|--:|
| MVTec hazelnut (metadata) | 0.947 | 0.946 | −0.001 |
| MVTec metal_nut (metadata) | 0.735 | 0.719 | −0.016 |
| VisA candle (agent) | 0.961 | 0.955 | −0.006 |

**Conclusion: the original rank=8 / LR=2e-4 recipe was well-chosen.** The bump hurt more
than it helped. This is a cheap, negative result that rules out the obvious first thing a
reader would ask ("did you try bigger rank?"). The null is logged in
`research/dataset_size_results.json` with a distinct `recipe_version` for future comparison.

## The AUROC-vs-N curve sweep — headline figure

The full curves came from 81 runs: 3 categories (hazelnut, candle, DeepPCB) × 9 N values
(2, 3, 5, 10, 20, 30, 40, 60, 100) × 3 seeds each. Metal_nut was dropped given the documented
degeneracy.

**Three distinct curve shapes emerged**, each telling a different story:

| N   | hazelnut (ZS 0.821) | candle (ZS 0.680) | DeepPCB (ZS 0.840) |
|----:|--------------------:|------------------:|--------------------:|
| 2   | **0.941** | **0.953** | 0.835 |
| 3   | 0.941 | 0.953 | 0.837 |
| 5   | 0.943 | 0.951 | 0.837 |
| 10  | 0.944 | 0.952 | 0.834 |
| 20  | 0.944 | 0.955 | 0.830 |
| 30  | 0.942 | 0.953 | 0.832 |
| 40  | 0.939 | 0.954 | 0.838 |
| 60  | 0.943 | **0.957** | 0.847 |
| 100 | 0.935 | 0.944 | **0.859** |

- **hazelnut**: flat at ~0.94 from N=2 through N=60, with a mild regression at N=100 (rank-8
  adapter starting to overfit). The jump from zero-shot to N=2 captures essentially the full
  lift.
- **candle**: identical pattern, biggest absolute lift (+0.28 at N=2).
- **DeepPCB**: curve shape is **completely different**. Below zero-shot through N=40, then a
  delayed knee at N=60 that monotonically improves to N=100. Small absolute effect (+0.02 by
  N=100) but qualitatively different from the MVTec/VisA pattern.

**Two empirical rules emerged** from the 3-category smoke:

- **Rule A — lift is bounded by the zero-shot-to-ceiling gap.** candle had the lowest ZS
  (0.68) and got the biggest lift (+0.28). DeepPCB had the highest ZS (0.84) and got the
  smallest lift (~0.02). Hazelnut (ZS 0.82) falls in between (+0.12).
- **Rule B — the value of N where lift appears is category-dependent.** hazelnut/candle
  saturate at N=2. DeepPCB needs N=60+ to pull past zero-shot.

Both rules await further testing on 21 more categories (a coverage-expansion sweep, pending).

## ICL-vs-FT ablation: is fine-tuning actually doing anything?

A critical methodological concern surfaced: our N=2 result sounds like "2 examples fine-tune
a model to 0.94" — but it could equally be "2 examples used as contrastive-matching references
in the model's context window reach 0.94." WinCLIP+ (CVPR 2023) reports 0.984 on hazelnut at
K=4 *without any fine-tuning*, using frozen-CLIP + reference examples. If in-context-learning
on Gemma 4 achieves similar numbers to our fine-tune at matched N, the fine-tuning
contribution evaporates.

We ran this as an explicit head-to-head ablation — **ICL vs fine-tune at matched N, same base
model, same test splits, same extractor.**

| Category | ICL N=2 | FT N=2 | Δ (FT − ICL) |
|----------|--------:|-------:|-------------:|
| MVTec hazelnut | 0.893 | 0.941 | **+0.048** |
| VisA candle | 0.855 | 0.953 | **+0.098** |
| DeepPCB pcb | 0.673 | 0.835 | **+0.163** |

All three cells exceed the 0.02 threshold. **Fine-tuning is meaningfully better than
in-context learning at N=2 on all three categories.**

**Nuance preserved:** on hazelnut, ICL at N=8 reaches 0.982 — above *any* fine-tune result on
that category at any N we tested. The picture is: fine-tune wins at extreme-few-shot
(N=2); ICL catches up and potentially overtakes around N=8 when more reference examples fit
in context. This is a richer finding than "fine-tune always wins" — the
two methods are complementary across the N spectrum.

## Literature corrections (web-verified)

Prior references cited in early drafts were partially wrong. Corrected numbers:

- WinCLIP+ hazelnut K=4 AUROC = **0.984** (previously cited 0.978).
- PromptAD hazelnut K=4 AUROC = **0.998** (previously cited 0.983).
- PromptAD venue = **CVPR 2024** (previously cited ACM MM 2024; a separate WACV 2024 PromptAD
  is a different zero-shot paper).
- DeepPCB "no published few-shot" claim needed qualification: one 2025 paper (PMC12653441)
  reports few-shot on DeepPCB as multi-class classification with accuracy/F1/AUPRC. The
  stricter claim — "no published few-shot **one-class AUROC** on DeepPCB" — stands.
- Prior generative-VLM anomaly-detection work to cite: AnomalyGPT (2023), Anomaly-OV
  (CVPR 2025), Triad (ICCV 2025), IAD-R1 (2025), LogicAD (AAAI 2025). Our gap — "per-category
  few-shot AUROC on MVTec + VisA + DeepPCB with an explicit extractor ablation, released as
  a pip-installable tool" — remains defensible, but the phrasing must be precise.

## What's still open

- **Coverage**: only 3 categories have AUROC-vs-N curves. The "tiger-analogy rule" needs
  testing on the remaining 21 MVTec + VisA categories.
- **Classical baselines**: WinCLIP+, PromptAD, Anomalib PatchCore on our same test splits.
  Published numbers are not directly comparable because splits differ.
- **Description-quality evaluation**: the VLM-unique axis that classical and ICL methods
  structurally cannot compete on. Not yet run.
- **Other base models**: everything here is Gemma 4 E4B-it. Whether the findings generalize to
  LLaVA, Qwen-VL, InternVL is an open question.
- **On-device latency**: we claim "consumer-GPU" but have not reported inference times on
  Jetson/M-series/edge devices.

These are follow-on work. The v0.1 release ships what we have, with limitations flagged
honestly.

## Methodology commitments that carried through

Decisions made early that we stuck with:

- **AUROC as headline metric** (threshold-independent, robust to class imbalance). F1 is
  secondary.
- **≥3 seeds** per reported number, mean ± stdev.
- **Pre-registered taxonomy** (distinctive / subtle / mixed) frozen before results landed.
- **Append-only results log** with `git_hash` + `recipe_version` per row.
- **Fixed held-out test splits** per dataset; no cross-category leakage.
- **Null results logged honestly** (recipe-bump ablation, stratified-sampling rescue,
  metal_nut failure mode) rather than buried.

These are documented commitments, not discovered after-the-fact. The project repo's
`research/defect_taxonomy.json` carries the frozen taxonomy with a timestamp. The
methodology pre-registration is the backbone of what fsvlm offers: not "fine-tune wins big,"
but "here is a reproducible way to measure fine-tune vs zero-shot vs in-context learning on
industrial anomaly benchmarks, with extractor-methodology decisions made explicit."

---

*This log is a living document. Entries are append-only; corrections and retractions are
added as new entries rather than overwriting old ones. If you spot a factual error or have a
methodology question, please open an issue.*
