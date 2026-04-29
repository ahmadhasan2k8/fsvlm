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

## Score-extractor correction (the critique that flipped the DeepPCB interpretation)

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

**Important prior-art context (added post-launch).** Logit-based scoring of LLM outputs is
a known technique, not novel to this project. [LogicQA (Jin et al., AAAI 2025)](https://arxiv.org/abs/2501.01767)
explicitly validates that "using the token prediction probability as the reliability of the
answer and using it as the Anomaly Score is valid" — same domain, same technique. The wider
LLM-evaluation literature has explored P(True), Single Logit Probability (SLP), and
Multi-Token Reliability Estimation (MTRE) for years. The honest framing of what fsvlm
contributes here is the *worked-example open-source cascade implementation* + *the
documented effect size on a public benchmark for practitioners who built the naive
first-token pipeline and didn't know there was a better default*. Not "the literature has
been measuring wrong." A subset of pipelines (those that text-parse for AUROC instead of
using a separate decoder or reporting F1 only) benefit from the cascade; AnomalyGPT,
Anomaly-OV, and similar use other architectures and metrics entirely.

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

## Pass 4 — full Tier A coverage with held-out hypothesis testing

After the v0.1 release, the next question was whether the pre-registered defect-taxonomy
(`research/defect_taxonomy.json`, frozen 2026-04-20T12:30Z) actually predicts which categories
benefit from few-shot fine-tuning. The answer is **no**, and the discovery process is more
interesting than a clean confirmation would have been.

### Stage 1 — taxonomy hypothesis falsified (8 cats)

The pre-registered taxonomy classified each category as distinctive (defect contrasts strongly
with the good baseline), subtle (contrast is small), or mixed. Prediction: distinctive
categories should show large lift from N=2 fine-tuning; subtle ones should show little or none.

We ran 8 categories spanning the three tags at N ∈ {0, 2, 10, 30} × 3 seeds. **Mean lift
inverted the prediction:**

- distinctive (4 cats): mean lift = +0.031 — small lift on the cats predicted to win biggest.
- subtle (3 cats): mean lift = +0.218 — large lift on the cats predicted to lose.
- mixed (1 cat): mean lift = +0.314.

The hypothesis was clearly falsified, with no overlap in lift ranges between distinctive and
subtle categories. But the data revealed something stronger: lift correlates inversely with
zero-shot AUROC. Categories where the base VLM was already strong gained little; categories
where the base VLM was at chance gained the most.

### Stage 2 — held-out test (6 new cats), partial confirmation

Three quantitative predictions were locked into git history *before* any stage 2 cell ran:

1. Per-category bands: ZS ≤ 0.55 cats lift ≥ +0.10; ZS ≥ 0.70 cats lift ≤ +0.06.
2. Spearman ρ across all 14 cats (8 stage1 + 6 stage2) ≤ -0.6 with p < 0.05.
3. N=2 captures ≥ 80% of best lift on ≥ 4 of 6 stage2 cats.

Two of three passed cleanly:

- Spearman ρ = -0.758, p = 0.00084 (strong margin past threshold).
- N=2 knee held on 4 of 6 stage2 cats (exactly threshold).

The per-category band prediction failed on visa/macaroni2: ZS = 0.495 (just below threshold)
gave lift +0.088 (just below the +0.10 band). Per the no-goalpost-moving discipline, the band
prediction was *not* retroactively softened. It was explicitly dropped before stage 3 ran,
with the structural reasoning logged in the commit message of `93a9fa6`.

### Stage 3 — second held-out test (10 more cats), rule passes

Two surviving predictions were locked in for stage 3:

1. Spearman ρ across all 24 cats ≤ -0.6 with p < 0.01.
2. N=2 knee on ≥ 6 of 10 stage 3 cats.

Final result: **ρ = -0.778, p = 4×10⁻⁶ across 24 cats. N=2 knee passed on 9 of 10 stage 3 cats**
(the 10th, mvtec/leather, was at-ceiling at ZS=0.999 with effectively no lift to capture —
not a knee failure). Both predictions passed.

The full Tier A coverage closed at 240 rows: 168 keep, 45 noop, 24 new_baseline, 3 discard.
The 3 discards are all mvtec/transistor N=30 — the cleanest example of "high-ZS already-strong
categories where N=30 fine-tuning actively hurts." See commit `13f0a62` for the close-out.

## Recipe stability sub-study

A natural reviewer concern: is the rule recipe-specific? We re-ran 5 cats spanning the full
ZS range (capsule, tile, wood, zipper, chewinggum) at N=2 under three recipe variants:

- rank=16 (vs default 8), lr=2e-4
- rank=32, lr=2e-4
- rank=8, lr=1e-4 (vs default 2e-4)

**All four variants (baseline + three) gave Spearman ρ = -1.000 on the 5-cat subset, with
per-cell AUROC differing by ≤ 0.01 from the baseline on every (variant × cat) pair.** The rule
is not recipe-specific within this perturbation range. See commit `7d6c2b8`.

A real bug surfaced during this sub-study: `fsvlm.cli train` was constructing `TrainingConfig`
with only `model_name` from `FSVLMConfig` — `lora_rank`, `learning_rate`, etc. fell back to
hardcoded dataclass defaults. The recipe-stability test would have silently re-run rank=8 four
times. Fixed in commit `edbefc7` by pulling all config defaults through and adding explicit
`--lora-rank` / `--learning-rate` CLI flags.

## ICL extension on the high-lift categories

The pass 5b ICL-vs-FT comparison (3 cats: hazelnut, candle, deeppcb/pcb) was extended to 6
cats spanning the full ZS range (capsule, tile, wood, zipper, transistor, chewinggum,
plus the visa pcb4). Headline: the FT-vs-ICL answer is **category-dependent and predictable
from ZS-AUROC**:

- **chewinggum (ZS=0.441)**: ICL N=2 AUROC=0.979, FT N=2=0.984. ICL ≈ FT. Both achieve full lift.
- **pcb4 (ZS=0.510)**: ICL N=2=0.650 (high seed-variance), FT N=2=0.903. **FT > ICL by ~0.25**.
- **transistor (ZS=0.712)**: ICL > FT — the trained model regresses on a category where ZS
  is already strong.

ICL N=8 ran into a 16 GB memory constraint (8 reference image-label pairs in the prompt +
Gemma 4 base + test inference exceeds the budget); N=2 and N=4 are clean. See commit `7d6c2b8`.

## Multi-model phase — rule transfers to 2 of 3 model families

The next test: does the rule generalize beyond Gemma? We tested Qwen3-VL-8B-Instruct (Alibaba,
2025-current) and Llama-3.2-11B-Vision-Instruct (Meta, Sept 2024) on the same 5-category
subset, identical recipe (rank=8, lr=2e-4, epochs=3).

A real compatibility bug surfaced when first running Qwen2.5-VL: TRL 0.24's `SFTTrainer.__init__`
round-trips `args` through `transformers.TrainingArguments.to_dict()` on certain code paths,
and that method obfuscates token-suffixed fields by replacing them with `<{NAME_UPPER}>`
placeholders. So `eos_token=None` becomes the literal string `<EOS_TOKEN>`, which fails the
vocab-lookup the trainer then performs. On Gemma 4 the round-trip didn't trigger; on
Qwen2.5-VL it did, every time. The workaround (commit `376a4fb`) installs a runtime monkey-patch
on `TrainingArguments.to_dict` that reverses the obfuscation for any `*_token` field. Without
this fix, multi-model FT does not start.

After the patch, the multi-model results:

| Model family | n cats | Spearman ρ | p | Result |
|---|---:|---:|---:|:---:|
| Gemma 4 E4B-it | 24 | −0.778 | < 10⁻⁵ | rule transfers |
| Qwen3-VL-8B-Instruct | 5 | −1.000 | < 10⁻⁴ | rule transfers |
| Llama-3.2-11B-Vision-Instruct | 5 (Llama-specific lowest-ZS cats) | +0.200 | 0.63 | does not transfer |

Notes on the Llama outcome:

- The 5 cats for Llama were Llama's own lowest-ZS cats (cable, transistor, macaroni1,
  macaroni2, pcb3 — chosen after a 24-cat Llama ZS profile sweep). This is the rule's
  strongest possible test on Llama; even so, all lifts collapsed to [-0.005, +0.070] AUROC.
- For comparison, Qwen3 on similar-ZS cats produced lifts in [+0.197, +0.443] — 10× the
  magnitude.
- F1 = 0 on 4 of 5 Llama trained cells (the model predicts all-good post-FT).
- 25.7M trainable parameters (= 0.51% of base) reported correctly. The adapter trains. The
  parameters update. But inference behavior doesn't change.

Five testable root-cause hypotheses for the Llama failure (each requires its own diagnostic
sweep, deferred to a v0.3 follow-up):

1. `target_modules='all-linear'` may not patch Llama-3.2-Vision's visual decoder cleanly.
2. 3 epochs at lr=2e-4 may be undertrained for an 11B vision model.
3. Llama's chat template + the PASS/FAIL token-logit scoring may have a tokenization mismatch.
4. Gradient checkpointing under unsloth may behave differently on this architecture.
5. Some combination of the above.

We did **not** adjust the recipe per model to make Llama pass. The pre-registered structure
holds the recipe constant; the recipe-stability sub-study above showed the rule is recipe-stable
on Gemma. Per-model recipe tuning would have made cross-model comparison meaningless and broken
the held-out structure. The honest result is "rule transfers to 2 of 3 model families tested
under identical recipe; the 3rd reveals a model-architecture boundary worth investigating."

See commit `fbb2f96` for the multi-model close-out and the four expert-review JSONs documenting
the loop's decision points across the multi-model phase.

## What's still open

- **Llama-3.2-Vision recipe-vs-architecture diagnostic**: Llama with rank=16 / lr=4e-4 /
  epochs=10 on 2-3 cats to test whether the rule failure is recipe-specific. Targeted at v0.3.
- **Classical baselines**: WinCLIP+, PromptAD, Anomalib PatchCore on our same test splits.
  Published numbers are not directly comparable because splits differ.
- **Description-quality evaluation**: the VLM-unique axis that classical and ICL methods
  structurally cannot compete on. Not yet run.
- **Fourth model family**: Pixtral-12B (Mistral, similar era) would test whether the 2-of-3
  pattern generalizes. The 16 GB GPU rules out Llama 4 (MoE, ~109B total params even at 4-bit).
- **On-device latency**: we claim "consumer-GPU" but have not reported inference times on
  Jetson/M-series/edge devices.

These are follow-on work. The current snapshot ships what we have, with limitations flagged
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
