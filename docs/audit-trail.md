# The eight-bug methodology audit

This document is the public companion to commit [`6c8cc9e`](https://github.com/ahmadhasan2k8/fsvlm/commit/6c8cc9e). It describes **ten implementation pitfalls** we found in our own benchmark pipeline and the fixes we applied. (Bugs 1–8 are the original audit; Bug 9 surfaced when integrating Llama-3.2-Vision as a third backbone; Pitfall #10 surfaced during the 50-epoch validation expansion and was discovered via cross-recipe checking on 2026-05-26.) **Every pitfall listed here is the kind of mistake any from-scratch implementer of a few-shot generative-VLM industrial AD pipeline would plausibly hit.** We document them as transferable hygiene checks.

If you fork `fsvlm` or build a similar pipeline, replicate the checks at the bottom of each bug section — they're cheap, generic, and would have caught these issues in our own code months earlier than they did.

> **Status.** Bugs 1-8 fixed in commit `6c8cc9e`; Bug 9 fix shipped behind `FSVLM_RESPONSE_STYLE_AWARE=1` opt-in; Pitfall #10 is a methodology pitfall rather than a fixable code bug, but the post-refactor trainer (`a7129dc` onward) honors `FSVLM_DISABLE_CHECKPOINTS=1` correctly so the policy is now controllable. Post-fix recipes: `v0.7-zs-logit-only*`, `v0.8-fixed-pipeline*`, `v0.4-longepoch-validation*` (best-eval-loss policy), `v0.5-2d-sweep-*` (last-epoch policy), `v0.5-pitfall10-*` (pre-registered Pitfall #10 controls). Pre-fix data (`v0.3-tier-a`, `v0.5-tier-a-qwen3`, `v0.6-*`) is retained in `research/dataset_size_results.json` for transparency. Any AUROC numbers in older documentation that reference pre-fix recipes should be treated as superseded.

## How the audit was conducted

A multi-pass code review of the pipeline, run pre-conclusion (i.e., before we drew empirical conclusions from the data). Each pass focused on a different lens:

1. **Training-pipeline pass** — examined `fsvlm/agents/training_agent.py` for hyperparameter scaling failures at small N.
2. **Data-pipeline pass** — examined the train/test split conventions in `research/dataset_size_sweep.py` and the WinCLIP+ baseline in `research/baselines/run_winclip.py`.
3. **Code-symmetry pass** — diff'd the zero-shot and trained-path evaluation code looking for asymmetric branches.

The training-pipeline pass produced six time-stamped JSON reports committed to `research/expert_reviews/`. The data-pipeline and code-symmetry passes' findings were synthesized into the audit-trail commit message rather than as separate JSON artifacts. The 8 bugs below are the union of issues raised across all three passes, ranked by severity.

---

## Bug 1: Scoring asymmetry between zero-shot and trained eval paths (severity: HIGH)

### Symptom
The zero-shot evaluation path used a keyword cascade:

```python
# Pre-fix _run_base_model_inference
if first_word == "FAIL":
    score = 0.9
elif first_word == "PASS":
    score = 0.1
else:
    score = prob_fail_logit  # logit-only fallback only when neither word
```

The trained-path evaluation used logit-only directly:

```python
# Pre-fix _run_adapter_inference
score = prob_fail_logit
```

On categories where the base model emits "PASS" verbatim for both good and defect images at zero-shot (because it's uncertain), the cascade collapses scores to 0.1 for both classes, destroying AUROC ranking.

### Quantified impact
On capsule (mvtec), Gemma 4 E4B-it:

| | AUROC |
|---|---:|
| Pre-fix ZS (cascade) | 0.471 |
| Post-fix ZS (logit-only) | **0.859** |
| Trained N=full (already logit-only) | 0.872 |

The pre-fix story was "fine-tuning lifts capsule from 0.47 → 0.87, a +0.40 gain." The post-fix story is "fine-tuning lifts capsule from 0.86 → 0.87, a +0.01 gain." **The +0.39 attributed to fine-tuning was scoring-asymmetric, not real.**

### Fix
Default scorer is now logit-only on both paths. Cascade is opt-in via `FSVLM_ENABLE_CASCADE=1` for backward-compatibility with pre-audit numbers. See [`research/tiered_validation.py:_run_base_model_inference`](https://github.com/ahmadhasan2k8/fsvlm/blob/main/research/tiered_validation.py).

### Generalizable hygiene check
Run the same scorer on both ZS and trained eval paths. Diff the implementations; **any branch that exists on one path and not the other is a candidate asymmetry.** Add the check to your pre-merge protocol if you have one.

---

## Bug 2: Hardcoded warmup_steps starves small-N training (severity: HIGH)

### Symptom
Pre-fix `fsvlm/agents/training_agent.py` had:

```python
warmup_steps = max(1, int(0.05 * 150))  # = 7, regardless of dataset size
```

With `grad_accum=8`, `batch=1`, `epochs=3`: at N=10, `total_optimizer_steps = (10 * 0.8 * 3) / (1 * 8) = 3`. **Warmup steps (7) > total optimizer steps (3)** — the learning rate never left the warmup ramp.

### Quantified impact
Effective LR stayed at near-0 for all N ≤ ~60. All `v0.3-tier-a` and `v0.5-tier-a-qwen3` small-N training cells trained with degenerate LR. Reported "few-shot fine-tuning lift" in those cells was largely measuring noise around the base model.

### Fix
```python
total_optimizer_steps = max(1, (n_train * num_epochs) // (batch * eff_grad_accum))
warmup_steps = max(1, int(warmup_ratio * total_optimizer_steps))
```

See [`fsvlm/agents/training_agent.py`](https://github.com/ahmadhasan2k8/fsvlm/blob/main/fsvlm/agents/training_agent.py).

### Generalizable hygiene check
Log `total_optimizer_steps` at training start; assert `warmup_steps <= total_optimizer_steps / 2`. If your trainer doesn't expose this, compute it yourself from `len(dataset)`, `epochs`, `batch_size`, and `grad_accum`.

---

## Bug 3: Gradient accumulation produces 0 optimizer steps at N=1 (severity: HIGH)

### Symptom
With `grad_accum=8`, `n_train=1`, `epochs=3`: total forward passes = 3, optimizer steps = `3 // 8 = 0`. **The "N=1 trained adapter" had untouched LoRA weights** — it was identical to the base model with an identity adapter attached.

### Quantified impact
All `v0.6-n1-ablation` cells (4 categories × 3 seeds × 2 model families) measured the base model with an identity adapter, not a fine-tuned one. Any "N=1 fine-tuning lift" story drawn from those rows was vacuous.

### Fix
```python
n_train = max(1, len(train_dataset))
eff_grad_accum = max(1, min(configured_grad_accum, n_train // batch_size))
```

### Generalizable hygiene check
After training, log `len(adapter_state_dict_diff_from_base)`; if zero, raise. Or simpler: log `optimizer_steps` and assert `> 0`.

---

## Bug 4: Train/eval prompt mismatch (severity: MEDIUM)

### Symptom
Training used the generic `inspection_prompt` ("You are a visual quality inspector..."). Evaluation used the category-specific `defect_prompt` ("Examine this capsule. Is it a normal..."). The trained adapter saw a prompt at training that it never saw at inference.

### Quantified impact
Suppressed measured fine-tune lift uniformly across all `v0.3-tier-a` / `v0.5-tier-a-qwen3` trained-path cells. Magnitude is hard to quantify per-cell because every cell was affected; the post-fix v0.8 numbers are the corrected baseline.

### Fix
Plumbed `--prompt` through `fsvlm.cli train` and the sweep harness so the per-category `defect_prompt` is used at both training and inference.

### Generalizable hygiene check
Log the exact prompt string used at training and at evaluation; assert string equality before reporting metrics.

---

## Bug 5: MVTec test-split asymmetry vs WinCLIP+ baseline (severity: MEDIUM)

### Symptom
`MVTecAdapter.test_set()` halves defects per subtype (correct convention). The initial WinCLIP+ runner returned ALL defects per subtype. WinCLIP+ tested on a ~1.6× larger anomaly pool than fsvlm did on the same categories.

### Quantified impact
13 of 24 matched-shot rows in the comparison table were apples-to-oranges. The WinCLIP+ AUROC numbers were systematically more pessimistic than fsvlm's because they included the harder defect-subtype samples.

### Fix
Mirrored halving in `research/baselines/run_winclip.py:mvtec_test_split`.

### Generalizable hygiene check
When integrating an external baseline, hash the test-set image lists and assert equality with your own. fsvlm publishes per-category SHA-256 hashes in [`research/baselines/test_set_hashes.json`](https://github.com/ahmadhasan2k8/fsvlm/blob/main/research/baselines/test_set_hashes.json).

---

## Bug 6: MVTec train pool overlaps test (severity: MEDIUM)

### Symptom
Pre-fix `MVTecAdapter.train_pool()` returned ALL defects without halving. At large N, sampling could draw test images into training — direct test-set contamination.

### Fix
`train_pool` returns first-half-per-subtype only. The other half stays in the test set.

### Generalizable hygiene check
At sweep start, hash the training image list and assert disjoint from test-set hashes. fsvlm does this via the `test_set_hashes.json` artifact.

---

## Bug 7: PASS/FAIL token-id [0] indexing (severity: MEDIUM, latent)

### Symptom
```python
pass_id = tokenizer.encode("PASS", add_special_tokens=False)[0]
fail_id = tokenizer.encode("FAIL", add_special_tokens=False)[0]
```

If a tokenizer splits "PASS" or "FAIL" into multiple subwords, `[0]` silently picks the first subword and the logit-ratio scoring is wrong but doesn't crash. Currently safe for Gemma 4's tokenizer, which yields single tokens for both. Would be **silently wrong** for any backbone whose tokenizer behaves differently.

### Fix
```python
_pass_ids = tokenizer.encode("PASS", add_special_tokens=False)
_fail_ids = tokenizer.encode("FAIL", add_special_tokens=False)
assert len(_pass_ids) == 1, f"Tokenizer splits 'PASS' into {len(_pass_ids)} subwords {_pass_ids}; the logit-ratio scorer needs single tokens."
assert len(_fail_ids) == 1, f"Tokenizer splits 'FAIL' into {len(_fail_ids)} subwords {_fail_ids}; the logit-ratio scorer needs single tokens."
pass_id = _pass_ids[0]
fail_id = _fail_ids[0]
```

Loud failure, not silent miscoring.

### Generalizable hygiene check
For any vocabulary-token-based scoring scheme, assert single-token tokenization at startup for every backbone you support.

---

## Bug 8: Cascade scoring as default (severity: MEDIUM)

### Symptom
Cascade scoring (Bug 1) was the default; logit-only required `FSVLM_DISABLE_CASCADE=1`. Anyone running fsvlm without reading the source code reproduced Bug 1 unknowingly.

### Fix
Cascade is now opt-in via `FSVLM_ENABLE_CASCADE=1`. Logit-only is the default. Default scorer choice is documented in the inline docstring of [`_run_base_model_inference`](https://github.com/ahmadhasan2k8/fsvlm/blob/main/research/tiered_validation.py).

### Generalizable hygiene check
Defaults matter more than configurability. If you ever find yourself documenting "the safe behavior is opt-in via env var," flip the default and make the unsafe behavior opt-in instead.

---

## Bug 9: Position-0 logit-ratio scorer assumes PASS/FAIL at first token (severity: HIGH for some backbones; surfaced 2026-05-03)

### Symptom
The single-token PASS/FAIL logit-ratio scorer reads probabilities at **position 0** of the model's response — the assumption being the model's first generated token is one of {PASS, FAIL} or close enough that the relative ratio carries the verdict signal.

This holds for backbones whose RLHF training accommodates terse compliance (Gemma 4 E4B-it, Qwen3-VL-8B-Instruct: both reliably emit PASS or FAIL as token 0 under our prompt). It **does not hold** for backbones with a different response style — Llama-3.2-11B-Vision-Instruct prefixes responses with words like "THE" (e.g., "The image shows...") or markdown headers like "**ANALYSIS" (e.g., "**Analysis:**\n..."). At position 0, the logits for PASS/FAIL are tiny (1-15% range) and effectively meaningless noise.

Bug 7's assertion (`len(tokenizer.encode("PASS")) == 1`) catches *tokenization* mismatch but does **not** catch *response-style* mismatch. The assertion passes for Llama-3.2; the scoring still degrades.

### Quantified impact
Llama-3.2-11B-Vision under v0.8-fixed-pipeline-llama32 zero-shot, with `first='THE'` or `first='**ANALYSIS'` consistently across the test set:

| Cat | Llama ZS | Gemma ZS | Qwen3 ZS | Likely artifact |
|---|---:|---:|---:|---|
| mvtec/capsule | 0.630 | 0.859 | 0.968 | Yes — first token "THE" |
| mvtec/transistor | 0.574 | 0.712 | 0.669 | Yes |
| mvtec/wood | **0.997** | 0.995 | 1.000 | No — saturated cat preserves rank even with degraded scorer |
| visa/capsules | 0.707 | 0.633 | 0.663 | Mixed — non-trivial |
| visa/chewinggum | 0.913 | 0.978 | 0.981 | Yes — drift below saturation |
| visa/pcb1 | 0.714 | 0.750 | 0.546 | Mixed — non-trivial |

The pattern is clearest on lift-bearing cats (capsule, transistor, chewinggum) where Llama trails both other backbones by a wide margin. On saturated cats the underlying class separation is strong enough that even degraded position-0 scoring preserves the ranking.

### Fix (proposed; not yet implemented)

Search the model's response for the FIRST occurrence of PASS or FAIL within the first N generated tokens (configurable; default 32). Compute the logit ratio at that token position rather than at position 0. If neither PASS nor FAIL appears in the search window, fall back to position-0 logits with a warning logged.

Pseudocode:
```python
def score_with_response_style_aware_logit(generated_token_ids, logits_per_step, search_window=32):
    for i in range(min(search_window, len(generated_token_ids))):
        if generated_token_ids[i] in (PASS_TOKEN, FAIL_TOKEN):
            p_pass = softmax(logits_per_step[i])[PASS_TOKEN]
            p_fail = softmax(logits_per_step[i])[FAIL_TOKEN]
            return p_fail / (p_pass + p_fail)
    # Fallback: position 0, log warning
    logger.warning(f"PASS/FAIL not found in first {search_window} tokens; falling back to position 0")
    p_pass = softmax(logits_per_step[0])[PASS_TOKEN]
    p_fail = softmax(logits_per_step[0])[FAIL_TOKEN]
    return p_fail / (p_pass + p_fail)
```

This requires capturing per-step logits during generation (currently we only capture position 0 for efficiency). Implementation deferred to v0.3.

### Generalizable hygiene check
For any vocabulary-token-based verdict scorer:
1. Log the first generated token (as a string, not just an id) for the first 5-10 test samples per backbone.
2. If the first token is reliably one of your verdict tokens → safe.
3. If it's a punctuation, article, or markdown delimiter → your scorer is being applied to noise. Either change the prompt to force terse output, or implement response-style-aware position lookup.

The fsvlm sweep harness now logs `first='<word>'` next to each test sample's score for exactly this kind of post-hoc detection.

### Why this wasn't caught earlier
Gemma 4 E4B-it (our primary development backbone) and Qwen3-VL-8B-Instruct (cross-family) both emit PASS/FAIL as token 0 reliably under our prompt. The bug is **silent** for them. It only surfaces when adding a third backbone whose RLHF training emphasizes a different response style. Bug 9 was caught precisely because the audit framework demands integrating new backbones — running fsvlm on Llama is what surfaced it. **The audit framework's value is exactly this: it catches what you didn't know to look for.**

### Severity rating
- **HIGH for response-style-incompatible backbones** (Llama-3.2-Vision and likely others with verbose/markdown-styled outputs)
- **NULL for response-style-compatible backbones** (Gemma 4, Qwen3-VL — already correct)

The fix is *additive*: response-style-aware position lookup degrades gracefully to position-0 scoring when PASS/FAIL aren't found in the search window. Implementing it doesn't change Gemma or Qwen3 numbers; it only repairs scoring for backbones that today are silently miscored.

---

## Pitfall #10: Checkpointing policy at long-epoch × small-N changes reported AUROC on lift-bearing cats and silently overrode a launch-script env var in our pre-refactor pipeline (severity: HIGH on lift-bearing cats at long epoch budgets; surfaced 2026-05-26)

**Symptom.** When running 50-epoch validation at N=10 on 27 cats × 5 seeds × 2 backbones, per-epoch checkpoint accumulation (~13.5 TB) exceeded `/tmp` capacity by orders of magnitude. The pragmatic workaround was an environment-variable opt-out (`FSVLM_DISABLE_CHECKPOINTS=1`) intended to switch `save_strategy` from `"epoch"` to `"no"` and force `load_best_model_at_end=False` (last-epoch weights). **A subsequent cross-recipe check (2026-05-26) revealed that the pre-refactor code path silently ignored this env var** — `save_strategy="epoch"` was hardcoded and `load_best_model_at_end` defaulted to `True` when an eval split was present. The Phase 1 27-category sweep that produced the headline 50ep × N=10 multi-seed values was therefore actually run under **best-eval-loss policy** rather than the last-epoch policy the launch script intended.

**Impact: jointly category-dependent and epoch-budget-dependent.** A focused 5-seed × 50-epoch × N=10 × pcb1 × Gemma rerun under explicit best-eval-loss policy lands at 0.6194 ± 0.0383 vs the last-epoch result 0.6190 ± 0.0210 — **Δμ = +0.0004 AUROC, essentially zero on pcb1 (a no-lift cat with no per-epoch variation to select between)**. A subsequent same-cell cross-recipe check on **capsule** × 50ep × N=10 × {seeds 42, 1337, 7} shows mean 0.876 under best-eval-loss vs 0.823 under last-epoch on identical cells, **Δμ = +0.053 AUROC — roughly 8× the seed-noise band** on a lift-bearing cat with meaningful per-epoch variation. A pre-registered same-policy control on capsule × **3-epoch** × N=full × 5 seeds × Gemma under last-epoch policy (`research/queue.json:pass8-pitfall10-capsule-policy-control`, committed before execution at `8517b57` 2026-05-26 21:05 PDT, run 2026-05-27) lands at 0.882 ± 0.017 vs Phase 1's 0.891 — Δμ = +0.009, within noise. The per-(cat, epoch-budget) magnitude table:

| (cat, epochs × N) | Δμ best-eval-loss − last-epoch | Mechanism |
|---|---:|:---|
| capsule, 3ep × N=full | +0.009 (within noise) | short budget — no per-epoch variation; best ≈ last |
| capsule, 50ep × N=10 | +0.053 | lift-bearing + long budget — best-of selects above last-epoch |
| transistor, 50ep × N=10 | +0.044 | same mechanism as capsule (from 2D sweep cross-recipe) |
| bottle, 50ep × N=10 | +0.000 | plateaus quickly — no per-epoch lift to select |
| pcb1, 50ep × N=10 | +0.0004 | catastrophic; no per-epoch variation |

**Fix.** Post-refactor (`a7129dc` onward) the trainer respects `FSVLM_DISABLE_CHECKPOINTS=1` and produces last-epoch weights when set. The pre-fix Phase 1 best-eval-loss values are retained for transparency under `v0.4-longepoch-validation*`; the post-fix last-epoch values live under `v0.5-2d-sweep-*`, `v0.5-pitfall10-capsule-control`, and chained successors. Any cross-recipe comparison requires an explicit policy label.

**Hygiene check.** When an env var or CLI flag is intended to switch a trainer behavior, write an assertion that fires loudly if the resulting `save_strategy` / `load_best_model_at_end` doesn't match the intended policy. The silent override of our `FSVLM_DISABLE_CHECKPOINTS=1` flag for ~3 weeks of experiments is the kind of pitfall that hides until cross-recipe comparison surfaces it.

**Operational implication.** Disk-management workarounds for long training can quietly change which model checkpoint your evaluation reports on; the per-(cat × epoch-budget) magnitude varies from 0 to ~0.05 AUROC within our data and **can flip individual-cat substitutability conclusions** (on capsule, the substitutability claim 50ep × N=10 ≈ 3ep × N=full survives under best-eval-loss but falsifies under last-epoch). Practitioners running long-epoch sweeps with disk pressure should label every cell with its checkpoint policy and not assume the recipe detail is small.

---

## Bonus: Llama-3.2-Vision VRAM ceiling at N=full on 16GB (not a bug; documented constraint)

In addition to Bug 9, the Llama re-run surfaced a hardware-recipe boundary: **all 6 N=full cells OOM'd** at ~14.3 GiB allocation on a 16 GiB GPU under our LoRA recipe (rank=8, batch=1, grad_accum=8, max_seq=1024). The 11B parameter count + activation memory + grad accumulation buffers + image-tokenization memory exceed the available VRAM headroom for training pools larger than ~50 examples per category.

This is not a bug — it's a real constraint of running an 11B Vision model at N=full on 16 GiB consumer hardware with our recipe. Workarounds (in order of effort):
1. Reduce `grad_accum` to 4 or 2 (changes the effective batch size; needs hyperparameter re-validation)
2. Reduce `max_seq_length` from 1024 to 512 (caps prompt+response length)
3. Switch to `bitsandbytes` 4-bit + paged Adam optimizer (reduces optimizer-state memory)
4. Use a 24+ GiB GPU

The `docs/backbones/llama-3.2-vision.md` card (to be added when the backbone-cards directory ships) will document this constraint with reproducer instructions and the workarounds above.

---

## What changed in the released data

The post-fix data lives under these recipe versions in `research/dataset_size_results.json`:

| Recipe version | What it contains |
|---|---|
| `v0.7-zs-logit-only` | Gemma 4 zero-shot baseline using logit-only scoring |
| `v0.7-zs-logit-only-qwen3` | Qwen3-VL zero-shot baseline using logit-only scoring |
| `v0.8-fixed-pipeline` | Gemma 4 fine-tuned at multiple N values, post all 8 fixes |
| `v0.8-fixed-pipeline-qwen3` | Qwen3-VL fine-tuned at multiple N values, post all 8 fixes |
| `v0.8-fixed-pipeline-full` | Gemma 4 N=full (whole training pool) cells |
| `v0.8-bisection` | Gemma 4 at N=30/60/100 (the FT-effectiveness "knee" region) |
| `v0.8-rank16-fullN` / `v0.8-rank32-fullN` | LoRA rank ablation at N=full |

The pre-fix data (`v0.3-tier-a`, `v0.5-tier-a-qwen3`, `v0.6-n1-ablation*`) is retained alongside the fixes for transparency. Use `recipe_version` as a filter in any analysis script.

---

## Why we publish this

A benchmark is only as good as the symmetry of the comparisons it lets you make. Eight bugs in our own pipeline produced a measurement environment where the comparisons weren't symmetric, and a non-trivial fraction of "few-shot fine-tuning lift" was an artifact rather than a real effect. We caught them by running our own benchmark against itself in a multi-pass code review with three different lenses. We're publishing them because:

1. **Other implementers will hit these.** The bugs are not exotic. Hardcoded warmup, mismatched prompts, opt-out unsafe defaults — all common-enough patterns that flagging them as a checklist is more useful than not.
2. **A benchmark that hides its own bugs is not a benchmark.** The point of a benchmark is to make comparisons fair. Documenting where ours wasn't fair, and how we fixed it, is part of the contract.
3. **Pre-fix data is in the public results JSON.** Anyone who wants to verify the impact of a specific bug can filter `dataset_size_results.json` by `recipe_version` and reproduce the deltas.

If you find a ninth bug, please open an issue or a PR. Audit contributions are highly welcome.
