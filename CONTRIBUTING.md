# Contributing to fsvlm

fsvlm is an open-source research project exploring VLM data efficiency on industrial
anomaly detection. Contributions that improve the research — new dataset readers, more
rigorous benchmarks, honest failure-mode analysis, edge-deployment paths — are very welcome.

## Before you start

1. Read **[POSITIONING.md](POSITIONING.md)** first. It spells out what this project is, what it is
   not, what the benchmarks need to cover, and failure modes we explicitly want to avoid. Most
   discussion about scope, priorities, and "is this worth doing" is settled there.
2. Skim **[CLAUDE.md](CLAUDE.md)** for the architectural rules (SOLID layers, DI, ABCs, typing
   standards, testing pyramid). Non-negotiable.

## Quick start for contributors

```bash
git clone https://github.com/ahmadhasan2k8/fsvlm.git
cd fsvlm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,serve,watch,ui,annotate]"
pip install pytest pytest-cov hypothesis ruff mypy
pytest tests/unit -q
```

## Making a change

1. **One concern per PR.** New dataset reader → its own PR. New benchmark script → its own PR.
2. **Write tests.** Every file with logic gets unit tests in `tests/unit/`. See
   `tests/unit/test_visa_reader.py` for the pattern. CI enforces it.
3. **Type-hint everything.** `mypy --ignore-missing-imports fsvlm` should be clean.
4. **Format with ruff.** `ruff format .` then `ruff check .` before pushing.
5. **Preserve the append-only invariant** on `research/dataset_size_results.json`. Never
   rewrite historical rows — only append. See the Aether-style provenance described in
   `research/verdict.py`.
6. **Benchmark honesty.** If your change affects training or evaluation, re-run the affected
   sweep with ≥3 seeds and post the delta. No cherry-picked seeds.

## What we want

- **New backbones** — add Llama-3.2-Vision, Phi-Vision, InternVL, MiniCPM-V, future Gemma/Qwen
  releases. Walkthrough below.
- **Dataset readers** for well-respected public industrial defect datasets (MVTec LOCO, Real-IAD,
  Severstal, etc.). Follow the `FolderLabelReader` / `VisAReader` / `DeepPCBReader` pattern.
- **Baseline comparisons** — wire in YOLOv11, Anomalib PatchCore, GPT-4o zero-shot, etc. so the
  benchmark table is complete per POSITIONING.md.
- **Edge-deployment paths** — ONNX / TensorRT / CoreML export scripts with reproducible latency
  numbers on Jetson / M-series / consumer GPUs.
- **Honest failure cases** — where fsvlm underperforms classical methods. Documented in
  `docs/benchmarks.md`. This is more valuable than polishing wins.
- **Bug fixes with reproducers.** See [`docs/audit-trail.md`](docs/audit-trail.md) for the kinds
  of bugs we already caught and the hygiene checks we adopted; PRs that catch a ninth are
  highly valued.

## Adding a new backbone

The benchmark is designed to be backbone-swappable. Adding a new VLM family — Llama-3.2-Vision,
Phi-Vision, InternVL, MiniCPM-V, the next Gemma/Qwen release — should require **no changes to
existing files**. The Qwen3-VL-8B-Instruct integration is the worked example to mirror.

**1. Verify the backbone loads via unsloth's `FastVisionModel`.**

```python
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(
    "your-org/your-vlm",
    load_in_4bit=True,
    max_seq_length=1024,
)
```

If unsloth doesn't yet support your model architecture, the integration is upstream of fsvlm —
contribute the FastVisionModel patch to unsloth first.

**2. Pass the backbone via the `FSVLM_DEFAULT_MODEL` environment variable.** No code change to
`fsvlm/` is needed for the common case:

```bash
export FSVLM_DEFAULT_MODEL=your-org/your-vlm
fsvlm setup --check     # confirms the model loads + reports VRAM
```

`fsvlm/config.py` exposes `default_model` as a Pydantic `BaseSettings` field with prefix
`FSVLM_`, so the env var binds automatically. Both the train CLI and the zero-shot evaluation
path in `research/tiered_validation.py` honour it.

**3. Verify the tokenizer satisfies the PASS/FAIL single-token assertion** (Bug 7 in
[`docs/audit-trail.md`](docs/audit-trail.md)):

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("your-org/your-vlm")
assert len(tok.encode("PASS", add_special_tokens=False)) == 1
assert len(tok.encode("FAIL", add_special_tokens=False)) == 1
```

If your tokenizer splits PASS or FAIL into multiple subwords, you cannot use the default
single-token logit-ratio scorer — you'll need to either pick alternative single-token verdict
words and contribute a configurable scorer, or implement a multi-token scoring scheme via a
new `ScoreExtractor` ABC subclass.

**4. Run the audit hygiene checks against your backbone.** From `docs/audit-trail.md`:

```bash
# Bug 2 + 3 sanity: small-N training actually trains
FSVLM_DEFAULT_MODEL=your-org/your-vlm \
  fsvlm train --images research/datasets/quickstart_mvtec_capsule_n10/ --epochs 3
# Look for "optimizer_steps > 0" and "warmup_steps <= optimizer_steps / 2" in logs.

# Bug 4 sanity: prompts match
grep -A1 "Training schedule" ~/.fsvlm/logs/*.log | tail -10
```

**5. Run a smoke sweep** on a 4-cat focus set to confirm the recipe works and nothing has
silently degenerated:

```bash
FSVLM_DEFAULT_MODEL=your-org/your-vlm \
python -m research.dataset_size_sweep \
    --recipe-version v0.8-yourbackbone-smoke \
    --datasets mvtec --categories capsule transistor wood \
    --n-values 0 -1 --seeds 42
```

`-n 0` is zero-shot; `-n -1` is the full training pool. The 4-cat smoke produces 8 result rows
in `research/dataset_size_results.json`, each tagged with your `recipe_version` and the
backbone name.

**6. Open the PR with three things:**

- The smoke-sweep result rows committed to `research/dataset_size_results.json` (append-only).
- A one-paragraph backbone-card in `docs/backbones.md` (create the file if it doesn't exist):
  HuggingFace repo URL, parameter count, VRAM at 4-bit, recipe sensitivities you encountered,
  any audit-check failures and how you resolved them.
- A test in `tests/unit/test_backbones.py` that verifies the tokenizer assertion + that
  `FSVLM_DEFAULT_MODEL` env override propagates correctly. The test should NOT load the model
  weights (CI runs CPU-only); mock the model loader and only assert the configuration plumbing.

**Recipe sensitivities are real.** When we added Llama-3.2-Vision, the default LoRA recipe
(rank=8, lr=2e-4, 3 epochs) produced an adapter that trained but whose inference behavior
didn't change — the parameters updated but the model emitted nearly-identical token
distributions to the base. Five testable root-cause hypotheses are documented in the research
log. **Don't claim "tested 3 backbones" without verifying that the trained adapter actually
changes inference behavior on the test set.** Use the `dataset_size_sweep.py` sanity outputs:
if AUROC at N=full is identical to AUROC at N=0 within ±0.005, your training run is suspect.
Document this in the backbone card.

## What we do NOT want

Read POSITIONING.md's "What this project IS NOT" section. In particular:

- **No aerial, drone, wind, solar, or utility-infrastructure** examples, datasets, demos, or
  prompt templates. Out of project scope — hard line.
- **No medical imaging** datasets (regulated domain).
- **No framework bloat** (platform features, marketplaces, plugin systems beyond the current
  `LabelReader` / `ReportGenerator` ABCs) unless tied to a concrete research need.
- **No training-speed headlines.** Those are unsloth's wins, not ours — we credit them clearly.

## Reporting issues

Open a GitHub issue with:

- Exact `pip show fsvlm` version + OS + Python + GPU + VRAM
- The minimal reproducer (a few lines of code or the exact CLI command)
- Expected vs. actual behaviour
- Relevant log snippet from `~/.fsvlm/logs/`

## Code of conduct

Be kind. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License for contributors

By contributing code, documentation, benchmarks, or data to fsvlm, you agree your contribution
will be licensed under the project's [Apache License 2.0](LICENSE) — a permissive
open-source license that allows commercial use. Apache 2.0 includes an explicit patent
grant; if you can't distribute your contribution under Apache 2.0 (e.g. due to an
incompatible upstream license), please don't open the PR.

## Credit

Training kernels come from [unsloth](https://github.com/unslothai/unsloth). Please respect
their attribution requirements in any derivative work.

## Citation (academic norm, not a legal requirement)

If you publish work that uses fsvlm — or extends it into a new dataset reader, baseline, or
methodology — we'd appreciate a citation. See `CITATION.cff` for BibTeX. Apache 2.0 does
not legally mandate citation, but citation is how open research stays healthy.
