# fsvlm

**A few-shot VLM benchmarker that fits on a 16 GB laptop GPU. Train an industrial defect
detector with as few as 2 labeled images, on your own hardware, with git-SHA + recipe-version
provenance on v0.1+ result rows and a 15-skill catalog of runtime-agnostic playbooks.**

[![CI](https://github.com/ahmadhasan2k8/fsvlm/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmadhasan2k8/fsvlm/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](pyproject.toml)
[![Built on unsloth](https://img.shields.io/badge/built_on-unsloth-purple.svg)](https://github.com/unslothai/unsloth)

---

## What the tool actually shows

![AUROC vs N labeled examples on three categories](docs/figures/n_shot_curves.png)

*Three categories swept end-to-end at N ∈ {0, 2, 3, 5, 10, 20, 30, 40, 60, 100} labeled
examples, 3 seeds per cell, on a single 16 GB laptop GPU. Recipe: Gemma 4 E4B-it via QLoRA,
v0.1 score-extractor cascade. Computed against the v0.1-cascade zero-shot baseline (the
baseline used by the chart), N = 2 captures **97.8 % of the lift on hazelnut**
(0.821 → 0.941) and **98.2 % on candle** (0.680 → 0.953). DeepPCB has a different shape
("delayed knee"): the base VLM is already at-ceiling at zero-shot, with small monotonic lift
only at N ≥ 60. Per-category variance and failure modes are documented honestly in
[docs/benchmarks.md § "When fsvlm is not the right choice"](docs/benchmarks.md).*

---

## A measurement finding worth knowing about

![Score-extractor cascade recovers AUROC](docs/figures/extractor_audit.png)

The naive score extractor used implicitly by much of the generative-VLM anomaly-detection
literature — *check the first generated word; if it's not "PASS" or "FAIL", default to a
constant* — silently penalises any image where the model emits explanatory prose first. fsvlm's
v0.1 default cascade reads the underlying token-logit probability instead. **Same model. Same
images. +0.15 to +0.34 AUROC on three of four categories tested.** This is a free improvement
for anyone running a generative VLM on this kind of task.

Details and the methodology audit are in [docs/research-log.md](docs/research-log.md).

---

## Fine-tune vs in-context learning at extreme few-shot

![Fine-tune vs in-context-learning at N=2](docs/figures/ft_vs_icl.png)

Same Gemma 4, same fixed test split, same prompts. At N = 2, fine-tuning wins on all three
categories. At N = 8 (not shown), in-context learning catches up and overtakes fine-tuning on
hazelnut — the picture is richer than "fine-tune always wins". Honest framing: **FT wins at
extreme-few-shot (N ≤ 2); ICL catches up by N ≈ 8 on categories where the base VLM is already
near ceiling.** See [docs/research-log.md](docs/research-log.md) for the deeper analysis.

---

## Why a researcher might care

| | Classical (Anomalib) | Frozen-CLIP (WinCLIP+, PromptAD) | Generative-VLM (AnomalyGPT, Anomaly-OV, Triad) | **fsvlm** |
|---|:---:|:---:|:---:|:---:|
| Per-category N-shot AUROC published | not the focus | aggregated, paper-only | aggregated, paper-only | **per-cell, append-only log** |
| Single consumer GPU | ✅ | ✅ | ❌ (cluster) | **✅ 16 GB laptop** |
| Open code, Apache 2.0 | ✅ | partial | partial | **✅** |
| Score-extractor disclosed as methodology axis | n/a | n/a | ❌ | **✅ v0.1 cascade** |
| Pre-registered taxonomy with frozen timestamp | ❌ | ❌ | ❌ | **✅** |
| Append-only result log (git SHA + recipe version per row) | ❌ | ❌ | ❌ | **✅ for v0.1+ rows** |
| Autoresearch loop pattern + reference drivers | n/a | n/a | n/a | **✅ docs + bash drivers** |
| Skill catalog with declarative eval JSONs (Anthropic schema) | n/a | n/a | n/a | **✅ 15 skills + reference harness** |

This table compares fsvlm's *measurement infrastructure*, not head-to-head AUROC. Numbers from
WinCLIP+ / PromptAD / AnomalyGPT / Triad come from their published papers and are **not yet
rerun on our splits**. Direct head-to-head AUROC comparison on identical splits is on the v0.2
roadmap (Anomalib PatchCore first). Several "✅" rows above describe dimensions the cited
projects weren't designed for — they're noted to be transparent about what fsvlm adds, not to
imply the others are deficient at their own goals.

---

## Why a practitioner might care

You have a folder of images. Half are good parts, half have defects. You want a detector. fsvlm
gives you, in three commands, on your own GPU, in under an hour:

- A trained adapter (~80 MB) that runs on the same 16 GB laptop GPU
- A validation report (HTML + JSON) showing AUROC, F1, confusion matrix, and a failure gallery
- An inference surface — single image, batch folder, drop-folder watch mode, FastAPI REST,
  Gradio UI — all sharing the same adapter and producing the same JSON

No cloud round-trip, no per-month subscription, no telemetry by default.

---

## Install and try in 30 seconds

```bash
pip install git+https://github.com/ahmadhasan2k8/fsvlm
fsvlm setup --check                                       # detect GPU, verify deps
python examples/quickstart/make_dataset.py                # 20 synthetic images
python examples/quickstart/check_pipeline.py              # 4 PASS checks, no GPU needed
```

If those four checks pass, your install is healthy. Move on to real training.

```bash
fsvlm setup                                                # download Gemma 4 E4B-it (4-bit)
fsvlm train --images ./my-data/                            # good/ + defect/ subdirs
fsvlm inspect new-image.jpg --adapter ~/.fsvlm/adapters/latest/
```

Or open the Gradio app for SAM-assisted point-prompt annotation + training:

```bash
fsvlm ui
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Interface layer    CLI · Gradio UI · FastAPI · watch-mode      │
├─────────────────────────────────────────────────────────────────┤
│  Agent layer                                                    │
│    Orchestrator · DataAgent · TrainingAgent · ValidationAgent   │
│    InspectorAgent · FeedbackAgent · AnnotationAgent · SAM       │
├─────────────────────────────────────────────────────────────────┤
│  Domain layer       pure Python — types, schemas, ABCs, config  │
└─────────────────────────────────────────────────────────────────┘
```

Three layers, dependencies inward only. The Domain layer can be imported in <1 s with no GPU
dependencies pulled in (CI verifies `import fsvlm` is torch-free). Heavy imports happen inside
the agents that need them.

Five extension points use abstract base classes registered through a decorator. Adding a new
implementation is **one new file**; no existing file changes.

| ABC | What it lets you add | v0.1 ships with |
|-----|----------------------|-----------------|
| `ModelBackend`    | A new VLM family — Qwen-VL, LLaVA, Phi-Vision, InternVL, MiniCPM-V, future Gemmas | Gemma 4 E4B-it (4-bit via unsloth) |
| `LabelReader`     | A new dataset / label format | folder, CSV, JSON, VisA `1cls.csv`, DeepPCB |
| `ScoreExtractor`  | A new way to turn VLM output into an anomaly score | v0.1 token-logit cascade + legacy keyword-match |
| `TrainingBackend` | A different fine-tuning library | unsloth (default and recommended) |
| `ReportGenerator` | A new validation-report output format | HTML (Jinja2), JSON |

One-PR recipes for each are in [docs/autoresearch.md § 6](docs/autoresearch.md).

---

## The skill catalog — 15 playbooks for the "0 → paper" path

Every routine workflow ships as a [Markdown playbook with YAML frontmatter](skills/README.md)
declaring `name`, `description`, `inputs`, `eval_artifact`, `pass_criteria`, and `escalation`.
**Two kinds of skills, with a sharp difference in how they run:**

### ✅ Procedural skills (10) — directly runnable from shell

Wrap a single underlying CLI command or script. Invoke from cron / Make / your own shell
loop / CI — no agent runtime required:

```bash
bash skills/_run.sh setup --check
bash skills/_run.sh train --images ./my-data/ --epochs 3
bash skills/_run.sh sweep --datasets mvtec --categories hazelnut --n-values 0 30 --seeds 42
bash skills/_run.sh verdict --results research/dataset_size_results.json --write
bash skills/_run.sh plot  --results research/dataset_size_results.json --output docs/figures/
```

The 10 procedural skills: `setup`, `train`, `inspect`, `validate`, `serve`, `sweep`,
`verdict`, `tiered-eval`, `plot`, `meta-eval`. The `_run.sh` dispatcher is real, ships in
v0.1, and resolves your local Python interpreter automatically.

### Using these with Claude Code

The repo ships a public [`.claude/`](.claude/) directory that wires the 15 skills as
slash-commands and installs a safety hook. Clone, `cd` in, run `claude`, and:

- All 15 skills are auto-discovered as `/setup`, `/train`, `/sweep`, `/verdict`, etc.
- The PreToolUse hook at [`.claude/hooks/block-dangerous.sh`](.claude/hooks/block-dangerous.sh)
  refuses to run `rm -rf`, `git push --force`, `git reset --hard`, deletion of training
  images, dropping database tables, and ~12 other common foot-gun patterns. Trivial to copy
  into other repos.

See [`.claude/README.md`](.claude/README.md) for the full layout and what's deliberately
NOT shipped (project-specific sub-agents).

### 🤝 Orchestrator skills (5) — semi-autonomous, need a runtime

Make conditional decisions ("if verdict says X, dispatch /sweep with these params; if expert
review says halt, halt"). They're **documented protocols, not autonomous runners** —
something has to interpret the procedure markdown, decide when to stop / pause / ask, and
call the procedural sub-skills via `skills/_run.sh`. Three execution paths:

1. **Claude Code** (the primary intended runtime) — drop `skills/<name>.md` into
   `~/.claude/skills/` and invoke as `/<name>`. The agent reads the procedure, makes the
   conditional calls, asks you when it hits a fork it can't resolve.
2. **Another agent runtime** (OpenAI Agents SDK, CrewAI, your own loop driver) — register
   the skill's procedure as a tool whose body executes the steps, calling
   `skills/_run.sh <sub-skill>` for each procedural sub-skill it dispatches.
3. **Manual** — read the skill markdown and step through it yourself, calling
   `skills/_run.sh` for sub-skills and making the decision-point judgment calls by hand.

Direct shell invocation (`skills/_run.sh autoresearch`) returns an explicit "needs a
runtime" error pointing at the three options above.

The 5 orchestrator skills: `autoresearch`, `improve-skill`, `improve-skills-auto`,
`expert-review`, `debug`.

### Eval JSONs + meta-loop

Each skill ships with a sibling [`skills/evals/<name>.eval.json`](skills/evals/) declaring
(prompt, expected trigger, assertions) following the
[Anthropic skill-creator eval schema](https://github.com/anthropics/skills). The portable
eval harness at `scripts/run_skill_eval.py` grades skills against the Anthropic Messages
API (needs `ANTHROPIC_API_KEY`).

> **Honest scope of v0.1.** The procedural skills + `_run.sh` are turnkey. The orchestrator
> skills are documented protocols + eval JSONs that need a runtime. The skill-self-improvement
> meta-loop (`/improve-skill`, `/improve-skills-auto`) is documented end-to-end but has no
> committed evidence of having actually self-improved a shipped skill — treat it as a
> protocol, not a live capability. The harness's `claude-code` and `openai-sdk` runtime
> adapters are stubs that describe the wiring but need per-stack implementation. PRs that
> turn any of these into turnkey runners are welcome.

> The two distinctive contributions on top of [Karpathy's autoresearch loop](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/)
> are **expert-agent consultation as a separate step from the verdict classifier**
> (anti-Goodhart guard) and **skills with declarative PASS/FAIL self-eval YAML frontmatter**
> (the contract that lets the loop iterate without a human in every loop body). See
> [docs/autoresearch.md § 4.5](docs/autoresearch.md) for the full "his vs ours" boundary.

---

## Reproducing every number on this page

**Fastest path — verify the headline numbers from the committed result rows in 1 second:**

```bash
python scripts/verify_readme_numbers.py
```

This re-derives every headline percentage and AUROC value from
`research/dataset_size_results.json` with the explicit recipe-version filter shown next to
each number. If your output diverges from the README, file an issue.

**Full reproduction — re-run the actual sweep that produced those rows (requires GPU):**

```bash
# Datasets (MVTec AD requires email registration; place under research/mvtec_data/)
bash research/datasets/download_visa.sh
bash research/datasets/download_deeppcb.sh

# AUROC-vs-N curves (Section 1 + Section 3 above)
bash research/run_sweep.sh \
  --datasets mvtec visa deeppcb \
  --categories hazelnut candle pcb \
  --n-values 0 2 3 5 10 20 30 40 60 100 \
  --seeds 42 1337 7 \
  --epochs 3

# Score-extractor audit (Section 2)
bash research/run_sweep.sh \
  --datasets mvtec visa deeppcb \
  --categories hazelnut metal_nut candle pcb \
  --n-values 0 \
  --seeds 42 \
  --extractor v0.1

# Verdict classifier marks each cell keep / discard / noop / new_baseline
python research/verdict.py --results research/dataset_size_results.json --write
```

Rows in [`research/dataset_size_results.json`](research/dataset_size_results.json) carry
`git_hash` + `recipe_version` from v0.1 onwards (89 % of the 159 logged rows; earlier
exploratory rows from before the provenance contract was finalised remain in the log without
those fields, marked by their absence). The `status` field is populated when
`research/verdict.py` is run against a cohort (currently 20 % of rows; running verdict on the
remaining cohorts is part of the v0.2 polish).

---

## Honest scope — v0.1

- ✅ One VLM family fully shipped (Gemma 4 E4B-it). Backend ABC supports adding others; PRs welcome.
- ✅ Three categories of full N-coverage. The remaining 24 MVTec + VisA categories have
  zero-shot + N=30 only. Pass 4 in the roadmap fills the rest.
- ✅ Methodology + skills + provenance + framework — all production-quality.
- ❌ Classical baselines (Anomalib PatchCore, WinCLIP+) not yet rerun on our splits.
- ❌ Edge deployment (ONNX export, INT8 quantization) — abstractions in place; exporters in v0.2.
- ❌ Multi-GPU / cluster training — out of scope; this is a single-consumer-GPU framework.

---

## Documentation

- [POSITIONING.md](POSITIONING.md) — what this is and is not, scoping decisions, decision date
- [skills/README.md](skills/README.md) — the 15-skill catalog index
- [docs/skills.md](docs/skills.md) — narrative overview of how skills work
- [docs/autoresearch.md](docs/autoresearch.md) — the loop pattern, ranked contributions, worked example
- [docs/benchmarks.md](docs/benchmarks.md) — methodology, coverage, failure modes
- [docs/research-log.md](docs/research-log.md) — decisions, findings, and failures along the way
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute
- [SECURITY.md](SECURITY.md) — how to report vulnerabilities
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — community standards

---

## Acknowledgements

- **[unsloth](https://github.com/unslothai/unsloth)** — the QLoRA training kernel. The
  training-speed and VRAM wins of the pipeline are theirs, not ours.
- **[Andrej Karpathy](https://github.com/karpathy)** — the autoresearch loop pattern (March
  2026 release) that this project's loop wraps with benchmark-driven discipline. Also the 2019
  *Recipe for Training Neural Networks* discipline this project tries to honour.
- **[Anthropic skill-creator](https://github.com/anthropics/skills)** — the eval-driven
  self-improvement pattern this project's skill catalog adopts.
- **[Segment Anything 2](https://github.com/facebookresearch/sam2)** — interactive mask annotation.
- **[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)**,
  **[VisA](https://github.com/amazon-science/spot-diff)**,
  **[DeepPCB](https://github.com/tangsanli5201/DeepPCB)** — the benchmark datasets.

---

## License

[**Apache License 2.0**](LICENSE). Fully open-source, commercial use allowed.

## Citation

**If you use fsvlm in published research, please cite us.** Community norm, not a legal
requirement of Apache 2.0:

```bibtex
@software{hasan_fsvlm_2026,
  author  = {Hasan, Ahmad Jarjis},
  title   = {fsvlm: Few-shot Fine-tuning Benchmarker for Vision-Language Models},
  year    = {2026},
  url     = {https://github.com/ahmadhasan2k8/fsvlm},
  license = {Apache-2.0}
}
```

See also [`CITATION.cff`](CITATION.cff) for GitHub's auto-citation widget.
