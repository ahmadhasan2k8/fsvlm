# fsvlm Skills Catalog

Runtime-agnostic Markdown playbooks for getting from **zero to a paper** with fsvlm. Every
skill is one Markdown file with YAML frontmatter (`name`, `description`, `inputs`,
`eval_artifact`, `pass_criteria`, `escalation`) plus a procedural body.

> **What's a "skill" here, concretely?** Two kinds:
>
> - **Procedural skills** (10): `setup`, `train`, `inspect`, `validate`, `serve`, `sweep`,
>   `verdict`, `tiered-eval`, `plot`, `meta-eval`. These wrap a single underlying CLI command
>   or script. They are **directly executable** via `bash skills/_run.sh <name> [args]` — the
>   `_run.sh` dispatcher is real, ships in v0.1, and works from cron / Make / shell.
> - **Orchestrator skills** (5): `autoresearch`, `improve-skill`, `improve-skills-auto`,
>   `expert-review`, `debug`. These compose other skills via conditional logic ("if verdict
>   says X, dispatch /sweep with these params; if expert says halt, halt"). They are
>   **documented protocols, not turnkey runners** — they need a runtime that can interpret
>   natural-language procedure markdown (Claude Code, OpenAI Agents SDK, CrewAI) or a human
>   stepping through them manually. Direct shell invocation returns an explicit "needs a
>   runtime" error.
>
> Eval JSONs and the skill-self-improvement loop (`/improve-skill`,
> `/improve-skills-auto`) follow the
> [Anthropic skill-creator pattern](https://github.com/anthropics/skills) but no shipped
> v0.1 evidence exists of the meta-loop having actually self-improved a skill — the loop is
> documented end-to-end and the eval JSONs are real, but the runner that closes the loop
> needs the runtime above. PRs welcome.

> **Adaptive, not hardcoded.** Every procedural skill takes `dataset`, `backend`,
> `categories`, `n_values`, `seeds` etc. as parameters. Nothing in this catalog is wired to
> MVTec or Gemma 4 specifically; they are defaults that work for the demonstration backend,
> replaceable for any new `LabelReader` or `ModelBackend`.

---

## The catalog

### Core CLI wrappers

| Skill | Purpose | Eval artifact |
|---|---|---|
| [setup](setup.md) | Detect GPU, recommend + download a base VLM, verify install | `~/.fsvlm/setup_status.json` |
| [train](train.md) | Fine-tune a QLoRA adapter on a folder of labeled images | `<adapter>/metrics.json` |
| [inspect](inspect.md) | Run an adapter on one image / folder / stream | stdout or output JSON |
| [validate](validate.md) | Evaluate an adapter on a held-out test set + report | `<output>/validation_report.json` |
| [serve](serve.md) | Launch UI / REST / watch-folder mode for deployment | `~/.fsvlm/serve_health.json` |
| [debug](debug.md) | Diagnose env / GPU / deps / adapter / dataset problems | `~/.fsvlm/debug.json` |

### Research-loop skills (the "0 → paper" critical path)

| Skill | Purpose | Eval artifact |
|---|---|---|
| [sweep](sweep.md) | Run a (dataset × category × N × seed) sweep, append rows with provenance | `research/dataset_size_results.json` |
| [verdict](verdict.md) | Classify rows: `new_baseline` / `keep` / `noop` / `discard` | the same results JSON, mutated in place |
| [tiered-eval](tiered-eval.md) | Zero-shot → few-shot → full-train monotonicity check | `research/tiered_results.json` |
| [plot](plot.md) | Turn results into AUROC-vs-N curves, comparison bars, CSV/MD/LaTeX tables | `<output>/manifest.json` |
| [autoresearch](autoresearch.md) | Drive one full loop pass: dispatch → verdict → expert-review → decide | `research/autoresearch_state.json` |

### Expert review (parameterised by role)

| Skill | Purpose | Eval artifact |
|---|---|---|
| [expert-review](expert-review.md) | Consult a domain reviewer (`training-specialist` / `domain-specialist` / your role) before locking the verdict — the anti-Goodhart contextual-review guard | `research/expert_reviews/<role>_<timestamp>.json` |

### Meta layer — skill self-improvement (the catalog improves itself)

| Skill | Purpose | Eval artifact |
|---|---|---|
| [meta-eval](meta-eval.md) | Run the eval suite against every skill (60/40 split, 3 samples per query for trigger-rate stability), produce a skill-health report | `~/.fsvlm/skill_health.json` |
| [improve-skill](improve-skill.md) | Self-improvement loop for one skill — propose minimal edit, validate on held-out 40%, Karpathy commit-or-revert | `research/skill_improvements/<skill>_<ts>.json` |
| [improve-skills-auto](improve-skills-auto.md) | Unattended driver: meta-eval → improve-skill per candidate → meta-eval again, with global rollback if the catalog regresses | `research/skill_meta_loop_state.json` |

Every skill ships with a sibling `skills/evals/<name>.eval.json` declaring (prompt, expected
trigger, assertions). The eval JSONs are the input to the meta layer — without them, the
self-improvement loop has nothing to grade against. See `skills/evals/setup.eval.json` for the
schema; the 12 core skills + 3 meta skills each have one.

---

## The "0 → paper" path, end to end

```
┌────────────────────────┐
│ /setup                 │  ← machine ready?
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│ /train  (one-off, or)  │
│ /tiered-eval           │  ← does fine-tuning help on this (backend, category)?
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│ /sweep                 │  ← collect (dataset × category × N × seed) rows
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│ /verdict               │  ← keep/discard/noop/new_baseline per cell
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│ /expert-review         │  ← contextual sanity check
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│ decide: next pass,     │
│ halt-null, or rollback │  ← updates queue.json
└──────────┬─────────────┘
           ▼
┌────────────────────────┐
│ /plot                  │  ← figures + tables for the paper
└────────────────────────┘
```

Wrap the inner cycle (sweep → verdict → expert-review → decide) in `/autoresearch` to run it
unattended. See [docs/autoresearch.md](../docs/autoresearch.md) for the loop's full design,
contributions ranking, and a worked example.

---

## Self-evaluation contract

Every skill declares two things in its frontmatter:

- `eval_artifact` — a path (or pattern) to the JSON file the skill produces
- `pass_criteria` — a list of conditions a runtime can mechanically check

When a runtime invokes a skill, it should:

1. Run the skill's procedure
2. Read `eval_artifact`
3. Check `pass_criteria` are all met
4. PASS → continue; FAIL → invoke the skill named in `escalation` (often `/debug`)

This contract means the loop can iterate without a human in every step. The expectation isn't
that everything always passes — it's that failures are loud, structured, and routable.

---

## Adopting these skills

### With Claude Code

```bash
cp skills/*.md ~/.claude/skills/
# (or symlink: ln -s "$(pwd)/skills" ~/.claude/skills/fsvlm)
# Then invoke with /setup, /train, /sweep, etc.
```

### With OpenAI Agents SDK / Anthropic SDK / CrewAI

For each skill, register a function with the signature in `inputs:` whose body is the Procedure
section. Pre-condition: read `eval_artifact`; post-condition: assert `pass_criteria`. Wire the
escalation pointer as a follow-up tool call.

### With plain shell + your favourite chatbot

The skills are runnable as straight shell commands (the Procedure section is the script). For
`/expert-review`, the procedure can render the prompt to stdout for manual paste into ChatGPT /
Claude / Gemini / Mistral / your local model — and you paste the JSON response back.

---

## Validator + eval harness

`scripts/validate_skills.py` checks that every file in this directory has well-formed
frontmatter, the required keys, and a sibling eval JSON. CI runs it on every PR.

```bash
python scripts/validate_skills.py
# {"summary": {"pass": 15, "warn": 0, "fail": 0, "total": 15}, ...}
```

`scripts/run_skill_eval.py` is the portable harness that runs the eval JSONs against an agent
runtime (default: minimal Anthropic Messages loop; also supports `claude-code` and `openai-sdk`
adapters). It produces the `skill_health.json` that `/meta-eval` consumes.

```bash
python scripts/run_skill_eval.py --runtime portable --samples-per-query 3
# wrote ~/.fsvlm/skill_health.json (aggregate_pass_rate=0.87)
```

The eval JSON schema follows the [Anthropic skill-creator](https://github.com/anthropics/skills)
convention: `(eval_id, eval_name, prompt, expected_skill, assertions)` per case. Assertion
types: `trigger`, `trigger_negation`, `input_eq`, `input_in_set`, `input_contains`,
`file_exists`, `json_field_gte`, `json_field_lte`, `json_field_eq`, `pass_criteria_met`,
`wall_time_under`. Add new types in `scripts/run_skill_eval.py`.

---

## Why this catalog matters

The `/expert-review` and self-eval-frontmatter ideas are fsvlm's two main contributions on top
of [Karpathy's autoresearch loop](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/);
the rest of the catalog (`/sweep`, `/verdict`, `/plot`, `/tiered-eval`, `/autoresearch`) is the
machinery that turns the loop into something a researcher can actually go from zero to a paper
with. See [docs/autoresearch.md § 4.5](../docs/autoresearch.md) for the full "Karpathy's
pattern vs what fsvlm adds" boundary.

---

## Roadmap

v0.1 ships the 12 functional skills + 3 meta skills above (15 total), with eval JSONs for each.
Planned for v0.2:

- `add-backend` — scaffold a new `ModelBackend` (Qwen-VL, LLaVA, etc.) with ABC + registry
  decorator + unit-test stub
- `add-reader` — scaffold a new `LabelReader` for a new dataset, with download script + frozen
  taxonomy template
- `clean` — cache management with safety checks (dry-run + confirmation)
- `release-bench` — bundle a paper-ready archive (results JSON + figures + per-cell table) for
  GitHub Release attachment

Contributions welcome via the same one-PR pattern documented in
[docs/autoresearch.md § 6](../docs/autoresearch.md).
