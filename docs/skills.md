# fsvlm Skills — How the Catalog Works

fsvlm ships **15 self-evaluating, runtime-agnostic skills** that take a researcher from zero
to a paper-ready figure pack: 6 core CLI wrappers, 5 research-loop skills, 1 expert-review
template, and 3 meta-layer skills that let the catalog improve itself. This page is the
narrative companion to [`skills/README.md`](../skills/README.md) (the catalog index).

---

## What problem the catalog solves

Coming to a benchmarking framework cold, you typically write three or four shell scripts and
some glue Python before you have anything reproducible. fsvlm replaces that bootstrapping
work with a set of small, declarative playbooks:

- Each skill is **one Markdown file** with YAML frontmatter declaring `name`, `description`,
  `inputs`, `eval_artifact`, `pass_criteria`, and `escalation`
- Each skill **wraps an existing fsvlm CLI command** (or composes a few) — no hidden Python
- Each skill **declares its own self-evaluation** — the calling runtime can mechanically check
  whether the skill did what it said it would
- Each skill is **runtime-agnostic** — adapt to Claude Code, OpenAI Agents SDK, CrewAI, or
  plain shell

The intent: a researcher should be able to clone the repo, run `/setup → /train → /sweep →
/verdict → /plot`, and have figures + tables ready for a paper draft within a working day.

---

## The "0 → paper" path

```
   /setup    → machine ready, model downloaded
   /train    → adapter on a folder of labeled images (one-off, or)
   /tiered-eval → confirm fine-tuning helps over zero-shot before the full sweep
   /sweep    → AUROC across N values × seeds × categories
   /verdict  → keep / discard / noop / new_baseline per cell
   /expert-review → contextual review (anti-Goodhart guard)
   /plot     → AUROC-vs-N curves + CSV/MD/LaTeX tables ready for Overleaf
```

Wrap the inner cycle (sweep → verdict → expert-review → decide) in `/autoresearch` to run it
unattended. See [docs/autoresearch.md](autoresearch.md) for the loop's full design.

---

## Adaptive, not hardcoded

Every skill takes its inputs as parameters. Nothing in the catalog is wired to MVTec or
Gemma 4 specifically — those are defaults that work for the demonstration backend, freely
replaceable for any new `LabelReader` or `ModelBackend`:

```bash
# Same /sweep skill, different backend + dataset
fsvlm/skills/_run.sh sweep \
  --backend qwen-vl \
  --dataset visa \
  --categories pcb1 pcb2 pcb3 \
  --n-values 2 5 10 30 \
  --seeds 42 1337 7
```

Adding a new backend or dataset is one new file each (the recipes in
[docs/autoresearch.md § 6](autoresearch.md) cover the contracts).

---

## Self-evaluation contract

Every skill produces machine-checkable JSON output and declares the conditions it must satisfy.
A calling runtime should:

1. Run the skill's procedure
2. Read the file at `eval_artifact`
3. Check every `pass_criteria` is met
4. PASS → continue; FAIL → invoke the skill named in `escalation` (often `/debug`)

This contract means the loop can iterate without a human in every step. Failures are loud,
structured, and routable. See `skills/setup.md` through `skills/expert-review.md` for the 12
single-purpose skills' contracts; see `skills/meta-eval.md`, `skills/improve-skill.md`, and
`skills/improve-skills-auto.md` for the meta layer that makes the catalog itself self-
improving.

---

## The meta layer — skills improving skills

The Karpathy autoresearch loop applied to skill files. Each skill ships with a sibling
`skills/evals/<name>.eval.json` declaring (prompt, expected trigger, assertions). The harness
sends each prompt to the agent runtime, runs each query 3× for trigger-rate stability, grades
the assertions, produces a skill-health report. The improvement loop:

1. **/meta-eval** runs the eval suite against every skill, produces `skill_health.json`
2. **/improve-skill** picks one flagged skill, splits its eval set 60/40, identifies failing
   training cases, asks an LLM proposer for a minimal edit, validates on the held-out 40%,
   commits if the test pass-rate improves by ≥ 0.10, reverts otherwise
3. **/improve-skills-auto** orchestrates both nightly: meta-eval → improve-skill (per
   candidate) → meta-eval again, with a global rollback if the catalog regresses in aggregate

This is **fsvlm's adaptation of the [Anthropic skill-creator eval pattern](https://github.com/anthropics/skills)**.
The 60/40 split + the 3-samples-per-query stabiliser + the diff-proposer + held-out validation
pattern are theirs. fsvlm contributes (a) the eval JSONs for its specific skill catalog, (b) a
runtime-portable harness so the loop runs in non-Claude-Code stacks, and (c) the catalog-level
rollback guard that lets `/improve-skills-auto` run unattended without risking silent global
drift.

---

## Adoption recipes

### With Claude Code

```bash
cp skills/*.md ~/.claude/skills/
# (or: ln -s "$(pwd)/skills" ~/.claude/skills/fsvlm)
# Then invoke /setup, /train, /sweep, etc.
# Pair with the official skill-creator plugin for the improvement loop.
```

### With OpenAI Agents SDK / Anthropic SDK / CrewAI

For each skill, register a function with the signature in its `inputs:` block. The function
body is the Procedure section (usually a subprocess.run on the fsvlm CLI). Pre-condition: read
`eval_artifact`. Post-condition: assert `pass_criteria`. Wire the `escalation` pointer as a
follow-up tool call.

### With plain shell + your favourite chatbot

The skills are runnable as straight shell scripts (the Procedure section IS the script). For
`/expert-review`, the procedure can render the prompt to stdout for manual paste into your
chatbot and you paste the JSON response back. Same audit trail.

### Validating + improving the catalog

```bash
python scripts/validate_skills.py            # frontmatter + eval-coverage check
python scripts/run_skill_eval.py             # run the harness, write skill_health.json
# (Then /improve-skill or /improve-skills-auto for the actuating layer.)
```

---

## What's NOT shipped

Skills tied to the specific industrial-anomaly research that drove fsvlm's development —
`/run-experiment` (Phase 0 baseline on MVTec hazelnut), `/validate-phase` (PLAN.md-specific),
`/launch-readiness` (POSITIONING.md-specific) — were intentionally **not** included. They were
useful for the project's internal autoresearch arc but would be noise for outside users. The
generic, parameterised replacements (`/sweep`, `/verdict`, `/tiered-eval`, `/autoresearch`)
cover the same workflows without locking in a specific dataset or recipe.

---

## See also

- [`skills/README.md`](../skills/README.md) — the catalog index with one-line descriptions
- [docs/autoresearch.md](autoresearch.md) — the loop pattern the meta layer uses
- [docs/benchmarks.md](benchmarks.md) — methodology + dataset coverage
- [Anthropic skill-creator](https://github.com/anthropics/skills) — the canonical reference
  for the eval JSON schema and the improve-skill loop
