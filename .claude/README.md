# `.claude/` — Claude Code project configuration

This directory wires fsvlm's skills + safety hook into [Claude Code](https://www.anthropic.com/claude-code)
when you run it from the repo root. Clone the repo, `cd` into it, run `claude`, and Claude
Code will:

- **Discover the 15 fsvlm skills** as slash-commands (`/setup`, `/train`, `/sweep`,
  `/verdict`, `/plot`, `/autoresearch`, etc.) — see [`skills/`](skills/) here, or the
  [catalog index](../skills/README.md) for narrative context.
- **Apply the safety hook** at [`hooks/block-dangerous.sh`](hooks/block-dangerous.sh) —
  intercepts every shell command and refuses to run `rm -rf`, `git push --force`, `git reset
  --hard`, deletion of image files, dropping database tables, etc. Exit 0 = allow; exit 2 =
  block with a reason printed to stderr.
- **Honor the permission denylist** in [`settings.json`](settings.json) — a second layer of
  defense matching common-mistake bash patterns even before the hook runs.

## Layout

```
.claude/
├── README.md                   ← this file
├── settings.json               ← hook wiring + permission denylist
├── skills/                     ← 15 skills, symlinked to ../skills/<name>.md
│   ├── setup.md → ../../skills/setup.md
│   ├── train.md → ../../skills/train.md
│   └── … (13 more)
├── agents/                     ← 2 expert sub-agents the autoresearch loop consults
│   ├── vlm-researcher.md       ← VLM fine-tuning specialist
│   └── defect-specialist.md    ← industrial-inspection specialist
└── hooks/
    └── block-dangerous.sh      ← PreToolUse safety hook
```

`.claude/skills/*.md` are **symlinks** to the canonical files in `../skills/`. Single source
of truth — edit the file in `skills/` and Claude Code's auto-discovery picks up the change.

`.claude/agents/*.md` are domain-specific sub-agents the `/expert-review` skill calls into.
The two shipped here (`vlm-researcher`, `defect-specialist`) are the actual reviewers
fsvlm's autoresearch loop consults. They're useful as **templates** for your own domain —
fork the schema (the YAML frontmatter declares trigger conditions, available tools, and
model) and rewrite the body for whatever domain you care about (security-specialist,
perf-budget-specialist, accessibility-specialist, etc.).

## Using these without Claude Code

Each skill is a runtime-agnostic Markdown playbook. The 10 procedural skills are also
runnable from any shell via `bash skills/_run.sh <name> [args]`. See
[`../skills/README.md`](../skills/README.md) for the full runtime-vs-shell matrix.

## What's NOT shipped

- **`.claude/settings.local.json`** — user-local overrides; Claude Code creates this
  on first run if you change settings interactively.
- **`.claude/projects/`** — Claude Code session data, machine-local.
- **A handful of obsolete `.claude/skills/*.md`** — early-iteration project-specific skills
  (`run-experiment`, `validate-phase`, `launch-readiness`, etc.) that were superseded by the
  v0.1 generic catalog. Kept on disk for the original author's local use; not in the public
  release.

## Adapting the safety hook for your repo

`block-dangerous.sh` is intentionally generic — the patterns it blocks (rm -rf, force
push, drop table, sudo, system killers, deletion of training data) apply to most software
projects. If you copy it into another repo, tweak the **fsvlm-specific guards** section near
the bottom (image deletion, adapter deletion, model deletion) for your data shapes. The
script is ~80 lines of `grep`-based pattern matching; easy to read and extend.

## Adapting the skills for your stack

The 10 procedural skills wrap fsvlm CLI commands and only make sense in this repo. The 5
orchestrator skills (`/autoresearch`, `/improve-skill`, `/improve-skills-auto`,
`/expert-review`, `/debug`) describe runtime-agnostic patterns that you can re-implement
for any stack — see [`docs/autoresearch.md`](../docs/autoresearch.md) for the loop pattern.
