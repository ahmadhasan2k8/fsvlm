#!/usr/bin/env bash
# skills/_run.sh — minimal shell dispatcher for the runtime-agnostic skills.
#
# Maps a skill name to its underlying CLI / script entry point. This exists so the
# procedural skills (setup, train, inspect, validate, serve, sweep, verdict,
# tiered-eval, plot, meta-eval) can be invoked from a plain shell, a cron job, or
# any runtime that can shell out — without requiring a full agent runtime to
# interpret the skill markdown.
#
# Orchestrator skills (autoresearch, improve-skill, improve-skills-auto,
# expert-review, debug) require a runtime that can interpret natural-language
# procedure steps and call sub-skills based on conditional logic. This dispatcher
# returns a clear error pointing at the skill markdown for those.
#
# Usage:
#   skills/_run.sh <skill-name> [args...]
#
# Examples:
#   skills/_run.sh setup --check
#   skills/_run.sh train --images ./my-data/ --epochs 3
#   skills/_run.sh sweep --datasets mvtec --categories hazelnut --n-values 0 30 --seeds 42
#   skills/_run.sh verdict --results research/dataset_size_results.json --write

set -euo pipefail

SKILL="${1:-}"
if [[ -z "$SKILL" ]]; then
    cat <<EOF
Usage: skills/_run.sh <skill-name> [args...]

Procedural skills (this dispatcher runs them directly):
  setup        train         inspect       validate      serve
  sweep        verdict       tiered-eval   plot          meta-eval

Orchestrator skills (require a runtime — see the skill markdown):
  autoresearch          improve-skill         improve-skills-auto
  expert-review         debug

Catalog: skills/README.md
EOF
    exit 1
fi
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve the python interpreter: prefer python3, fall back to python
PYTHON="${PYTHON:-$(command -v python3 || command -v python || echo python3)}"

case "$SKILL" in
    # Core CLI wrappers — invoke via python -m to avoid PATH ambiguity with the
    # local fsvlm/ package directory
    setup)         exec "$PYTHON" -m fsvlm.cli setup "$@" ;;
    train)         exec "$PYTHON" -m fsvlm.cli train "$@" ;;
    inspect)       exec "$PYTHON" -m fsvlm.cli inspect "$@" ;;
    validate)      exec "$PYTHON" -m fsvlm.cli validate "$@" ;;
    serve)         exec "$PYTHON" -m fsvlm.cli serve "$@" ;;

    # Research-loop primitives — wrap the underlying scripts
    sweep)         exec bash "$REPO_ROOT/research/run_sweep.sh" "$@" ;;
    verdict)       exec "$PYTHON" "$REPO_ROOT/research/verdict.py" "$@" ;;
    tiered-eval)   exec "$PYTHON" "$REPO_ROOT/research/tiered_validation.py" "$@" ;;
    plot)          exec "$PYTHON" "$REPO_ROOT/research/plots.py" "$@" ;;
    meta-eval)     exec "$PYTHON" "$REPO_ROOT/scripts/run_skill_eval.py" "$@" ;;

    # Orchestrator skills — need a runtime that interprets procedure markdown
    autoresearch|improve-skill|improve-skills-auto|expert-review|debug)
        cat >&2 <<EOF
/$SKILL is an orchestrator skill — it composes other skills based on conditional
logic that needs a runtime to interpret. Options:

  1. Drive it from Claude Code: drop skills/$SKILL.md into ~/.claude/skills/
     and invoke as /$SKILL — the agent reads the procedure and dispatches.

  2. Drive it from another agent runtime (OpenAI Agents SDK, CrewAI, your own
     orchestrator) by registering the procedure as a tool that calls _run.sh
     for sub-skills.

  3. Drive it manually: read skills/$SKILL.md and execute the procedure steps
     yourself, calling _run.sh for sub-skills as you go.

Direct shell dispatch is not supported for orchestrator skills.
EOF
        exit 2
        ;;

    *)
        echo "Unknown skill: $SKILL" >&2
        echo "Run skills/_run.sh with no arguments to list available skills." >&2
        exit 1
        ;;
esac
