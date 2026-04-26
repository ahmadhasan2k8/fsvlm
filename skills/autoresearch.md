---
name: autoresearch
description: |
  Drive one iteration of the autoresearch loop end-to-end: read the next pending pass from
  queue.json, dispatch to the right skill (/sweep, /tiered-eval, etc.), run /verdict on the
  new rows, consult /expert-review, and decide the next action (next pass, halt, or rollback).
  This is fsvlm's adaptation of Karpathy's autoresearch loop for benchmark-driven research —
  see docs/autoresearch.md.
  TRIGGER when: user wants to advance the research arc by one pass without micromanaging each
  step; OR when running unattended in a /loop wrapper.
  SKIP when: user wants to run a single skill (use it directly); when no pending pass exists
  in queue.json (write the next hypothesis first).
inputs:
  - queue_path (path, default: research/queue.json)
  - results_path (path, default: research/dataset_size_results.json)
  - dry_run (bool, default: false) — print the planned actions without executing
  - max_minutes (int, default: 120) — abort if the pass exceeds this wall time
eval_artifact: research/autoresearch_state.json
pass_criteria:
  - file exists at eval_artifact
  - JSON keys: pass_id, pass_status, verdict_summary, expert_recommendation, next_action, decided_at
  - pass_status is one of: completed | halted_null | halted_done | rolled_back | aborted_timeout
  - if next_action is "next_pass: <id>", that pass exists in queue.json with status pending
  - the queue.json's current pass status is updated to match pass_status (atomic write)
escalation: |
  /autoresearch is the orchestrator; failures within it route to /debug, but the loop itself
  does not have a higher escalation — it stops and surfaces the state to the user. If
  pass_status is `aborted_timeout`, the partial rows are kept (they'll have status null until
  the next /verdict pass).
---

# Skill: autoresearch

## Purpose

The orchestrator. One invocation = one full pass through the loop documented in
`docs/autoresearch.md`. Designed to be wrapped in a `/loop` (Claude Code), a cron job, or a
plain `while true; do …; sleep 60; done` shell loop for unattended multi-day research.

## Procedure

1. **Read the queue.** Load `queue_path`, find the first pass with status `pending` or
   `pending_user_approval`. If none, write `pass_status: halted_done` and exit.

2. **Verify clean tree.** `git diff --quiet && git diff --cached --quiet` — fail with
   `pass_status: aborted_timeout` and reason `dirty_tree` if not.

3. **Dispatch.** Map the pass's `kind` field to the right skill:
   - `kind: sweep` → invoke `/sweep` with the pass's parameters
   - `kind: tiered_eval` → invoke `/tiered-eval`
   - `kind: ablation` → invoke `/sweep` with the ablation's recipe overrides
   - other → fail with `unknown pass kind`

4. **Wait for the dispatched skill to PASS.** If it FAILs, write `pass_status: aborted_timeout`
   (or the more specific failure) and exit.

5. **Run /verdict** on the cohort the dispatched skill produced:
   ```bash
   skills/_run.sh verdict --recipe-version "$RECIPE" --write
   ```

6. **Consult /expert-review** with the verdict summary as context:
   ```bash
   skills/_run.sh expert-review \
     --role training-specialist \
     --results-subset /tmp/cohort.json \
     --hypothesis "$(jq -r .hypothesis_primary < /tmp/this_pass.json)"
   ```

7. **Decide.** Combine the verdict's status counts with the expert's recommendation:
   - all `noop` and expert says "halt: null result" → `pass_status: halted_null`
   - any `keep` or `new_baseline` and expert says "next_config: <spec>" → write the next pass
     to queue.json with the spec, set `pass_status: completed`,
     `next_action: "next_pass: <new_id>"`
   - any `discard` and expert says "rollback" → `git checkout` the recipe files,
     `pass_status: rolled_back`
   - mixed verdict + low-confidence expert → `pass_status: completed`,
     `next_action: "halt_for_user_review"`, surface to the user

8. **Atomically update** queue.json (mark the just-finished pass `completed` with
   `verdict_summary` populated) AND write `eval_artifact` (autoresearch_state.json).

## State file schema

```json
{
  "pass_id": "pass3-curve-with-tiny-N",
  "pass_status": "completed",
  "verdict_summary": "9 new_baseline, 2 keep, 1 noop on hazelnut/candle/pcb at N≤100",
  "expert_recommendation": "next_config: launch ICL ablation at same cells",
  "next_action": "next_pass: pass5b-icl-ablation",
  "decided_at": "2026-04-25T22:14:00Z",
  "git_hash": "...",
  "elapsed_minutes": 47.3,
  "skills_invoked": ["sweep", "verdict", "expert-review"]
}
```

## Self-evaluation

PASS if the state file is written with all required keys, the queue.json is consistent (the
just-finished pass is marked completed and the proposed next pass exists if next_action says
so), and the elapsed time is under `max_minutes`.

## Failure modes

- **Dirty tree at start**: refuse to run; uncommitted changes break provenance. User must
  commit or stash.
- **Dispatched skill never PASSes**: surface the skill's eval_artifact, mark the pass
  `aborted_timeout`, leave the partial rows for /verdict to discard.
- **Expert review hallucinates / outputs malformed JSON**: re-render the prompt with stricter
  format examples; if it fails twice, halt and surface to the user.
- **Two consecutive passes recommend the same `next_config`**: loop is stuck; halt with
  `pass_status: halted_null` and surface.

## Running unattended

```bash
# Claude Code: `/loop /autoresearch` — fires once per harness wakeup, self-paces
# Cron: */30 * * * * cd /path/to/repo && skills/_run.sh autoresearch
# Plain shell:
while true; do
  skills/_run.sh autoresearch || break
  sleep 60
done
```

The loop terminates naturally when the queue is empty (`pass_status: halted_done`) or when a
pass declares a null result (`pass_status: halted_null`).

## Adapting to your runtime

- **Claude Code**: drop into `~/.claude/skills/`; wrap in `/loop /autoresearch` for unattended
  multi-day operation
- **OpenAI Agents SDK**: register the orchestrator as the top-level Assistant, the other skills
  as its sub-tools
- **Plain shell**: implement steps 1–8 in `bash`; pure I/O on JSON files

## Pairs with

- All of /sweep, /tiered-eval, /verdict, /expert-review — autoresearch is the conductor
- **/debug** — when a sub-skill fails, autoresearch routes the eval_artifact to /debug
