---
name: improve-skills-auto
description: |
  Unattended driver for the meta-loop: run /meta-eval, then run /improve-skill on every
  flagged candidate, then re-run /meta-eval to confirm the catalog is healthier than before.
  This is the meta-version of /autoresearch — Karpathy's loop applied to the skill catalog as
  a whole, repeatable on a nightly cadence. Designed to be wrapped in a `/loop`, cron, or
  shell `while`.
  TRIGGER when: nightly cadence, before a release, or after a batch of skill edits where
  drift is likely.
  SKIP when: there's no eval coverage yet (write eval JSONs first), or when CI is currently
  red (don't propose edits on a broken tree).
inputs:
  - max_skills_per_run (int, default: 3) — cap improvements per invocation; rest defer to next run
  - improvement_threshold (float, default: 0.10) — passed through to /improve-skill
  - dry_run (bool, default: false)
  - max_total_minutes (int, default: 120)
eval_artifact: research/skill_meta_loop_state.json
pass_criteria:
  - file exists at eval_artifact
  - JSON keys: started_at, finished_at, before_health, after_health, candidates_processed (list), summary
  - every candidates_processed entry has: skill, decision, gain
  - after_health.aggregate_pass_rate >= before_health.aggregate_pass_rate − 0.02 (no net regression)
  - elapsed_minutes <= max_total_minutes
escalation: |
  /improve-skills-auto is the orchestrator. If after_health is worse than before_health
  beyond the −0.02 noise floor: rollback every commit this run made (cherry-pick the audit
  records, `git revert` each), surface to user. Without this guard a bad proposer round could
  globally regress.
---

# Skill: improve-skills-auto

## Purpose

The meta-loop driver. One invocation = one full sweep of:

```
/meta-eval (before) → /improve-skill (per candidate) → /meta-eval (after)
```

with a global rollback if the catalog regressed in aggregate.

> **Honest scope.** Same as `/autoresearch`: this is an orchestrator skill that needs a
> runtime to interpret its conditional logic. v0.1 ships:
>
> - the documented procedure below
> - `skills/_run.sh` dispatcher for the procedural sub-skills (`meta-eval`,
>   `improve-skill`)
> - eval JSONs for every skill under `skills/evals/`
>
> v0.1 does **not** ship a turnkey runner that drives this loop unattended. To actually
> execute it, drive from Claude Code (`/improve-skills-auto`), from another agent runtime
> (OpenAI Agents SDK, CrewAI), or manually step through the procedure. There is no committed
> evidence in this release of the meta-loop having actually self-improved a skill (no
> `research/skill_improvements/` outputs). PRs that turn this into a turnkey runner are
> welcome.

## Procedure

1. **Pre-checks.** Working tree clean. CI status (if available) is green. Otherwise abort
   with `aborted_dirty_tree` or `aborted_ci_red`.

2. **Snapshot the catalog.** `BEFORE_SHA=$(git rev-parse HEAD)`.

3. **Baseline /meta-eval.**
   ```bash
   skills/_run.sh meta-eval --output /tmp/before_health.json
   ```
   Capture per-skill + aggregate pass rates.

4. **Iterate over candidates.**
   ```bash
   CANDS=$(jq -r '.candidates_for_improvement[]' /tmp/before_health.json | head -n "$max_skills_per_run")
   for skill in $CANDS; do
     ELAPSED=$(date +%s); ((ELAPSED - START > max_total_minutes * 60)) && break
     skills/_run.sh improve-skill --skill-name "$skill" \
       --improvement-threshold "$improvement_threshold" \
       ${dry_run:+--dry-run} \
       --output "/tmp/improve_${skill}.json"
   done
   ```

5. **Re-run /meta-eval after edits.**
   ```bash
   skills/_run.sh meta-eval --output /tmp/after_health.json
   ```

6. **Compare aggregates. Karpathy at the catalog level:**
   ```bash
   BEFORE=$(jq -r '.aggregate_pass_rate // .per_skill | map(.assertion_pass_rate) | add / length' /tmp/before_health.json)
   AFTER=$(jq -r '.aggregate_pass_rate // .per_skill | map(.assertion_pass_rate) | add / length' /tmp/after_health.json)
   DELTA=$(echo "$AFTER - $BEFORE" | bc -l)
   if (( $(echo "$DELTA < -0.02" | bc -l) )); then
     # net regression — roll back every commit this run made
     git reset --hard "$BEFORE_SHA"
     STATUS=rolled_back_global_regression
   else
     STATUS=ok
   fi
   ```

7. **Write the loop state.**
   ```json
   {
     "started_at": "...", "finished_at": "...", "elapsed_minutes": 47.3,
     "before_health": {"aggregate_pass_rate": 0.81, "per_skill": [...]},
     "after_health":  {"aggregate_pass_rate": 0.86, "per_skill": [...]},
     "candidates_processed": [
       {"skill": "train",  "decision": "committed", "gain": 0.12, "commit_sha": "..."},
       {"skill": "sweep",  "decision": "reverted",  "gain": 0.04},
       {"skill": "verdict","decision": "aborted_min_eval_set_size"}
     ],
     "status": "ok",
     "summary": "Improved 1/3 candidates, aggregate pass-rate +0.05. No global regression."
   }
   ```

## Self-evaluation

PASS if the loop state file exists with the schema above AND `status in {ok, rolled_back_*,
aborted_*}` AND elapsed under `max_total_minutes`.

## Running unattended

```bash
# Claude Code: /loop /improve-skills-auto — fires on the harness's cadence
# Cron (nightly):  0 3 * * * cd /path/to/repo && skills/_run.sh improve-skills-auto
# Plain shell:
while true; do
  skills/_run.sh improve-skills-auto || break
  sleep 86400  # daily
done
```

## Why the global rollback guard

A single /improve-skill run validates against its own held-out 40%, which is honest. But a
proposer can occasionally win on one skill's hold-out while breaking a different skill's
trigger boundary (e.g. tightening /train's TRIGGER text such that /retrain now wins those
prompts). The catalog-level /meta-eval re-run + rollback catches that interaction.

## Failure modes

- **No candidates flagged**: the catalog is healthy. `summary: "no candidates; loop idle"`.
- **Every candidate reverts**: proposer is stuck. Surface; consider switching `proposer_model`
  or expanding eval coverage so failures are more diagnostic.
- **Global regression triggered rollback**: every committed edit from this run is undone. The
  audit record names which skills the proposer thought it had improved; user should review.

## Adapting to your runtime

- **Claude Code**: `/loop /improve-skills-auto` for a self-pacing skill-maintenance loop
- **Cron / systemd timer**: run nightly; the loop is idempotent
- **Manual**: invoke after batch skill edits as a regression check

## Pairs with

- **/meta-eval** + **/improve-skill** — the two skills this orchestrates
- **/autoresearch** — the parallel pattern, applied to research passes instead of skill files

## Prior art

The combination of (a) per-skill train/test split with proposer-edit-validate from
[Anthropic skill-creator](https://github.com/anthropics/skills) and (b)
catalog-level rollback guard is the wrapper fsvlm contributes — letting the skill-improvement
loop run unattended without risking silent global drift.
