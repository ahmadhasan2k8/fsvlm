---
name: meta-eval
description: |
  Run the eval suite against every skill and produce a skill-health report. Follows the
  Anthropic skill-creator eval pattern: each skill has a `skills/evals/<name>.eval.json` with
  (prompt, assertions) test cases, the harness sends each prompt to an agent runtime, runs each
  query multiple times for trigger-rate stability, and grades the result against the assertions.
  The output is the input to /improve-skill.
  TRIGGER when: nightly cadence, before a release, or after editing one or more skill files.
  SKIP when: the eval JSONs haven't been created yet (no signal to grade against).
inputs:
  - skills_dir (path, default: skills/)
  - evals_dir (path, default: skills/evals/)
  - runtime (string, default: portable) — portable | claude-code | openai-sdk
  - model (string, default: claude-haiku-4-5-20251001) — the agent's underlying LLM
  - samples_per_query (int, default: 3) — Anthropic's recommended trigger-rate stabiliser
  - max_iterations (int, default: 5) — per-query agent step budget
eval_artifact: ~/.fsvlm/skill_health.json
pass_criteria:
  - file exists at eval_artifact
  - JSON keys: window, runtime, model, per_skill (list), candidates_for_improvement (list)
  - every per_skill entry has: name, eval_set_size, trigger_rate, assertion_pass_rate, failed_evals
  - every per_skill entry's trigger_rate and assertion_pass_rate are in [0.0, 1.0]
  - candidates_for_improvement is sorted by ((1 − assertion_pass_rate) × eval_set_size) descending
escalation: |
  /meta-eval is mechanical. Output feeds /improve-skill (single skill) or
  /improve-skills-auto (all candidates) — those are the actuating layer.
---

# Skill: meta-eval

## Purpose

Run the eval suite, produce the skill-health report. The Anthropic skill-creator pattern
applied to fsvlm's whole skill catalog: which skills' descriptions trigger correctly, which
skills produce outputs that pass their assertions, and which need attention.

## Eval JSON convention

Each skill at `skills/<name>.md` has a sibling eval set at `skills/evals/<name>.eval.json`:

```json
{
  "skill": "train",
  "evals": [
    {
      "eval_id": 0,
      "eval_name": "labeled-folder-default-backend",
      "prompt": "I have 50 images of plastic parts in /data/parts/, half good half defective. Get me an adapter.",
      "expected_skill": "train",
      "assertions": [
        {"name": "skill_triggered", "type": "trigger", "value": "train"},
        {"name": "produced_eval_artifact", "type": "file_exists", "value_template": "${adapter_dir}/metrics.json"},
        {"name": "f1_above_baseline", "type": "json_field_gte", "path": "f1", "value": 0.5},
        {"name": "did_not_trigger_inspect", "type": "trigger_negation", "value": "inspect"}
      ]
    },
    {
      "eval_id": 1,
      "eval_name": "inference-only-should-skip",
      "prompt": "Run my existing adapter on /data/incoming/img.jpg.",
      "expected_skill": "inspect",
      "assertions": [
        {"name": "skill_triggered", "type": "trigger", "value": "inspect"},
        {"name": "did_not_trigger_train", "type": "trigger_negation", "value": "train"}
      ]
    }
  ]
}
```

Assertion types the harness understands (extensible — register new ones in
`scripts/run_skill_eval.py`):

- `trigger` — the named skill was the one the agent invoked
- `trigger_negation` — the named skill was NOT invoked (catches over-eager triggering)
- `file_exists` — a path (with template variables expanded) exists after the skill ran
- `json_field_gte` / `json_field_lte` / `json_field_eq` — the eval_artifact JSON has a field
  satisfying the comparison
- `pass_criteria_met` — every condition in the skill's frontmatter `pass_criteria` is met
- `wall_time_under` — the skill completed within `value` seconds

## Procedure

```bash
python scripts/run_skill_eval.py \
  --skills-dir "$skills_dir" \
  --evals-dir "$evals_dir" \
  --runtime "$runtime" \
  --model "$model" \
  --samples-per-query "$samples_per_query" \
  --max-iterations "$max_iterations" \
  --output "$eval_artifact"
```

The harness:

1. For each skill with an eval set, sends each `prompt` to the agent `samples_per_query` times
2. Records the trigger decision (which skill the agent chose) per sample → trigger rate
3. For each prompt where the right skill triggered, evaluates assertions on the output
4. Aggregates per-skill: trigger rate, assertion-pass rate, failed eval IDs
5. Sorts skills by improvement priority

Output schema:
```json
{
  "window": {"started_at": "2026-04-25T22:14Z", "duration_seconds": 287.4},
  "runtime": "portable",
  "model": "claude-haiku-4-5-20251001",
  "per_skill": [
    {
      "name": "train",
      "eval_set_size": 6,
      "trigger_rate": 1.00,
      "assertion_pass_rate": 0.83,
      "failed_evals": [
        {"eval_id": 4, "eval_name": "qwen-vl-backend-tiny-N",
         "failed_assertion": "f1_above_baseline (got 0.41, needed 0.50)"}
      ],
      "improvement_priority": "low"
    }
  ],
  "candidates_for_improvement": ["sweep", "verdict"]
}
```

## Self-evaluation

PASS if the file exists with the schema above and every per-skill entry has the required
fields. If a skill has no eval JSON, list it under `skills_without_evals` in the output and
continue (not a failure, just a coverage gap).

## Failure modes

- **Eval JSON malformed**: the harness writes the offending file's path under `parse_errors`
  and continues with the rest.
- **Agent runtime not reachable**: surface clearly — the harness can't grade what it can't run.
- **`samples_per_query=1`**: trigger rate is binary per-prompt, no smoothing. Anthropic
  recommends ≥3.

## Adapting to your runtime

- **Claude Code**: the `runtime=claude-code` mode invokes the project's installed skills
  directly via the harness wrapper
- **OpenAI Agents SDK**: `runtime=openai-sdk` registers each skill as a tool, runs the
  prompts through an Assistant
- **Portable** (default): the harness's own thin agent loop using the Anthropic Messages API
  or any OpenAI-compatible endpoint
- **Anthropic skill-creator**: if you already use the official `skill-creator` plugin, point it
  at `skills/evals/` directly — same JSON schema

## Pairs with

- **/improve-skill** — consumes `candidates_for_improvement` for one-skill-at-a-time editing
- **/improve-skills-auto** — meta-loop: meta-eval → improve-skill (per candidate) → meta-eval again

## Prior art

This skill follows the [Anthropic skill-creator eval pattern](https://github.com/anthropics/skills);
fsvlm contributes the eval JSONs for its own skill catalog and a portable harness so the loop
runs in any runtime, not just Claude Code.
