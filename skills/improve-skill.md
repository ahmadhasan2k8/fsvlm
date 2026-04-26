---
name: improve-skill
description: |
  Self-improvement loop for a single skill, following the Anthropic skill-creator pattern: split
  the skill's eval set 60/40 train/test, identify failing training cases, propose a minimal edit
  via an LLM proposer, validate the proposed edit on the held-out 40%, apply Karpathy's commit-
  or-revert primitive (commit if held-out pass-rate improves, revert otherwise). Recursive
  autoresearch — Karpathy's loop applied to skill files instead of training code.
  TRIGGER when: /meta-eval flagged a skill in `candidates_for_improvement`; OR a user
  explicitly wants to improve a specific skill.
  SKIP when: the skill has no eval JSON (write one first via skill-creator or by hand);
  when the eval set is too small (< 5 cases) for a meaningful split.
inputs:
  - skill_name (string, required) — file under skills/ to improve, without .md
  - evals_dir (path, default: skills/evals/)
  - runtime (string, default: portable)
  - model (string, default: claude-haiku-4-5-20251001)
  - proposer_model (string, default: claude-opus-4-7) — beefier model for proposing edits
  - train_test_split (float, default: 0.6)
  - samples_per_query (int, default: 3)
  - improvement_threshold (float, default: 0.10) — held-out pass-rate gain required to commit
  - max_iterations (int, default: 5) — proposal-edit-validate cycles per run
  - dry_run (bool, default: false)
eval_artifact: research/skill_improvements/${skill_name}_${timestamp}.json
pass_criteria:
  - file exists at eval_artifact
  - JSON keys: skill, baseline_train_pass_rate, baseline_test_pass_rate, candidate_test_pass_rate, decision, edit_diff, iterations
  - decision is one of: committed | reverted | dry_run | aborted_no_signal | aborted_min_eval_set_size
  - if decision == committed: skill file changed AND a git commit exists AND
    candidate_test_pass_rate − baseline_test_pass_rate ≥ improvement_threshold
  - if decision == reverted: skill file unchanged (git diff empty)
escalation: |
  /improve-skill is the actuating step. Two consecutive `reverted` decisions for the same
  skill → surface to user; the failing assertions are probably not addressable by editing the
  skill .md alone (CLI bug, runtime bug, or environment issue). Route to /debug.
---

# Skill: improve-skill

## Purpose

Documented protocol for running Karpathy's commit-or-revert loop on skill files:

- **The single editable file** = `skills/<skill_name>.md`
- **The single objective metric** = held-out assertion pass-rate from the skill's eval set
- **The Karpathy primitive** = commit if held-out gain ≥ `improvement_threshold`, else revert

The 60/40 train/test split + 3-samples-per-query stabiliser come straight from the
[Anthropic skill-creator eval methodology](https://github.com/anthropics/skills).

> **Honest scope.** Like `/autoresearch` and `/improve-skills-auto`, this is an orchestrator
> skill that needs a runtime to drive (it makes conditional decisions about whether to commit
> or revert based on eval results). v0.1 ships the procedure below + the
> `skills/_run.sh meta-eval` dispatcher; the loop body needs Claude Code, another agent
> runtime, or manual execution. No committed evidence of this loop having self-improved a
> shipped skill.

## Procedure

1. **Pre-checks.** Working tree clean for `skills/${skill_name}.md`; eval JSON exists at
   `${evals_dir}/${skill_name}.eval.json` with at least 5 cases. Otherwise:
   `aborted_min_eval_set_size`.

2. **Split the eval set.** Deterministic split (seed = hash of skill name): `train_test_split`
   fraction goes to `/tmp/eval_train.json`, the rest to `/tmp/eval_test.json`. Hold the test
   set aside; nothing in the loop sees it until step 7.

3. **Baseline run on the training set.**
   ```bash
   python scripts/run_skill_eval.py \
     --skill "$skill_name" \
     --eval-set /tmp/eval_train.json \
     --runtime "$runtime" --model "$model" \
     --samples-per-query "$samples_per_query" \
     --output /tmp/baseline_train.json
   BASELINE_TRAIN=$(jq -r .assertion_pass_rate /tmp/baseline_train.json)
   ```

4. **Baseline run on the held-out test set** (recorded but not used as proposer signal):
   ```bash
   python scripts/run_skill_eval.py \
     --skill "$skill_name" --eval-set /tmp/eval_test.json \
     --output /tmp/baseline_test.json
   BASELINE_TEST=$(jq -r .assertion_pass_rate /tmp/baseline_test.json)
   ```

5. **Iterate up to `max_iterations` proposal cycles.** Each iteration:

   a. **Identify failing training cases.** From `/tmp/baseline_train.json` (or the previous
      iteration's training run), extract `failed_evals` with their assertion details.

   b. **Render the proposer prompt** (the proposer LLM is `proposer_model`):
      ```
      You are a skill engineer.

      Current skill file (full text):
        <skills/${skill_name}.md>

      The skill has eval set <skills/evals/${skill_name}.eval.json>. On the training split,
      the following cases FAILED:

        <list of failing cases with prompt + which assertion failed + the model output>

      Propose a MINIMAL edit to the skill file that addresses the most common failure pattern
      WITHOUT breaking any case that is currently passing. Output strictly:

        1. A unified diff (markdown ```diff fence) for skills/${skill_name}.md
        2. One paragraph rationale: which failure pattern, why this won't break passing cases
        3. Predicted held-out pass-rate delta

      FORBIDDEN edits: changing the skill's `name`, the public `inputs` contract, or the
      `eval_artifact` field. PERMITTED: TRIGGER/SKIP guidance, pass_criteria thresholds,
      procedure steps, failure modes section.
      ```

   c. **Apply the diff** to a candidate copy of the skill file. Validate the patched file with
      `scripts/validate_skills.py` — if frontmatter broke, discard this iteration's proposal
      and continue (or stop at max_iterations).

   d. **Re-run the harness on the training set** with the candidate. If candidate train pass-
      rate ≥ baseline + `improvement_threshold`, accept the candidate as the new baseline and
      proceed to step 6. Else revert and try another iteration.

6. **Final validation on the held-out test set.** This is the only place the test set is read:
   ```bash
   cp /tmp/candidate_skill.md "skills/${skill_name}.md"
   python scripts/run_skill_eval.py \
     --skill "$skill_name" --eval-set /tmp/eval_test.json \
     --output /tmp/candidate_test.json
   CANDIDATE_TEST=$(jq -r .assertion_pass_rate /tmp/candidate_test.json)
   ```

7. **Karpathy primitive.**
   ```bash
   GAIN=$(echo "$CANDIDATE_TEST - $BASELINE_TEST" | bc -l)
   if (( $(echo "$GAIN >= $improvement_threshold" | bc -l) )) && [[ "$dry_run" != "true" ]]; then
     git add "skills/${skill_name}.md"
     git commit -m "improve-skill ${skill_name}: held-out pass-rate +${GAIN}"
     DECISION=committed
   else
     git checkout "skills/${skill_name}.md"   # revert
     DECISION=reverted
   fi
   ```

8. **Write the audit record.**

   ```json
   {
     "skill": "train",
     "baseline_sha": "abc1234",
     "baseline_train_pass_rate": 0.50,
     "baseline_test_pass_rate": 0.40,
     "iterations": [
       {"i": 1, "candidate_train_pass_rate": 0.83, "accepted": true, "rationale": "..."}
     ],
     "candidate_test_pass_rate": 0.80,
     "gain": 0.40,
     "decision": "committed",
     "commit_sha": "def5678",
     "edit_diff": "--- a/skills/train.md\n+++ b/skills/train.md\n@@ ...",
     "decided_at": "2026-04-25T22:14:00Z"
   }
   ```

## Self-evaluation

PASS if the audit JSON exists with the schema above AND the git state matches `decision` (new
commit on HEAD if `committed`; no diff vs baseline_sha if `reverted`).

## Why the held-out validation matters

Without the train/test split, the proposer overfits the skill to the failing cases it can see.
The 40% hold-out is what protects the loop from Goodharting its own eval set — same logic as
the `noop` status in `verdict.py`, but applied at the meta-skill layer.

## Failure modes

- **Eval set too small (< 5 cases)**: split is meaningless. `aborted_min_eval_set_size`. Add
  more cases via skill-creator or by hand.
- **Proposer's diff breaks frontmatter**: `scripts/validate_skills.py` catches it, the
  iteration's candidate is discarded.
- **Held-out gain < threshold despite training-split gain**: classic overfitting; the loop
  reverts. Karpathy-discipline.
- **Two consecutive runs revert**: the failures aren't skill-file-addressable; escalate.

## Adapting to your runtime

- **Claude Code with the `skill-creator` plugin**: this skill's procedure is essentially what
  `skill-creator` already does. Point `skill-creator` at `skills/evals/<name>.eval.json` and
  it runs the same loop.
- **Standalone**: this Markdown is invokable as a procedural runbook in any agent runtime
  that can register tools and call LLMs.

## Pairs with

- **/meta-eval** — provides `candidates_for_improvement`
- **/improve-skills-auto** — drives this for every flagged candidate, unattended

## Prior art

The 60/40 train/test split, the 3-samples-per-query stabiliser, the diff-proposer + held-out
validation pattern are from the [Anthropic skill-creator](https://github.com/anthropics/skills).
fsvlm's contribution is the eval JSONs for its own skill catalog and a runtime-portable
harness.
