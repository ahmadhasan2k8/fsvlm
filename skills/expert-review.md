---
name: expert-review
description: |
  Consult a domain-expert reviewer (LLM agent, human peer, or a co-author) on a sweep's results
  before locking the verdict. Separates the *mechanical* "is this number better?" question
  (answered by /verdict) from the *contextual* "is this number meaningful?" question. The
  load-bearing anti-Goodhart guard between verdict and decision in the autoresearch loop.
  TRIGGER when: /verdict has classified a cohort and the loop is about to commit a recipe
  change OR halt; OR when monotonicity broke in /tiered-eval; OR when /sweep produced cells
  that look surprising.
  SKIP when: results are unambiguously a regression (run rollback directly) or unambiguously
  a `noop` (no decision to defend).
inputs:
  - role (string, required) — training-specialist | domain-specialist | <your-role>
  - results_subset (path, required) — JSON file with the rows under review (typically
    one cohort filtered from dataset_size_results.json)
  - hypothesis (string, required) — the queue's hypothesis for the pass under review
  - prior_reviews (list of paths, optional) — prior /expert-review outputs for context
  - llm_endpoint (string, optional) — chat-completion endpoint URL; if unset, render the
    prompt and let the user paste it into their preferred chatbot manually
eval_artifact: |
  Role-conditional path:
    role in {training-specialist, domain-specialist, <other method roles>} →
      research/expert_reviews/${role}_${timestamp}.json    (committed, public)
    role == strategy-reviewer (or any role ending in -strategy) →
      _local/paper_workspace/expert_reviews/${role}_${timestamp}.json    (gitignored, private)
pass_criteria:
  - file exists at eval_artifact
  - JSON keys: role, diagnosis, recommendation, confidence, rationale
  - confidence in {low, medium, high}
  - recommendation is one of: "next_config: <spec>" | "halt: null result" | "halt: done" | "rollback"
  - if eval_artifact is under research/, the file contains NONE of the forbidden phrases
    (publishable / the paper / manuscript / submission / conference / workshop deadline /
    arXiv / camera-ready / reviewer #N / novelty / publication-worthy)
escalation: |
  /expert-review is the contextual-judgment step itself; it does not escalate further. Its
  output feeds the decision step (the human, or /autoresearch's decision body). If two
  successive expert reviews disagree, surface both to the user and halt the loop.
---

# Skill: expert-review

## Purpose

> **Honest scope.** `/expert-review` is an orchestrator skill — its core action is calling
> a chat-completion API (or rendering a prompt for a human reviewer) and parsing the
> response. That dispatch + parsing needs a runtime: Claude Code can call this skill
> directly (it IS an LLM); other runtimes need to wire it to their preferred API; manual
> mode renders the prompt and you paste the response back. Direct shell invocation via
> `bash skills/_run.sh expert-review` returns a "needs runtime" error pointing here.

The Tier-1 contribution of fsvlm's autoresearch adaptation: a structured review step between
the mechanical verdict and the recipe-mutation decision. Implementable on any chat-completion
API, or by a human peer with a templated prompt.

## Procedure

1. Render the prompt from the template:
   ```
   You are a [role] specialist. You will be given:

     - the most recent sweep rows (JSON, attached)
     - the queue's hypothesis for this pass: <hypothesis>
     - any prior expert recommendations relevant to this category: <prior_reviews>

   Your job:
     1. Read the data. Do not invent rows.
     2. Diagnose: what does the data show? (1-3 sentences)
     3. Recommend: next config to try, OR "halt — null result", OR "halt — done".
        For "next config", be specific (knob = value pairs).
     4. Confidence: low / medium / high.
     5. Rationale: 2-3 sentences explaining the recommendation.

   Hard constraint on output (METHOD ROLES ONLY — training-specialist,
   domain-specialist, and any other role NOT ending in -strategy):
     Frame every sentence as PROJECT-DECISION reasoning, not strategy reasoning.
     Do NOT use the words "publishable", "the paper", "manuscript", "submission",
     "conference", "workshop deadline", "arXiv", "camera-ready", "reviewer #N",
     "novelty", "contribution to the literature", or "publication-worthy".
     The output JSON is committed to a public repo and consumed by the autoresearch
     loop's decision step — keep it about the data and the next experimental
     action only. Strategy-flavored thinking has its own role (see below).

   For STRATEGY ROLES (strategy-reviewer or any role ending in -strategy):
     The constraint above is INVERTED. Strategy roles are explicitly invited to
     reason about framing, citation requirements, venue fit, novelty arguments,
     and how findings would land with a critical reader. The output goes to
     _local/paper_workspace/expert_reviews/ which is gitignored — your
     reasoning will not appear publicly. Be candid; that's the point.

   Output strictly as JSON:
     { "role": "<role>", "diagnosis": "...", "recommendation": "...",
       "confidence": "...", "rationale": "..." }
   ```

2. If `llm_endpoint` is set, POST the prompt + attachments and capture the response. Otherwise,
   write the rendered prompt to stdout for manual paste-and-reply.

3. Parse the JSON response. Sanity-check it has the required keys; if not, fail loudly rather
   than silently archive a malformed review.

4. Archive the response under `research/expert_reviews/<role>_<timestamp>.json`. The reviews
   directory is part of the audit trail for the loop.

## Role templates

`role` is a free-form string. Two roles seed the catalog; add your own as needed.

### `training-specialist`
Focus: training dynamics, LR-vs-quantization interactions, LoRA-rank choice, overfit signals,
recipe drift across passes. Best for diagnosing why a sweep cell underperformed or why
monotonicity broke.

### `domain-specialist`
Focus: per-category failure modes, what kinds of defects this VLM should/shouldn't see,
whether the dataset's labeling is the bottleneck. Best for translating ML metrics into
domain-engineering language. Method role — output public.

### `strategy-reviewer` (private — output gitignored)

Focus: framing decisions, citation requirements, venue fit, novelty arguments, the
"how would a critical outside reader receive this?" angle. Has access to the same data
as method roles but is explicitly invited to think strategically about how findings
should be presented externally.

The eval_artifact for this role lives at
`_local/paper_workspace/expert_reviews/strategy-reviewer_<timestamp>.json` (gitignored).
The role exists precisely so strategy thinking happens with the loop's full data context
without leaking into public artifacts.

Use this role when:
  - About to lock a major framing decision (which finding leads, which is footnote)
  - Considering citing or omitting a piece of prior work
  - Weighing whether a result is ready for external argument vs. needs more evidence
  - Asked "is this a real contribution, or am I overclaiming?"

The pre-commit hook + the eval_artifact path constraint make it mechanically hard for
strategy-reviewer output to land in a public commit. The honest workflow is:
strategy-reviewer thinks freely, writes to `_local/`, you read it, you decide what (if
anything) makes it into the public README / docs / paper draft.

### Adding a custom role

Drop a `prompts/expert_<role>.txt` file with role-specific framing; the procedure above
will pick it up via the `role` parameter. Method roles output to `research/expert_reviews/`
(public, with the forbidden-words constraint). Strategy roles must end their name in
`-strategy` and output to `_local/paper_workspace/expert_reviews/` (private, no constraint).
The role-name suffix is what routes the output — keep the convention to keep the guard.

## Self-evaluation

PASS if a JSON file exists at `eval_artifact` with all required keys and confidence in the
allowed set. The CONTENT of the review is not validated mechanically — that's the point of
contextual review.

## Failure modes

- **Reviewer hallucinates rows or numbers**: detected by hash-checking the diagnosis text
  against the input rows. If the reviewer cited a number that isn't in `results_subset`, mark
  the review `quality: low` in the archive and re-run.
- **Reviewer always says "next config: [vague]"**: prompt isn't strict enough; add explicit
  examples of bad-faith refusal-to-decide.
- **Two reviewers disagree**: surface to the user; do not auto-resolve. Disagreement is signal.

## Adapting to your runtime

- **Claude Code**: drop into `~/.claude/skills/`; the skill body invokes a sub-agent with the
  role-templated prompt
- **OpenAI Agents SDK**: register `expert_review(role, results_subset, hypothesis, prior_reviews,
  llm_endpoint)` as a tool; back it with an Assistant configured per role
- **Solo researcher with no agent stack**: render the prompt to stdout, paste into your favourite
  chatbot, paste the response back. Same audit trail.

## Pairs with

- **/verdict** — its mechanical output is the input to the contextual review here
- **/autoresearch** — orchestrates the verdict → expert-review → decision pipeline
- **/sweep** — when expert says `next_config`, that becomes the input to the next sweep
