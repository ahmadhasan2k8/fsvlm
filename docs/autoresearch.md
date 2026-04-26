# Autoresearch — fsvlm's Adaptation of Karpathy's Loop for Benchmark-Driven Research

> **Credit up front.** The "autoresearch loop" pattern is **[Andrej Karpathy's](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/)**.
> Karpathy released a ~630-line autoresearch script in March 2026 that ran 700 experiments over
> two days and discovered 20 optimizations to its own training code. The named pattern, the
> propose → train → evaluate → commit-or-revert cycle, and the minimal three-component recipe
> (one editable file, one objective metric, one time budget per experiment) are his. **fsvlm did
> not invent the loop.** What this page documents is fsvlm's *adaptation* of the pattern for
> *benchmark-driven research* — the layer above Karpathy's minimal loop where we add
> per-category sweeps, multi-seed cells, an explicit `noop` status, expert-agent review, and a
> pre-registered taxonomy. Section 4.5 makes the "his vs ours" boundary explicit.

A meta-component of fsvlm: how sweeps are queued, executed, classified, expert-reviewed, and
committed back to the result log. The pattern generalizes to any benchmark-driven research loop
where you want disciplined iteration without giving up human judgment. This page documents what
the loop is, the pieces of it that ship in this repo, a worked example, and recipes for adapting
it to other tools.

> **Honest framing.** "Semi-autonomous" means a human still picks the hypotheses, reviews
> verdicts, and decides when to halt. The loop accelerates iteration, it does not replace
> judgment. The whole point is to *speed up the cycle of `hypothesis → run → analyze → decide`*
> while keeping the decisions auditable.

---

## 1. The pattern

```
       ┌────────────────────────────────────────┐
       │  hypothesis  (queue.json — committed)  │
       └──────────────────┬─────────────────────┘
                          ▼
              ┌────────────────────────┐
              │  skill: run sweep      │  ← procedural step with self-eval
              │  (run_sweep.sh, etc.)  │     (PASS / FAIL JSON output)
              └───────────┬────────────┘
                          ▼
       ┌──────────────────────────────────────────┐
       │  append rows → dataset_size_results.json │  ← provenance row
       │  (git_hash, recipe_version, status TBD)  │
       └───────────────────┬──────────────────────┘
                           ▼
              ┌────────────────────────┐
              │  verdict.py            │  ← keep/discard/noop/new_baseline
              │  (anti-goodhart tols)  │     classifier; pure compute
              └───────────┬────────────┘
                          ▼
              ┌──────────────────────────┐
              │  expert review           │  ← domain-expert sanity check
              │  (LLM agent or human)    │     before locking the verdict
              └───────────┬──────────────┘
                          ▼
       ┌──────────────────────────────────────────┐
       │  decision: next pass | halt | rollback   │  ← updates queue.json
       └──────────────────────────────────────────┘
```

Each box is a small, replaceable component. Each arrow is a structured handoff (JSON in, JSON
out). The whole loop runs on top of one append-only result log that is the ground truth.

---

## 2. What ships in this repo

The pieces of the loop that are public and work standalone:

### 2.1 `research/queue.json` — the hypothesis queue

A versioned JSON file listing each "pass" (a coherent block of runs answering one hypothesis).
Each pass entry carries:

- `id` — short slug (e.g., `pass3-curve-with-tiny-N`)
- `status` — `pending` / `running` / `completed` / `pending_user_approval`
- `hypothesis_primary` — one-sentence hypothesis being tested
- `description` — what the pass does and why
- `verdict_summary` — once completed, a one-line summary of what was learned
- (optionally) configuration for the sweep: dataset, categories, N-values, seeds, recipe knobs

**Schema-versioned** (`schema_version: 2`) so future loaders can migrate.

The queue lives in git. Updates are commits. This is the audit trail of *what was hypothesized*,
independent of *what was found*.

### 2.2 The sweep driver — `research/run_sweep.sh`

A single shell command per pass. Reads CLI flags (or env-vars for recipe overrides), calls the
fsvlm CLI, writes one row per (dataset, category, N, seed) to `research/dataset_size_results.json`.
The driver is intentionally thin — it composes existing `fsvlm train` + `fsvlm validate`
subprocess calls.

Companion drivers exist for variants: `research/passes/pass5b_driver.sh` (ICL ablation),
`research/passes/pass6_driver.sh`
(metal_nut rescue), etc. Each is a one-file shell script that pins exact recipe knobs.

### 2.3 The result log — `research/dataset_size_results.json`

**Append-only**. Every row carries:

```json
{
  "dataset": "mvtec", "category": "hazelnut", "n_samples": 30, "seed": 42,
  "auroc": 0.9438, "f1": 0.9216,
  "git_hash": "eeaa71d…", "git_short": "eeaa71d", "git_dirty": false,
  "recipe_version": "v0.3-curve",
  "status": "keep",
  "is_zero_shot": false, "notes": []
}
```

Bad runs are marked `discard`, never deleted. The log is the ground truth for every reported
number. Any number in `docs/benchmarks.md`, the README, or a future paper can be traced to a
specific commit + recipe version through this file.

### 2.4 The verdict classifier — `research/verdict.py`

Pure computation: aggregates rows by cell, compares each cell's metrics to the prior baseline,
and assigns one of four statuses:

| Status         | Trigger                                          | What it means                            |
|----------------|--------------------------------------------------|------------------------------------------|
| `new_baseline` | No prior baseline for this cell                  | This recipe is the new anchor            |
| `keep`         | AUROC or F1 lift ≥ `ABS_LIFT` (0.03)             | Real improvement; adopt the change       |
| `noop`         | All metrics within ±`NOOP_TOL` (0.02) of baseline| No behavioral change; **anti-goodharting** |
| `discard`      | Worse than baseline beyond `NOISE` (0.02)        | Roll back the recipe change              |

Thresholds live as constants at the top of `verdict.py` — tune them per project, not per row.

The classifier never edits the working tree or runs git. The calling layer (a human, an LLM
agent, or a CI job) decides whether to act on the verdict (e.g., `git checkout` the recipe
files on a `discard`).

### 2.5 The provenance contract

Every row links a measurement to:

- `git_hash` / `git_short` — the exact commit state of the repo when the run started
- `git_dirty` — whether the working tree had uncommitted changes (informs trust)
- `recipe_version` — a human-readable cohort tag (`v0`, `v0.1-extractor-fix`, `v0.3-curve`,
  `v0.4-icl-baseline`, …) for grouping

Combined: every reported number can be reproduced bit-for-bit by checking out `git_hash`,
applying the recipe knobs implied by `recipe_version`, and re-running the seed.

---

## 3. What lives outside this repo (and why)

The loop in fsvlm was driven by an LLM-based assistant orchestrating the steps. Those
assistant-specific artifacts are intentionally **not** in the public repo:

- The internal lab notebook (chronological narrative of every decision and its motivation)
- The expert-agent consultation transcripts (conversational JSON between the assistant and
  domain-expert sub-agents)
- The skill / agent definitions specific to the assistant

The public summary of all of that lives in [docs/research-log.md](research-log.md) — a
sanitized, assistant-agnostic distillation of the same narrative.

The **pattern** is portable; the specific assistant configuration is not. The next sections
describe the pattern in enough detail to rebuild it on any stack — Claude Code, OpenAI Agents
SDK, Cursor, your own Python orchestrator, or pen-and-paper with a co-author.

---

## 4. The semi-autonomous loop in detail

### 4.1 Skills — procedural steps with self-evaluation

A "skill" in this project's terminology is a small, reproducible procedure with a defined
output and a self-eval criterion. Generic shape:

```yaml
name: run-sweep
description: |
  TRIGGER when: user wants to evaluate a recipe change across N values
  SKIP when: user is asking a one-off question, not running a benchmark
inputs:
  - dataset (mvtec | visa | deeppcb | …)
  - categories (list)
  - n_values (list of int)
  - seeds (list of int)
  - recipe_overrides (env-var dict)
outputs:
  - rows appended to research/dataset_size_results.json
eval_artifact: research/dataset_size_results.json
pass_criteria: |
  - one row per (dataset, category, N, seed) tuple
  - every row has non-null auroc, f1, git_hash, recipe_version
  - no row's git_dirty is true (clean tree at run time)
```

The skill is a thin shell over `bash research/run_sweep.sh …`. The PASS / FAIL is mechanically
checkable from the output JSON — no human required for the routine case. Failures
(missing rows, dirty tree, exceptions) trigger escalation.

You can implement skills as: Claude Code skills, OpenAI Agent functions, CrewAI roles, plain
Makefile targets with a JSON post-condition checker, or shell scripts with a verifier hook.
What matters is the contract: *one input, one output, one self-eval that says PASS or FAIL.*

### 4.2 Expert agents — domain reviewers

After every pass, the loop consults a small number of "expert" reviewers before locking the
verdict. In this project the experts were two LLM sub-agents:

- A **VLM fine-tuning specialist** — reads the sweep results, judges whether training dynamics
  look healthy, recommends the next config or "halt"
- A **defect-detection / industrial-inspection specialist** — interprets per-category failure
  modes in terms a quality engineer would care about, identifies likely-classical-territory
  categories

Generic expert prompt skeleton (works with any chat-completion API):

```
You are a [VLM fine-tuning | industrial-inspection] specialist. You will be given:

  - the most recent sweep rows (JSON)
  - the queue's hypothesis for this pass
  - any prior expert recommendations relevant to this category

Your job:
  1. Read the data. Do not invent rows.
  2. Diagnose: what does the data show? (1-3 sentences)
  3. Recommend: next config to try, OR "halt — null result", OR "halt — done".
     For "next config", be specific (knobs + values).
  4. Confidence: low / medium / high.
  5. Rationale: 2-3 sentences.

Output strictly as JSON:
  { "diagnosis": "...", "recommendation": "...", "confidence": "...",
    "rationale": "..." }
```

The expert's output is structured JSON, archived alongside the pass. The human (or the loop's
decision step) can override the expert's recommendation; the override is logged.

The reason expert review is a *separate step* from the verdict classifier: `verdict.py` answers
"is this number better than the prior number?" — a mechanical question. The expert answers "is
this number *meaningful*?" — a contextual question. Conflating the two is how
benchmark-grinding loops Goodhart their own metrics.

### 4.3 The decision step

Inputs:

- The pass's hypothesis (from `queue.json`)
- The verdict statuses (from `verdict.py`)
- The expert recommendations (JSON from §4.2)
- The repo state (current branch, uncommitted changes)

Outputs (always one of):

1. **Next pass** — append a new pass to `queue.json`, set its status to `pending`. Loop continues.
2. **Halt — null result** — mark the current pass `completed` with a verdict_summary like
   "NULL — N=2 lift is within noise on all 3 categories"; loop ends until the human picks a new
   hypothesis.
3. **Halt — done** — methodology converged; pass moves to `completed`, results land in
   `docs/benchmarks.md`.
4. **Rollback** — verdict is `discard`; the loop runs `git checkout` on the recipe files,
   marks the pass `completed (rolled back)`, and surfaces the diagnosis to the human.

The decision step is the only place that *writes* to `queue.json` and the only place that *can*
mutate the repo. Constraining mutation to one step keeps the audit trail clean.

### 4.4 Anti-goodharting in the loop

Two design choices stop the loop from drifting toward a metric instead of an answer:

- **`noop` is a status.** A change that moves nothing is *not* a win. The loop treats it as no
  evidence and the recipe stays at baseline. Without this, anything within noise gets adopted
  and the recipe drifts.
- **Pre-registered taxonomies.** Defect subtypes are tagged `distinctive` vs `subtle` *before*
  results are observed and committed with a frozen ISO timestamp. If the tagging is edited
  after results land, the rows it explains are invalidated. Without this, the post-hoc
  narrative writes itself.

### 4.5 Karpathy's pattern vs what fsvlm adds — ranked by contribution strength

Karpathy's autoresearch provides **the loop** (propose → train → evaluate → commit-or-revert,
agent in a continuous cycle) and **the minimal contract** (one editable file, one objective
metric, one time budget per experiment). Both are used as-is. fsvlm's contributions wrap that
loop with the discipline you need when "did the metric go up?" is too coarse — many noisy small-N
cells, multiple metrics, an audit-driven release target.

The contributions, ranked honestly by strength (novelty × load-bearingness × what someone else
would actually adopt):

#### 🟢 Tier 1 — distinctive and load-bearing

1. **Expert-agent consultation as a separate step from the verdict classifier.** Karpathy's
   loop has no domain-review step — the optimisation target is a single objective metric and
   the agent's job is to move it. We split the cycle into a *mechanical* verdict (`verdict.py`:
   "is this number better than the prior one, beyond noise?") and a *contextual* expert review
   ("is this number *meaningful* given the category, the data, the failure modes?"). The split
   exists because on small-N benchmarks with noisy metrics it is easy for a loop to chase
   improvements that are statistically real but practically meaningless. The expert step is the
   anti-Goodhart guard. Implementation: a templated prompt skeleton (§4.2) + structured JSON
   output, archivable in the repo. Implementable on any chat-completion API. **This is the
   piece we'd most encourage others to adopt.**

2. **Skills as the loop's atomic step, with PASS/FAIL self-eval baked into YAML frontmatter.**
   Karpathy's loop is one editable file driving one metric. As soon as you scale beyond that —
   multiple sweep drivers, multiple data-prep steps, a real CLI surface — you need an audit-
   friendly contract for "did this step do what it said it would?". Our answer: every skill
   declares an `eval_artifact` (output JSON path) and a `pass_criteria` (a checkable list)
   upfront. The self-eval is mechanical; failures escalate. Generic shape in §4.1. The novelty
   isn't "function calling" — every agent framework has that — it's making the *self-eval
   criterion declarative and machine-checkable* so the loop can progress without a human
   inspecting every output.

#### 🟡 Tier 2 — small but load-bearing

3. **The `noop` status as a first-class verdict.** Karpathy's loop is binary: better → keep,
   not better → revert. That works for a single editable file driving a single loss; it breaks
   when many recipe knobs have noisy small-effect outcomes on small test sets, because anything
   within noise gets adopted and the recipe drifts. We add `noop` ("change moved nothing beyond
   ±0.02") as an explicit third status that holds the baseline. Tiny mechanically — ~10 lines
   in `verdict.py` — but it is the single most load-bearing anti-Goodhart primitive in the
   classifier. The fourth status, `new_baseline`, is bookkeeping.

4. **Pre-registered taxonomy with frozen ISO timestamp as a binding artifact.** Pre-registration
   itself is decades-old open-science discipline (clinical trials, replication-crisis-era
   psychology) — credit there, not here. What's distinctive is **integrating it with the loop's
   mutation gate**: the taxonomy is committed to the repo with a frozen ISO timestamp, and any
   post-hoc edit to it invalidates the rows it explains. The verdict classifier reads the
   timestamp; the loop refuses to lock a verdict against a taxonomy edited after the relevant
   rows landed. That binding is what makes pre-registration enforceable here instead of merely
   polite.

#### ⚪ Tier 3 — routine extensions, listed for completeness

5. **Versioned hypothesis queue (`queue.json`) with named passes.** Helpful for auditability;
   nothing intellectually distinctive — it is project management.

6. **Per-(dataset × category × N) sweeps for curve-shape observability.** Direct application of
   Karpathy's loop to a benchmark grid; the observation "this lets you see curve shape rather
   than a single scalar" is useful but obvious.

7. **Multi-seed cells, AUROC + F1 reported jointly with mean ± stdev.** Baseline statistical
   hygiene from the ML community at large; not a contribution, just the price of admission for
   credibility on small test sets.

8. **Provenance row schema (`git_hash` / `recipe_version` / `status` per row).** This is the
   Aether-style provenance row — adopted, not invented.

#### Short version

If you take **only two things** from this doc into your own loop, take **(1) the expert-review
step as a guard against the verdict classifier auto-locking on noisy improvements**, and **(2)
the skills-with-declarative-self-eval contract so the loop can iterate without a human in
every loop body**. Those are the load-bearing additions to Karpathy's minimal recipe. The rest
is plumbing or borrowed discipline.

---

## 5. A worked example — Pass 3 (the AUROC-vs-N curve sweep)

| Step | What happened |
|------|---------------|
| **Hypothesis** | "If even N=2 produces useful lift on distinctive-defect categories, the tiger-analogy holds at the limit — one or two examples is enough for some defects." Logged in `queue.json` as `pass3-curve-with-tiny-N`. |
| **Skill: run sweep** | `bash research/run_sweep.sh --datasets mvtec visa deeppcb --categories hazelnut candle pcb --n-values 0 2 3 5 10 20 30 40 60 100 --seeds 42 1337 7 --epochs 3` — appends ~90 rows to `dataset_size_results.json` with `recipe_version=v0.3-curve`. |
| **Append-only log** | Each row carries `git_hash`, `recipe_version`, `is_zero_shot`. Self-eval: every (dataset, category, N, seed) tuple has exactly one row with non-null AUROC. PASS. |
| **Verdict.py** | Per-cell mean ± stdev computed; vs the Pass 1 baseline (N=30, recipe v0.1) the new cells classify as: most `new_baseline` (no prior tiny-N data), some `keep` (DeepPCB N=60/100 monotonic lift), some `noop` (hazelnut N=100 within noise). |
| **Expert review** | The VLM-specialist agent diagnosed "knee at N=2 on hazelnut and candle; flat-at-ZS on DeepPCB through N=40 — different curve shape, recommend ICL ablation as next pass." Confidence: high. |
| **Decision** | Next pass = an ICL-vs-FT ablation at the same cells. Pass 3 marked completed with verdict_summary "Curves confirm tiger-analogy at N=2 on distinctive cats. DeepPCB has 'delayed knee' shape — already at-ceiling at ZS, small monotonic lift only at N≥60." |
| **Audit** | All of the above is reconstructable from public files: `queue.json` for the hypothesis trail, `dataset_size_results.json` for the rows, `verdict.py` for the classification. The expert's JSON output and the human's decision are logged outside the public repo (lab notebook). |

The narrative version of the same arc lives in [docs/research-log.md](research-log.md) for
readers who want the prose.

---

## 6. Adapting the pattern to your own work

You don't need fsvlm-the-tool to use the autoresearch pattern. The loop itself is
[Karpathy's](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/) —
his ~630-line script is the canonical reference for the minimal version, and Shopify and others
are already adapting it across non-ML domains. What fsvlm contributes (per §4.5) is the
benchmarking discipline you wrap around the loop when one objective metric isn't enough. The
minimal set you need to bring either of them into your own project:

### 6.1 The minimal setup

1. **A versioned hypothesis file.** JSON or YAML. One entry per coherent block of runs.
   Status field, hypothesis field, verdict_summary field. Lives in git.
2. **An append-only result log.** One row per atomic (config × seed) run. Every row has the
   metrics + git_hash + recipe_version. Never edit; mark bad rows as discard.
3. **A verdict classifier.** ~100 lines of Python. Compares each cell to a prior baseline and
   labels keep / discard / noop / new_baseline with explicit numerical thresholds. Make `noop`
   a status; that's the anti-goodharting move.
4. **At least one expert review per pass.** Doesn't need to be an LLM — could be a co-author
   over Slack with a templated prompt. The point is to separate "is this number better?" from
   "is this number meaningful?".
5. **A decision step that's the only writer.** Single point that mutates the queue and the
   recipe. Audit-friendly.

### 6.2 If you're using Claude Code or another agent framework

Wire each box of the loop in §1 to a tool / skill / function. The agent drives the loop, the
human approves transitions between passes. The expert sub-agents are just additional functions
the orchestrator calls — same prompt skeleton as §4.2.

### 6.3 If you're a solo researcher with no agent stack

- Replace "skill" with "shell script + assert at the end"
- Replace "expert agent" with a templated prompt you paste into your favourite chatbot once per pass
- Replace "decision step" with a five-minute weekly review where you read the verdicts and
  update `queue.json` by hand
- Keep everything else identical

You'll lose iteration speed compared to a fully wired-up loop, but the discipline (provenance,
anti-goodharting, pre-registration) is what makes the results trustworthy. That's preserved.

### 6.4 What to copy from this repo as a starting point

- `research/verdict.py` — the classifier. Apache 2.0; copy-and-adapt freely.
- `research/queue.json` — the schema is in `schema_version: 2`. Copy the structure.
- `research/run_sweep.sh` — pattern for a thin sweep driver that wraps an existing CLI.
- `research/dataset_size_results.json` — see the row format for a working provenance contract.

The fsvlm CLI surface (`fsvlm train`, `fsvlm validate`) is what these drivers wrap. If your
work uses a different model / dataset / task, swap that in — the loop machinery is unchanged.

---

## 7. Honest limits

- **The loop accelerates routine iteration; it does not generate hypotheses.** The interesting
  question — "what should I test next?" — remains a human's job. The loop only makes the
  testing-and-classifying part fast and disciplined.
- **Expert agents can be wrong.** They are pattern-matchers over training data; they are not
  oracle physicists. The loop logs their recommendations so wrong calls can be reviewed later;
  it does not let them auto-merge.
- **`verdict.py` is opinionated.** The thresholds (0.03 lift, 0.02 noop) are tuned for AUROC /
  F1 in the 0.7–0.99 range on small test sets. For other metrics or other regimes, retune.
- **Provenance is only as good as the discipline.** If anyone hand-edits the results JSON or
  bypasses `git_dirty` checks, the audit trail is broken. Code-review the drivers; don't trust
  silently.
- **Pre-registration is most of the value.** A taxonomy committed *after* results are observed
  is worse than no taxonomy. The frozen ISO timestamp is the load-bearing detail; keep it.

---

## 8. Where to read more

- [docs/research-log.md](research-log.md) — the assistant-agnostic narrative of the actual research
  arc that produced the v0.1 results, including the wrong turns.
- [docs/benchmarks.md](benchmarks.md) — the methodology commitments and dataset coverage.
- [POSITIONING.md](../POSITIONING.md#self-evaluating-framework) — the self-evaluation features
  that make the loop's output trustworthy.
- The append-only `research/dataset_size_results.json` — the ground truth.

---

## 9. Acknowledgements

The pattern draws on, and explicitly extends, prior work — none of which is fsvlm's:

- **Andrej Karpathy — autoresearch (March 2026)** — *the loop itself.* The named pattern, the
  propose → train → evaluate → commit-or-revert cycle, the minimal three-component recipe
  (one editable file, one objective metric, one time budget per experiment). The whole shape of
  what this page describes follows Karpathy's release; fsvlm's contribution is the
  benchmarking-discipline wrapper around the loop documented in §4.5, not the loop. Background:
  the [Fortune coverage](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/),
  the [DataCamp guide](https://www.datacamp.com/tutorial/guide-to-autoresearch),
  [Shopify's adoption write-up](https://shopify.engineering/autoresearch).
- **Andrej Karpathy — "A Recipe for Training Neural Networks" (2019)** — separate from the
  autoresearch loop: the older training-discipline blog post for the rules that null results
  matter, you should eyeball a fixed batch, and one hypothesis per experiment beats grinding.
  Influences §4.4's anti-goodharting choices.
- **The Anthropic-style sub-agent delegation pattern** — the shape of the expert-reviewer step
  in §4.2.
- **The Aether-style append-only result log with git provenance** — the shape of the row schema
  in §2.3 and the four-status verdict classifier in §2.4.
- **Ben Recht's "outside view" framing** on benchmark hygiene — the reason `noop` is a status
  and not a celebration.

**What this project actually contributes**, ranked by strength (full version in §4.5):

1. **Expert-agent consultation as a separate step from the verdict classifier** — the
   anti-Goodhart contextual-review guard. Tier 1.
2. **Skills with declarative PASS/FAIL self-eval (`eval_artifact` + `pass_criteria` YAML
   frontmatter)** — the audit-friendly contract that lets the loop iterate without a human in
   every loop body. Tier 1.
3. **The `noop` status as a first-class verdict** — small but load-bearing anti-drift
   primitive in `verdict.py`. Tier 2.
4. **Binding pre-registration via frozen ISO timestamp on the taxonomy** — pre-registration
   itself is older than us; the binding-to-the-loop-gate integration is distinctive. Tier 2.

The rest (the queue, the per-cell sweeps, multi-seed reporting, the provenance row schema) is
plumbing or adopted from prior open-science / ML-hygiene / Aether discipline.
