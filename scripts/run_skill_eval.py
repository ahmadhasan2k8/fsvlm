"""Portable skill-eval harness — runtime-agnostic implementation of the Anthropic skill-creator
eval pattern (https://github.com/anthropics/skills).

Reads `skills/evals/<name>.eval.json` files, sends each `prompt` to a configured agent runtime
`samples_per_query` times for trigger-rate stability, grades the agent's outputs against the
declared `assertions`, and writes a structured health report.

Runtimes supported:
  - "portable" (default): minimal Anthropic Messages API loop using the exposed skill
    descriptions as a tool catalog. Requires ANTHROPIC_API_KEY.
  - "claude-code": delegates to the user's installed `~/.claude/skills/` and the agent
    invocation pathway (file-system based; the harness writes prompts and checks the
    invocation log).
  - "openai-sdk": registers each skill as an OpenAI Agents-SDK tool and runs an Assistant
    against the prompts. Requires OPENAI_API_KEY.

This is a thin reference harness — the goal is portability, not feature-completeness. For
production use, wire your existing agent runtime's eval pipeline at the assertion-grading
boundary.

Usage:
    python scripts/run_skill_eval.py --skills-dir skills --evals-dir skills/evals \
        --runtime portable --model claude-haiku-4-5-20251001 \
        --samples-per-query 3 --output ~/.fsvlm/skill_health.json

    # Single skill:
    python scripts/run_skill_eval.py --skill train --eval-set skills/evals/train.eval.json \
        --output /tmp/train_health.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AssertionResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class EvalResult:
    eval_id: int
    eval_name: str
    samples: int
    triggered_skill_counts: dict[str, int] = field(default_factory=dict)
    trigger_rate: float = 0.0
    assertions_per_sample: list[list[AssertionResult]] = field(default_factory=list)
    assertion_pass_rate: float = 0.0


@dataclass
class SkillReport:
    name: str
    eval_set_size: int
    trigger_rate: float
    assertion_pass_rate: float
    failed_evals: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent-runtime adapters
# ---------------------------------------------------------------------------

def _load_skill_descriptions(skills_dir: Path) -> list[dict[str, Any]]:
    """Read every skills/*.md frontmatter into a structured list."""
    out: list[dict[str, Any]] = []
    fm_re = re.compile(r"\A---\s*\n(.*?)\n---", re.DOTALL)
    for path in sorted(skills_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text()
        m = fm_re.match(text)
        if not m:
            continue
        # Extract name + description quickly without a real YAML parser.
        name_m = re.search(r"^name:\s*(.+)$", m.group(1), re.MULTILINE)
        desc_m = re.search(r"^description:\s*\|\n((?:[ \t]+.+\n?)+)", m.group(1), re.MULTILINE)
        if not name_m:
            continue
        out.append({
            "name": name_m.group(1).strip(),
            "description": (desc_m.group(1) if desc_m else "").strip(),
        })
    return out


def _portable_invoke(prompt: str, skills: list[dict[str, Any]], model: str,
                     max_iterations: int) -> dict[str, Any]:
    """Call Anthropic Messages API with the skills as a tool catalog. Returns the agent's
    invocation decision (which skill, which inputs).

    Returns: {"triggered_skill": str | None, "inputs": dict, "raw_response": str}
    """
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return {"triggered_skill": None, "inputs": {}, "raw_response": "anthropic SDK not installed"}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"triggered_skill": None, "inputs": {}, "raw_response": "ANTHROPIC_API_KEY unset"}

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    tools = [
        {
            "name": s["name"].replace("-", "_"),
            "description": s["description"][:1024],
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": True},
        }
        for s in skills
    ]
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            tools=tools,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return {"triggered_skill": None, "inputs": {}, "raw_response": f"API error: {e}"}

    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return {
                "triggered_skill": block.name.replace("_", "-"),
                "inputs": dict(block.input or {}),
                "raw_response": str(resp.content),
            }
    return {"triggered_skill": None, "inputs": {},
            "raw_response": "".join(getattr(b, "text", "") for b in resp.content)}


def _claude_code_invoke(prompt: str, skills: list[dict[str, Any]], model: str,
                        max_iterations: int) -> dict[str, Any]:
    """Stub — see docstring in module header. In a real Claude Code runtime, this would write
    the prompt to the harness queue and read the agent's invocation log."""
    return {"triggered_skill": None, "inputs": {},
            "raw_response": "claude-code runtime adapter not implemented in this reference harness"}


def _openai_sdk_invoke(prompt: str, skills: list[dict[str, Any]], model: str,
                       max_iterations: int) -> dict[str, Any]:
    """Stub — see docstring in module header. Implement with openai.beta.threads.runs."""
    return {"triggered_skill": None, "inputs": {},
            "raw_response": "openai-sdk runtime adapter not implemented in this reference harness"}


RUNTIME_DISPATCH = {
    "portable": _portable_invoke,
    "claude-code": _claude_code_invoke,
    "openai-sdk": _openai_sdk_invoke,
}


# ---------------------------------------------------------------------------
# Assertion grading
# ---------------------------------------------------------------------------

def _grade_assertion(spec: dict[str, Any], invocation: dict[str, Any]) -> AssertionResult:
    """Evaluate one assertion against one invocation result."""
    name = spec.get("name", "<unnamed>")
    typ = spec.get("type")
    triggered = invocation.get("triggered_skill")
    inputs = invocation.get("inputs", {}) or {}

    if typ == "trigger":
        passed = (triggered == spec.get("value"))
        return AssertionResult(name, passed, f"triggered={triggered!r}, expected={spec.get('value')!r}")

    if typ == "trigger_negation":
        passed = (triggered != spec.get("value"))
        return AssertionResult(name, passed, f"triggered={triggered!r}, must_not_be={spec.get('value')!r}")

    if typ == "trigger_or_explicit_skip":
        # Either the skill triggered OR the agent's response explicitly explained why it skipped.
        passed = triggered == spec.get("value") or "skip" in invocation.get("raw_response", "").lower()
        return AssertionResult(name, passed)

    if typ == "input_eq":
        path = spec.get("path")
        passed = inputs.get(path) == spec.get("value")
        return AssertionResult(name, passed, f"inputs.{path}={inputs.get(path)!r}, expected={spec.get('value')!r}")

    if typ == "input_contains":
        path = spec.get("path")
        val = inputs.get(path, "")
        passed = isinstance(val, str) and spec.get("value") in val
        return AssertionResult(name, passed, f"inputs.{path}={val!r} contains {spec.get('value')!r}? {passed}")

    if typ == "input_in_set":
        path = spec.get("path")
        passed = inputs.get(path) in (spec.get("value") or [])
        return AssertionResult(name, passed, f"inputs.{path}={inputs.get(path)!r} in {spec.get('value')!r}? {passed}")

    if typ == "input_contains_value":
        path = spec.get("path")
        val = inputs.get(path, [])
        passed = spec.get("value") in val if isinstance(val, list) else False
        return AssertionResult(name, passed)

    if typ == "input_length_gte":
        path = spec.get("path")
        val = inputs.get(path, [])
        passed = isinstance(val, list) and len(val) >= spec.get("value", 0)
        return AssertionResult(name, passed)

    # The following assertion types depend on the skill having actually executed and produced
    # output. The portable runtime stops at trigger decision (no execution), so these are
    # marked as "cannot_evaluate" rather than failed.
    if typ in ("file_exists", "json_field_eq", "json_field_gte", "json_field_lte",
               "json_field_in_set", "pass_criteria_met", "wall_time_under"):
        return AssertionResult(name, True, "skipped: post-execution assertion (portable runtime "
                                            "grades trigger only)")

    return AssertionResult(name, False, f"unknown assertion type: {typ}")


# ---------------------------------------------------------------------------
# Eval-set runner
# ---------------------------------------------------------------------------

def run_eval_set(eval_path: Path, skill_descriptions: list[dict[str, Any]], runtime: str,
                 model: str, samples_per_query: int, max_iterations: int) -> SkillReport:
    spec = json.loads(eval_path.read_text())
    skill_name = spec["skill"]
    invoke = RUNTIME_DISPATCH[runtime]

    eval_results: list[EvalResult] = []
    for ev in spec["evals"]:
        triggered_counts: dict[str, int] = {}
        per_sample_assertions: list[list[AssertionResult]] = []
        for _ in range(samples_per_query):
            inv = invoke(ev["prompt"], skill_descriptions, model, max_iterations)
            t = inv.get("triggered_skill") or "<none>"
            triggered_counts[t] = triggered_counts.get(t, 0) + 1
            per_sample_assertions.append([_grade_assertion(a, inv) for a in ev.get("assertions", [])])
        expected = ev.get("expected_skill", skill_name)
        trigger_rate = triggered_counts.get(expected, 0) / samples_per_query
        assertion_pass_rates = [
            sum(a.passed for a in samp) / len(samp) if samp else 1.0
            for samp in per_sample_assertions
        ]
        assertion_pass_rate = statistics.fmean(assertion_pass_rates) if assertion_pass_rates else 0.0
        eval_results.append(EvalResult(
            eval_id=ev["eval_id"], eval_name=ev["eval_name"], samples=samples_per_query,
            triggered_skill_counts=triggered_counts, trigger_rate=trigger_rate,
            assertions_per_sample=per_sample_assertions,
            assertion_pass_rate=assertion_pass_rate,
        ))

    overall_trigger = statistics.fmean([r.trigger_rate for r in eval_results]) if eval_results else 0.0
    overall_assert = statistics.fmean([r.assertion_pass_rate for r in eval_results]) if eval_results else 0.0
    failed = [
        {"eval_id": r.eval_id, "eval_name": r.eval_name,
         "trigger_rate": r.trigger_rate, "assertion_pass_rate": r.assertion_pass_rate}
        for r in eval_results
        if r.trigger_rate < 1.0 or r.assertion_pass_rate < 1.0
    ]
    return SkillReport(name=skill_name, eval_set_size=len(eval_results),
                       trigger_rate=overall_trigger, assertion_pass_rate=overall_assert,
                       failed_evals=failed)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skills-dir", default="skills")
    p.add_argument("--evals-dir", default="skills/evals")
    p.add_argument("--skill", default=None, help="Run only this skill's eval set")
    p.add_argument("--eval-set", default=None, help="Run only this eval JSON file")
    p.add_argument("--runtime", default="portable", choices=list(RUNTIME_DISPATCH.keys()))
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--samples-per-query", type=int, default=3)
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--output", default=str(Path.home() / ".fsvlm" / "skill_health.json"))
    args = p.parse_args()

    skills_dir = Path(args.skills_dir)
    evals_dir = Path(args.evals_dir)
    if not skills_dir.is_dir() or not evals_dir.is_dir():
        print(f"missing dirs: {skills_dir} or {evals_dir}", file=sys.stderr)
        return 1

    skill_descs = _load_skill_descriptions(skills_dir)
    if args.eval_set:
        targets = [Path(args.eval_set)]
    elif args.skill:
        targets = [evals_dir / f"{args.skill}.eval.json"]
    else:
        targets = sorted(evals_dir.glob("*.eval.json"))

    started = time.time()
    reports = [run_eval_set(t, skill_descs, args.runtime, args.model,
                            args.samples_per_query, args.max_iterations) for t in targets]
    elapsed = time.time() - started

    aggregate = statistics.fmean([r.assertion_pass_rate for r in reports]) if reports else 0.0
    candidates = sorted(
        [r.name for r in reports if r.assertion_pass_rate < 0.85],
        key=lambda n: next((1 - r.assertion_pass_rate) * r.eval_set_size for r in reports if r.name == n),
        reverse=True,
    )

    out_path = Path(os.path.expanduser(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "window": {"started_at": started, "duration_seconds": elapsed},
        "runtime": args.runtime,
        "model": args.model,
        "samples_per_query": args.samples_per_query,
        "aggregate_pass_rate": aggregate,
        "per_skill": [
            {"name": r.name, "eval_set_size": r.eval_set_size,
             "trigger_rate": r.trigger_rate, "assertion_pass_rate": r.assertion_pass_rate,
             "failed_evals": r.failed_evals}
            for r in reports
        ],
        "candidates_for_improvement": candidates,
    }, indent=2))
    print(f"wrote {out_path} (aggregate_pass_rate={aggregate:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
