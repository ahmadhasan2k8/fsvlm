"""Validate every skill under skills/ has a complete, well-formed frontmatter.

Checks:
- Starts with YAML frontmatter fenced by ``---``
- Required keys present: name, description
- Recommended keys present (warn if missing): eval_artifact, pass_criteria
- description contains TRIGGER and SKIP guidance
- A sibling eval JSON exists at skills/evals/<name>.eval.json (recommended)

Exits non-zero on any failure. Prints a JSON summary.
"""

import json
import re
import sys
from pathlib import Path

REQUIRED = {"name", "description"}
RECOMMENDED = {"eval_artifact", "pass_criteria"}

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
EVALS_DIR = SKILLS_DIR / "evals"
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> dict[str, str] | None:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    body = m.group(1)
    out: dict[str, str] = {}
    current_key: str | None = None
    for line in body.splitlines():
        if not line.strip():
            continue
        key_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if key_match and not line.startswith(" "):
            current_key = key_match.group(1)
            out[current_key] = key_match.group(2).strip()
        elif current_key is not None and line.startswith(" "):
            out[current_key] += " " + line.strip()
    return out


def validate(path: Path) -> dict:
    result = {
        "file": path.name,
        "status": "pass",
        "missing_required": [],
        "missing_recommended": [],
        "trigger_skip_present": False,
        "name_matches_file": False,
    }
    text = path.read_text()
    fm = parse_frontmatter(text)
    if fm is None:
        result["status"] = "fail"
        result["reason"] = "no YAML frontmatter"
        return result

    for key in REQUIRED:
        if key not in fm or not fm[key]:
            result["missing_required"].append(key)
    for key in RECOMMENDED:
        if key not in fm or not fm[key]:
            result["missing_recommended"].append(key)

    desc = fm.get("description", "")
    result["trigger_skip_present"] = "TRIGGER" in desc and "SKIP" in desc

    name_in_file = fm.get("name", "").strip().strip("\"'")
    result["name_matches_file"] = name_in_file == path.stem

    eval_path = EVALS_DIR / f"{path.stem}.eval.json"
    result["eval_set_present"] = eval_path.exists()
    if result["eval_set_present"]:
        try:
            eval_spec = json.loads(eval_path.read_text())
            result["eval_set_size"] = len(eval_spec.get("evals", []))
        except json.JSONDecodeError as e:
            result["eval_set_present"] = False
            result["eval_set_parse_error"] = str(e)

    if result["missing_required"] or not result["trigger_skip_present"] or not result["name_matches_file"]:
        result["status"] = "fail"
    elif result["missing_recommended"] or not result["eval_set_present"]:
        result["status"] = "warn"
    return result


def main() -> int:
    files = sorted(p for p in SKILLS_DIR.glob("*.md") if p.name != "README.md")
    if not files:
        print("No skills found under", SKILLS_DIR)
        return 1
    results = [validate(p) for p in files]
    summary = {
        "pass": sum(1 for r in results if r["status"] == "pass"),
        "warn": sum(1 for r in results if r["status"] == "warn"),
        "fail": sum(1 for r in results if r["status"] == "fail"),
        "total": len(results),
    }
    print(json.dumps({"summary": summary, "results": results}, indent=2))
    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
