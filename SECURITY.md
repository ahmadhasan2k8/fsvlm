# Security Policy

## Reporting a vulnerability

If you believe you've found a security vulnerability in fsvlm — for example, a
dependency with a known CVE that we haven't updated, a path-traversal issue in a
file-reading utility, arbitrary code execution via a crafted adapter file, or
anything else that could meaningfully harm users — **please report it privately**
rather than opening a public issue.

Open a draft security advisory in this repository on GitHub:

1. Go to the **Security** tab of the repository.
2. Click **"Report a vulnerability"** (or use the **"New draft security advisory"** button).
3. Describe the issue, reproduction steps, and suggested fix if you have one.

GitHub's draft advisories are private to maintainers until published.

## What to expect

- Acknowledgement of receipt within 7 days.
- Initial triage within 14 days.
- A fix within 30 days for high-severity issues; longer for lower severity.
- Credit in the release notes when the fix ships, unless you prefer to remain anonymous.

## Scope

In scope:

- Bugs in fsvlm's Python code (`fsvlm/` package, CLI, REST API, Gradio UI)
- Dependency issues in the pinned packages listed in `pyproject.toml`
- Unsafe defaults in the training/inference pipeline

Out of scope:

- Bugs in upstream dependencies (unsloth, transformers, PyTorch, Gradio) — report to those
  projects directly; we will pin to their fixed versions once available.
- Issues that require an attacker to already have local filesystem or GPU access to the user's
  machine (fsvlm is a local-first tool; we assume the local environment is trusted).
- Model-output issues (e.g., hallucinated defect descriptions) — these are research
  limitations, not security vulnerabilities. File them as regular issues.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✓         |
| < 0.1   | ✗         |

Only the latest minor version receives security fixes. When a new minor lands, the previous one
enters unsupported status after a 30-day grace period.

## License

fsvlm is licensed under Apache 2.0 (see `LICENSE`). Commercial use is permitted; we
appreciate a citation when the work is used in published research.
