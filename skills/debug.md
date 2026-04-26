---
name: debug
description: |
  Diagnose problems with the local install, the GPU, an adapter, or a dataset. Produces a
  structured report with PASS/FAIL per check so the next skill (or a human) can act on it.
  TRIGGER when: any other skill's pass_criteria fails, or user reports unexpected behavior.
  SKIP when: the problem is a simple typo / missing argument — fix that first.
inputs:
  - focus (string, default: all) — gpu | deps | adapter | dataset | config | all
  - adapter (path, optional) — required when focus=adapter
  - dataset (path, optional) — required when focus=dataset
eval_artifact: ~/.fsvlm/debug.json
pass_criteria:
  - file exists at eval_artifact
  - JSON keys: focus, checks (list of check results), summary, recommendations
  - every check has: name, status (pass|warn|fail), detail
  - summary indicates the overall status of the focus area
escalation: |
  /debug is the terminal skill — it does not escalate further; it explains the failure clearly
  enough that the user (or the calling skill) can act. If the problem is genuinely outside
  fsvlm's control (CUDA driver, hardware), the recommendations field says so.
---

# Skill: debug

## Purpose

When something doesn't work, this skill produces a machine-readable report that names the
specific failed check and the suggested fix.

## Procedure

Run the relevant diagnostic block(s) based on `focus`:

### `focus=gpu`
```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### `focus=deps`
```bash
python -c "import fsvlm; print(fsvlm.__version__)"
python -c "import unsloth; import bitsandbytes; import transformers; import peft" 2>&1
pip show fsvlm 2>&1 | head -10
```

### `focus=adapter`
```bash
ls -la "$adapter/"
python -c "
import json
m = json.load(open('$adapter/fsvlm_metadata.json'))
print(json.dumps(m, indent=2))
"
fsvlm inspect "$(find tests/fixtures -name '*.jpg' | head -1)" --adapter "$adapter"
```

### `focus=dataset`
```bash
python -c "
from fsvlm.agents.data_agent import DataAgent
from fsvlm.config import FSVLMConfig
import json
report = DataAgent(FSVLMConfig()).prepare('$dataset').report
print(json.dumps({
  'total_images': report.total_images,
  'good_count': report.good_count,
  'defect_count': report.defect_count,
  'errors': report.errors[:10],
}, indent=2, default=str))
"
# Reports: image count per class, suspicious aspect ratios, corrupt files
```

### `focus=config`
```bash
cat ~/.fsvlm/config.toml 2>/dev/null || echo "no user config — using defaults"
env | grep '^FSVLM_'
```

Aggregate every check into a single JSON:
```json
{
  "focus": "all",
  "checks": [
    {"name": "gpu_available", "status": "pass", "detail": "RTX 5080 Laptop, 16 GB"},
    {"name": "torch_cuda", "status": "pass", "detail": "torch.cuda.is_available() == True"},
    {"name": "unsloth_imported", "status": "fail",
     "detail": "ImportError: bitsandbytes wheel for CUDA 12.6 missing"}
  ],
  "summary": "GPU is fine; unsloth dependency is broken. fsvlm cannot train until fixed.",
  "recommendations": [
    "pip install --upgrade bitsandbytes",
    "If that fails, pip install bitsandbytes-cuda126"
  ]
}
```

## Self-evaluation

PASS if `eval_artifact` exists and contains the schema above. Note: a /debug PASS does NOT
imply the underlying problem is fixed — it just means the diagnostic ran cleanly. The
`checks` array reports the actual diagnosis.

## Failure modes

- **`nvidia-smi` not found**: no NVIDIA driver — fsvlm cannot train.
- **All checks pass but the calling skill still fails**: bug. File an issue with the debug.json
  and the calling skill's eval_artifact attached.
- **Adapter focus fails because adapter path doesn't exist**: surface a clear "no such adapter"
  error rather than a stack trace.

## Adapting to your runtime

- **Claude Code**: drop into `~/.claude/skills/`
- **OpenAI Agents SDK**: register `debug(focus, adapter, dataset)`
- **Plain shell**: invoke the relevant Procedure block(s) and aggregate manually
