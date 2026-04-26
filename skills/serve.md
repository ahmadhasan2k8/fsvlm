---
name: serve
description: |
  Launch one of fsvlm's inference surfaces: Gradio UI (annotation + inspection), FastAPI REST
  server, or watch-folder daemon. All three share the same adapter format and emit identical
  JSON. Records its PID + health for later teardown.
  TRIGGER when: user wants to deploy a trained adapter for interactive use or as a service.
  SKIP when: user is in a research-loop phase (use /sweep, /validate, /inspect instead).
inputs:
  - mode (string, required) — ui | rest | watch
  - adapter (path, optional) — adapter directory; default: ~/.fsvlm/adapters/latest/
  - port (int, default: 7860 for ui, 8080 for rest)
  - watch_dir (path, required if mode=watch) — folder to watch
  - background (bool, default: true) — run in background; if false, blocks
eval_artifact: ~/.fsvlm/serve_health.json
pass_criteria:
  - file exists at eval_artifact
  - JSON keys: mode, pid, port, adapter, started_at, status
  - status == "running"
  - port is responsive (HTTP 200 on /health for rest mode; HTTP 200 on / for ui)
escalation: |
  If pass_criteria fails:
    - port already in use: increment port and retry once
    - adapter load fails at startup: surface stderr, route to /debug with focus=adapter
    - watch_dir does not exist: report and exit (do not auto-create)
---

# Skill: serve

## Purpose

Stand up a long-running inference surface. Think of this as the "deployment" skill, distinct
from the "research-loop" skills.

## Procedure

1. Resolve the adapter, default to the latest symlink. Verify it loads (a one-shot inspect on
   a fixture image — fail fast if it doesn't).

2. Launch the chosen mode:
   ```bash
   case "$mode" in
     ui)    fsvlm ui    --adapter "$adapter" --port "$port" ${background:+&} ;;
     rest)  fsvlm serve --adapter "$adapter" --port "$port" ${background:+&} ;;
     watch) fsvlm watch "$watch_dir" --adapter "$adapter" ${background:+&} ;;
   esac
   PID=$!
   ```

3. Health-check after 3 seconds:
   ```bash
   case "$mode" in
     ui|rest) curl -fsS "http://localhost:$port/health" || curl -fsS "http://localhost:$port/" ;;
     watch)   ps -p "$PID" >/dev/null ;;
   esac
   ```

4. Write health JSON:
   ```json
   {
     "mode": "ui",
     "pid": 12345,
     "port": 7860,
     "adapter": "/home/user/.fsvlm/adapters/myadapter/",
     "started_at": "2026-04-25T21:00:00Z",
     "status": "running"
   }
   ```

## Self-evaluation

PASS if the process is alive AND (for ui|rest) the health endpoint responds AND the JSON
status is "running". Otherwise FAIL with the failed sub-check named.

## Failure modes

- **Port in use**: another process holds it. Bump and retry; if persistent, surface `lsof -i
  :$port` output to the user.
- **Adapter load fails at startup**: the adapter wasn't readable. Run /debug.
- **Process dies after 3 s**: usually a missing dep (e.g., gradio not installed). Suggest
  `pip install -e ".[ui]"` for ui mode, `".[serve]"` for rest, `".[watch]"` for watch.

## Teardown

```bash
kill "$PID"  # or the value from serve_health.json
```

The health file is left on disk as the audit record; remove it manually after teardown.

## Adapting to your runtime

- **Claude Code**: drop into `~/.claude/skills/`
- **OpenAI Agents SDK**: register `serve(mode, adapter, port, watch_dir, background)`
- **Plain shell**: invoke the Procedure block directly
