---
name: Bug report
about: Report a reproducible bug
title: "[Bug] "
labels: bug
---

## What happened

A clear description of the bug.

## What you expected

What you thought would happen instead.

## Minimal reproducer

The smallest command or code snippet that triggers the bug. Ideally include the full command line and any configuration.

```bash
# e.g.
fsvlm train --images ./my-data/ --epochs 3
```

## Environment

- **fsvlm version**: `fsvlm --version`
- **Python**: `python --version`
- **OS**: (e.g., Ubuntu 22.04, macOS 14, WSL2)
- **GPU**: `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv` (paste output)
- **Installed extras**: `pip show fsvlm | grep Requires` (or list which extras you installed: `[train]`, `[serve]`, `[ui]`, etc.)

## Logs

Any relevant output from the terminal or from `~/.fsvlm/logs/`. Please wrap in a code block or attach as a file if long.

```
paste logs here
```

## Additional context

Anything else that might help us reproduce — sample images (if license allows), dataset size, config file, etc.
