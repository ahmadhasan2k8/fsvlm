---
name: setup
description: |
  Prepare the local environment to run fsvlm: detect GPU, recommend a base VLM that fits the
  available VRAM, download the model weights, and verify the install. Idempotent — safe to re-run.
  TRIGGER when: user is on a fresh machine, swapping GPUs, switching backends, or hits a
  "model not found" error.
  SKIP when: user just wants to inspect a single image with an already-cached model. Use
  /inspect directly instead.
inputs:
  - backend (string, default: gemma4-e4b) — registered ModelBackend name
  - model_size (string, default: auto) — small | medium | large | auto
  - check_only (bool, default: false) — verify install without downloading
eval_artifact: ~/.fsvlm/setup_status.json
pass_criteria:
  - file exists at eval_artifact
  - JSON keys: gpu_detected, vram_total_gb, backend, model_path, deps_ok
  - gpu_detected is true OR cpu_inference_only_acknowledged is true
  - deps_ok is true
escalation: |
  If pass_criteria fails:
    - GPU not detected: run `/debug` with focus=gpu, surface the result to user
    - Insufficient VRAM: re-run with model_size=small, or recommend a smaller backend
    - Deps broken: report which Python deps failed to import; suggest pip install -e ".[train]"
---

# Skill: setup

## Purpose

One-stop environment readiness check. Tells the user whether their machine can run fsvlm at all,
and if so, picks the right base model for the available VRAM. Idempotent.

## Procedure

1. Detect GPU and VRAM:
   ```bash
   fsvlm setup --check
   ```
   Captures GPU name, VRAM total/free, CUDA version, compute capability.

2. If `--check_only` is true, stop here and write the status JSON.

3. Otherwise, download the recommended model for the chosen backend:
   ```bash
   fsvlm setup --backend "$backend" ${model_size:+--model "$model_size"}
   ```
   Default `model_size=auto` lets the backend pick: small for ≤8 GB VRAM, medium for ≤16 GB,
   large for >16 GB.

4. Verify the install end-to-end with a no-op inference call:
   ```bash
   fsvlm inspect --help >/dev/null
   ```

5. Write the status JSON to `~/.fsvlm/setup_status.json`:
   ```json
   {
     "gpu_detected": true,
     "gpu_name": "RTX 5080 Laptop",
     "vram_total_gb": 16.0,
     "vram_free_gb": 14.2,
     "cuda_version": "12.6",
     "backend": "gemma4-e4b",
     "model_path": "~/.fsvlm/models/gemma4-e4b/",
     "deps_ok": true
   }
   ```

## Self-evaluation

PASS if all `pass_criteria` are met. FAIL otherwise; include the specific missing key or
falsy value in the output.

## Failure modes

- **No NVIDIA GPU**: fsvlm cannot train without one. Inference can run on CPU but is slow.
  Report and exit; do not silently fall back.
- **VRAM too small for chosen backend**: surface the requirement vs. available, recommend
  a smaller variant.
- **Network failure during model download**: HuggingFace Hub is sometimes flaky; retry once.
  If it fails twice, surface the URL and ask the user to retry manually.
- **`bitsandbytes` import fails**: usually means the wheel for the user's CUDA version isn't
  installed. Suggest `pip install bitsandbytes --upgrade`.

## Adapting to your runtime

- **Claude Code**: `cp skills/*.md ~/.claude/skills/` and invoke as `/setup`
- **OpenAI Agents SDK**: register `setup(backend, model_size, check_only)` as a tool
- **Plain shell**: `bash` the Procedure block directly
