---
name: inspect
description: |
  Run a fine-tuned adapter (or zero-shot base model) on one image, a folder, or a streamed list.
  Produces JSON with PASS/FAIL, confidence, defects, and natural-language description per image.
  TRIGGER when: user has an adapter and wants predictions on new data.
  SKIP when: user has not trained an adapter and wants the zero-shot baseline only — pass
  --no-adapter explicitly; do not infer.
inputs:
  - target (path, required) — image file or folder of images
  - adapter (path, optional) — adapter directory; default: ~/.fsvlm/adapters/latest/
  - backend (string, default: gemma4-e4b)
  - output_format (string, default: json) — json | csv | summary
  - output_path (path, optional) — where to write results; default: stdout
eval_artifact: ${output_path}  (or stdout if unset)
pass_criteria:
  - one record per input image
  - every record has: image_path, pass_fail, confidence, defects (list), description
  - confidence is in [0.0, 1.0]
  - inference_time_ms is recorded
escalation: |
  If pass_criteria fails:
    - missing records: input file unreadable; check image MIME types and surface skipped paths
    - confidence outside [0,1]: extractor bug; run /debug with focus=adapter
    - all records identical: degenerate output; check the adapter loaded correctly (look at
      metadata schema_version)
---

# Skill: inspect

## Purpose

Use a trained adapter to make predictions. Single-image, batch, and streaming all produce the
same JSON record format so downstream tooling is uniform.

## Procedure

1. Resolve the adapter path. If `adapter` is unset, use `~/.fsvlm/adapters/latest/` (a symlink
   maintained by /train). Verify the adapter's `fsvlm_metadata.json` exists and `schema_version`
   is current.

2. Run inference:
   ```bash
   fsvlm inspect "$target" \
                 --adapter "$adapter" \
                 --backend "$backend" \
                 --output "$output_format" \
                 ${output_path:+--output-path "$output_path"}
   ```

3. Parse the output. JSON record schema (one per image):
   ```json
   {
     "image_path": "/path/to/img.jpg",
     "pass_fail": false,
     "confidence": 0.91,
     "defects": [
       {"type": "crack", "location": "lower-right", "severity": "major", "confidence": 0.88}
     ],
     "description": "A crack defect is visible in the lower-right region of the part.",
     "model_name": "gemma-4-e4b-it",
     "adapter_name": "myadapter-v3",
     "adapter_version": 1,
     "inference_time_ms": 178.4
   }
   ```

## Self-evaluation

For each input image, exactly one record. Validate confidence range, presence of required keys,
and that `inference_time_ms > 0`. PASS if all records pass.

## Failure modes

- **Adapter loaded but predictions are constant**: `schema_version` mismatch — the adapter was
  trained against a newer fsvlm version. Surface a migration hint.
- **Slow inference (>5 s per image)**: unsloth not loaded properly, or running on CPU. Re-check
  /setup output.
- **`UnicodeDecodeError` on image read**: corrupt or non-image file in folder; skip and continue,
  list skipped paths in the output summary.

## Adapting to your runtime

- **Claude Code**: drop into `~/.claude/skills/`
- **OpenAI Agents SDK**: register `inspect(target, adapter, backend, output_format, output_path)`
- **Plain shell**: invoke the Procedure block directly
