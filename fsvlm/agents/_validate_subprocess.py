"""Subprocess entry point for validation.

Runs validation in a clean process to avoid CUDA context memory issues
after training. The training process's CUDA context retains ~10GB even
after del model + gc.collect() + empty_cache().

Usage (internal):
    python -m fsvlm.agents._validate_subprocess \\
        --adapter /path/to/adapter \\
        --samples /path/to/samples.json \\
        --config /path/to/config.json \\
        --output /path/to/report.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Import heavy deps only in this subprocess
    from fsvlm.agents.validation_agent import ValidationAgent
    from fsvlm.config import FSVLMConfig
    from fsvlm.types import LabeledSample, LoRAConfig, TrainingConfig

    # Load inputs
    with open(args.samples) as f:
        samples_data = json.load(f)
    val_samples = [
        LabeledSample(
            image_path=Path(s["image_path"]),
            label=s["label"],
            description=s.get("description", ""),
        )
        for s in samples_data
    ]

    with open(args.config) as f:
        config_data = json.load(f)

    # Reconstruct TrainingConfig
    lora_data = config_data.pop("lora", {})
    config_data.pop("output_dir", None)
    lora = LoRAConfig(**lora_data) if lora_data else LoRAConfig()
    tc = TrainingConfig(lora=lora, **config_data)

    dvlm_config = FSVLMConfig()
    validator = ValidationAgent(dvlm_config)
    report = validator.validate(Path(args.adapter), val_samples, tc)

    # Serialize report
    report_dict = asdict(report)
    Path(args.output).write_text(json.dumps(report_dict, indent=2, default=str))


if __name__ == "__main__":
    main()
