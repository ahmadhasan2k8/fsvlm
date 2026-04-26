"""Subprocess entry point for training.

Runs training in a clean process so the CUDA context is fully released
when it exits, freeing all GPU memory for the validation subprocess.

Usage (internal):
    python -m fsvlm.agents._train_subprocess \\
        --samples /path/to/samples.json \\
        --config /path/to/config.json \\
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    import os

    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    from fsvlm.agents.training_agent import TrainingAgent
    from fsvlm.config import FSVLMConfig
    from fsvlm.types import (
        DatasetReport,
        LabeledSample,
        LoRAConfig,
        PreparedDataset,
        TrainingConfig,
    )

    # Load samples
    with open(args.samples) as f:
        data = json.load(f)

    train_samples = [
        LabeledSample(Path(s["image_path"]), s["label"], s.get("description", "")) for s in data["train"]
    ]
    val_samples = [
        LabeledSample(Path(s["image_path"]), s["label"], s.get("description", "")) for s in data["val"]
    ]

    report = DatasetReport(
        total_images=data["report"]["total_images"],
        good_count=data["report"]["good_count"],
        defect_count=data["report"]["defect_count"],
    )
    dataset = PreparedDataset(
        train_samples=train_samples,
        val_samples=val_samples,
        report=report,
        seed=data.get("seed", 3407),
    )

    # Load training config
    with open(args.config) as f:
        config_data = json.load(f)

    lora_data = config_data.pop("lora", {})
    config_data.pop("output_dir", None)
    lora = LoRAConfig(**lora_data) if lora_data else LoRAConfig()
    tc = TrainingConfig(lora=lora, output_dir=Path(args.output_dir), **config_data)

    # Train
    dvlm_config = FSVLMConfig()
    agent = TrainingAgent(dvlm_config)
    result = agent.train(dataset, output_dir=Path(args.output_dir), training_config=tc)

    # Serialize result
    result_data = {
        "adapter_path": str(result.adapter_path),
        "train_loss_history": result.train_loss_history,
        "elapsed_seconds": result.elapsed_seconds,
    }
    Path(args.result_file).write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    main()
