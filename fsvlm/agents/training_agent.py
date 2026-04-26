"""Training Agent — QLoRA fine-tuning via Unsloth.

Cython-compatible: no module-level torch/transformers/unsloth imports,
no exec/eval, no complex metaclasses. Heavy imports inside function bodies.
Dependencies injected via constructor.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fsvlm.config import FSVLMConfig
from fsvlm.types import (
    LabeledSample,
    LoRAConfig,
    PreparedDataset,
    TrainingConfig,
    TrainingResult,
)


class TrainingAgent:
    """Runs QLoRA fine-tuning on a prepared dataset.

    Args:
        config: FSVLM configuration.
        event_bus: Optional EventBus for progress notifications.
    """

    def __init__(
        self,
        config: FSVLMConfig,
        event_bus: object | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus  # EventBus or None — typed as object for Cython compat

    def train(
        self,
        dataset: PreparedDataset,
        output_dir: Path | None = None,
        training_config: TrainingConfig | None = None,
    ) -> TrainingResult:
        """Fine-tune a VLM adapter on the prepared dataset.

        Args:
            dataset: PreparedDataset from the Data Agent.
            output_dir: Where to save the adapter. Defaults to config.adapters_dir.
            training_config: Training hyperparameters. Defaults from config.

        Returns:
            TrainingResult with adapter path and training stats.
        """
        import gc

        import torch
        from loguru import logger

        tc = training_config or self._build_default_config()
        out = output_dir or self._config.adapters_dir / "latest"
        out.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training with model={tc.model_name}, rank={tc.lora.rank}, epochs={tc.num_train_epochs}")

        # Build conversation-format samples
        train_convos = self._build_conversations(dataset.train_samples, tc)
        val_convos = self._build_conversations(dataset.val_samples, tc)

        # Load model and train
        start_time = time.time()
        adapter_path, loss_history = self._run_training(train_convos, val_convos, tc, out)
        elapsed = time.time() - start_time

        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Training complete in {elapsed:.0f}s, adapter at {adapter_path}")

        return TrainingResult(
            adapter_path=adapter_path,
            config=tc,
            train_loss_history=loss_history,
            elapsed_seconds=elapsed,
        )

    def _build_default_config(self) -> TrainingConfig:
        """Build a TrainingConfig from FSVLMConfig defaults."""
        c = self._config
        return TrainingConfig(
            model_name=c.default_model,
            load_in_4bit=c.load_in_4bit,
            max_seq_length=c.max_seq_length,
            lora=LoRAConfig(rank=c.default_lora_rank, alpha=c.default_lora_alpha),
            num_train_epochs=c.default_max_epochs,
            per_device_train_batch_size=c.default_batch_size,
            gradient_accumulation_steps=c.default_gradient_accumulation,
            learning_rate=c.default_learning_rate,
            seed=c.default_seed,
            max_image_size=c.max_image_size,
        )

    def _build_conversations(
        self,
        samples: list[LabeledSample],
        tc: TrainingConfig,
    ) -> list[dict]:
        """Convert LabeledSamples into VLM conversation-format dicts."""

        from fsvlm.utils.image import load_image

        conversations = []
        for sample in samples:
            img = load_image(sample.image_path, max_size=tc.max_image_size)

            pass_fail = "PASS" if sample.label == "good" else "FAIL"
            if sample.description:
                response_text = f"{pass_fail}\n{sample.description}"
            else:
                if sample.label == "good":
                    response_text = (
                        f"{pass_fail}\nNo defects detected. "
                        "The item appears intact with normal surface texture."
                    )
                else:
                    response_text = (
                        f"{pass_fail}\nDefect detected. The item shows visible damage on its surface."
                    )

            conversations.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": tc.inspection_prompt},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": response_text},
                            ],
                        },
                    ],
                }
            )
        return conversations

    def _run_training(
        self,
        train_convos: list[dict],
        val_convos: list[dict],
        tc: TrainingConfig,
        output_dir: Path,
    ) -> tuple[Path, list[float]]:
        """Execute the actual training loop. All heavy imports here."""
        import os

        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

        import torch
        from datasets import Dataset
        from loguru import logger
        from transformers import Trainer
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastVisionModel
        from unsloth.trainer import UnslothVisionDataCollator

        # Load model
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=tc.model_name,
            max_seq_length=tc.max_seq_length,
            load_in_4bit=tc.load_in_4bit,
            load_in_16bit=not tc.load_in_4bit,
            use_gradient_checkpointing="unsloth",
            device_map="cuda:0",
        )

        # Apply LoRA
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=tc.lora.finetune_vision_layers,
            finetune_language_layers=tc.lora.finetune_language_layers,
            finetune_attention_modules=tc.lora.finetune_attention_modules,
            finetune_mlp_modules=tc.lora.finetune_mlp_modules,
            r=tc.lora.rank,
            lora_alpha=tc.lora.alpha,
            lora_dropout=tc.lora.dropout,
            bias="none",
            random_state=tc.seed,
            use_rslora=False,
            target_modules="all-linear",
        )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        train_dataset = Dataset.from_list(train_convos)
        val_dataset = Dataset.from_list(val_convos)

        # VRAM headroom check — skip eval during training if tight
        vram_free = torch.cuda.mem_get_info()[0] / (1024**3)
        if vram_free < 6.0:
            logger.info(f"VRAM headroom: {vram_free:.1f}GB free — skipping eval during training")
            eval_strategy = "no"
            load_best = False
        else:
            eval_strategy = "epoch"
            load_best = True

        training_args = SFTConfig(
            per_device_train_batch_size=tc.per_device_train_batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            warmup_steps=max(1, int(tc.warmup_ratio * 150)),
            num_train_epochs=tc.num_train_epochs,
            learning_rate=tc.learning_rate,
            fp16=False,
            bf16=tc.bf16,
            optim=tc.optim,
            weight_decay=tc.weight_decay,
            lr_scheduler_type=tc.lr_scheduler_type,
            seed=tc.seed,
            output_dir=str(output_dir / "checkpoints"),
            report_to="none",
            logging_steps=1,
            save_strategy="epoch",
            eval_strategy=eval_strategy,
            load_best_model_at_end=load_best,
            metric_for_best_model="eval_loss" if load_best else None,
            greater_is_better=False if load_best else None,
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            dataset_num_proc=4,
        )

        # SafeSFTTrainer: skip logits-dependent metrics that crash
        # with Unsloth's compiled lazy logits on Gemma 4
        class SafeSFTTrainer(SFTTrainer):
            def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, **kwargs):
                return Trainer.compute_loss(
                    self,
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

        trainer = SafeSFTTrainer(
            model=model,
            processing_class=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
        )

        # Add progress event callback if event bus is available
        if self._event_bus is not None:
            from transformers import TrainerCallback

            bus = self._event_bus
            train_start = time.time()

            class ProgressCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs and "loss" in logs:
                        from fsvlm.events import TrainingProgressEvent

                        bus.emit(
                            TrainingProgressEvent(
                                epoch=int(state.epoch or 0),
                                total_epochs=int(args.num_train_epochs),
                                step=state.global_step,
                                total_steps=state.max_steps,
                                loss=logs.get("loss", 0.0),
                                learning_rate=logs.get("learning_rate", 0.0),
                                elapsed_seconds=time.time() - train_start,
                            )
                        )

            trainer.add_callback(ProgressCallback())

        trainer.train()

        # Extract loss history
        loss_history = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]

        # Save adapter (not merged — avoids Gemma4ClippableLinear bug)
        adapter_path = output_dir / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        # Save training log
        log_path = output_dir / "training_log.json"
        log_path.write_text(json.dumps(trainer.state.log_history, indent=2))

        # Free model
        del trainer, model, tokenizer
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        return adapter_path, loss_history
