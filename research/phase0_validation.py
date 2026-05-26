"""Phase 0: ML Validation — Prove Gemma 4 can do defect detection.

Single script with three functions mapping to future agents:
  - prepare_data()  → Data Agent
  - train_adapter() → Training Agent
  - evaluate()      → Validation Agent

Gate: AUROC >= 90% on MVTec AD hazelnut, or reassess model choice.
Model: Gemma 4 26B-A4B (MoE, Apache 2.0) via Unsloth QLoRA.
Fallback: Gemma 4 E4B (dense 4B) if 26B OOMs on 16GB VRAM.

Usage:
    python phase0_validation.py --data ./mvtec_hazelnut/
    python phase0_validation.py --data ./mvtec_hazelnut/ --model-name unsloth/gemma-4-E4B-it
    python phase0_validation.py --data ./mvtec_hazelnut/ --download-data
"""

from __future__ import annotations

# Must be set before any Unsloth imports to ensure logits are returned for TRL
import os

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All hyperparameters and settings for reproducibility."""

    # Model
    model_name: str = "unsloth/gemma-4-26B-A4B-it"
    load_in_4bit: bool = True
    max_seq_length: int = 1024

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    bf16: bool = True

    # Data
    test_split: float = 0.2
    seed: int = 3407
    max_image_size: int = 560  # longest edge in pixels

    # Prompt
    inspection_prompt: str = (
        "You are a visual quality inspector. Examine this image of a hazelnut. "
        "Respond with exactly $pass_token or $fail_token on the first line. "
        "On the second line, describe what you see."
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalMetrics:
    """Evaluation results."""

    auroc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    confusion_matrix: list[list[int]]
    num_test_samples: int
    predictions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def download_mvtec_hazelnut(output_dir: Path) -> Path:
    """Download MVTec AD hazelnut category.

    MVTec AD is available at https://www.mvtec.com/company/research/datasets/mvtec-ad
    This function downloads the hazelnut category and reorganizes into good/defect folders.

    Args:
        output_dir: Directory to save the dataset.

    Returns:
        Path to the prepared dataset directory with good/ and defect/ subdirectories.
    """
    dataset_dir = output_dir / "mvtec_hazelnut"

    if dataset_dir.exists() and (dataset_dir / "good").exists() and (dataset_dir / "defect").exists():
        good_count = len(list((dataset_dir / "good").glob("*.png")))
        defect_count = len(list((dataset_dir / "defect").glob("*.png")))
        if good_count > 0 and defect_count > 0:
            print(f"Dataset already exists: {good_count} good, {defect_count} defect images")
            return dataset_dir

    print("Downloading MVTec AD hazelnut dataset...")

    # MVTec AD hazelnut download
    # The dataset is hosted at mvtec.com — we use their direct download link
    import urllib.request

    mvtec_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/hazelnut.tar.xz"
    tar_path = output_dir / "hazelnut.tar.xz"
    extract_dir = output_dir / "hazelnut_raw"

    if not tar_path.exists():
        print(f"Downloading from {mvtec_url}...")
        print("(This is ~260MB, may take a few minutes)")

        def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  {mb_down:.1f}/{mb_total:.1f} MB ({pct}%)", end="", flush=True)

        urllib.request.urlretrieve(mvtec_url, str(tar_path), reporthook=_progress_hook)
        print()  # newline after progress

    # Extract
    if not extract_dir.exists():
        print("Extracting archive...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(tar_path), "r:xz") as tar:
            tar.extractall(str(extract_dir))

    # Reorganize into good/defect structure
    # MVTec structure: hazelnut/train/good/, hazelnut/test/good/, hazelnut/test/{crack,cut,hole,print}/
    raw_hazelnut = extract_dir / "hazelnut"
    if not raw_hazelnut.exists():
        # Try without subdirectory
        raw_hazelnut = extract_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    good_dir = dataset_dir / "good"
    defect_dir = dataset_dir / "defect"
    good_dir.mkdir(exist_ok=True)
    defect_dir.mkdir(exist_ok=True)

    # Copy good images (from both train and test)
    good_idx = 0
    for split in ["train", "test"]:
        src = raw_hazelnut / split / "good"
        if src.exists():
            for img in sorted(src.glob("*.png")):
                shutil.copy2(str(img), str(good_dir / f"good_{good_idx:04d}.png"))
                good_idx += 1

    # Copy defect images (all test defect categories)
    defect_idx = 0
    test_dir = raw_hazelnut / "test"
    if test_dir.exists():
        for category_dir in sorted(test_dir.iterdir()):
            if category_dir.is_dir() and category_dir.name != "good":
                for img in sorted(category_dir.glob("*.png")):
                    defect_name = category_dir.name
                    shutil.copy2(
                        str(img),
                        str(defect_dir / f"defect_{defect_name}_{defect_idx:04d}.png"),
                    )
                    defect_idx += 1

    print(f"Dataset prepared: {good_idx} good, {defect_idx} defect images in {dataset_dir}")

    # Cleanup
    if tar_path.exists():
        tar_path.unlink()
    if extract_dir.exists():
        shutil.rmtree(str(extract_dir))

    return dataset_dir


def prepare_data(
    image_dir: Path,
    config: ExperimentConfig,
) -> tuple[list[dict], list[dict], list[int], list[int]]:
    """Read good/defect folders, build conversation-format training data.

    Maps to future Data Agent. Reads images from a directory with good/ and defect/
    subdirectories, resizes them, and builds conversation-format samples for VLM training.

    Args:
        image_dir: Directory containing good/ and defect/ subdirectories.
        config: Experiment configuration.

    Returns:
        Tuple of (train_samples, val_samples, train_indices, val_indices).
        Each sample is a dict with a "messages" key containing the conversation.
        Indices track which original samples went to train vs val for reproducibility.
    """
    from PIL import Image

    good_dir = image_dir / "good"
    defect_dir = image_dir / "defect"

    if not good_dir.exists():
        raise FileNotFoundError(f"No 'good' directory found at {good_dir}")
    if not defect_dir.exists():
        raise FileNotFoundError(f"No 'defect' directory found at {defect_dir}")

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    # Collect all samples
    samples: list[dict] = []

    for img_path in sorted(good_dir.iterdir()):
        if img_path.suffix.lower() in image_extensions:
            samples.append({
                "image_path": str(img_path),
                "label": "good",
                "response": "PASS. No defects detected. The hazelnut appears intact with normal surface texture.",
            })

    for img_path in sorted(defect_dir.iterdir()):
        if img_path.suffix.lower() in image_extensions:
            # Extract defect type from filename if available (e.g., defect_crack_0001.png)
            defect_type = "surface defect"
            name_parts = img_path.stem.split("_")
            if len(name_parts) >= 2:
                defect_type = name_parts[1]

            samples.append({
                "image_path": str(img_path),
                "label": "defect",
                "response": (
                    f"FAIL. Defect detected: {defect_type}. "
                    f"The hazelnut shows visible {defect_type} damage on its surface."
                ),
            })

    if len(samples) == 0:
        raise ValueError(f"No images found in {image_dir}/good/ or {image_dir}/defect/")

    print(f"Found {len(samples)} images: "
          f"{sum(1 for s in samples if s['label'] == 'good')} good, "
          f"{sum(1 for s in samples if s['label'] == 'defect')} defect")

    # Deterministic shuffle and split
    rng = np.random.RandomState(config.seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    split_idx = int(len(samples) * (1 - config.test_split))
    train_indices = sorted(indices[:split_idx].tolist())
    val_indices = sorted(indices[split_idx:].tolist())

    def _build_conversation(sample: dict) -> dict:
        """Build a conversation-format sample for VLM training."""
        img = Image.open(sample["image_path"]).convert("RGB")

        # Resize if needed (keep aspect ratio, max edge = config.max_image_size)
        w, h = img.size
        if max(w, h) > config.max_image_size:
            scale = config.max_image_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        from fsvlm.prompts.verdict import resolve_inspection_prompt

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": resolve_inspection_prompt(config.inspection_prompt)},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": sample["response"]},
                    ],
                },
            ],
        }

    print("Building conversation-format samples...")

    # Oversample minority class (defect) in training set to balance classes
    train_good_idx = [i for i in train_indices if samples[i]["label"] == "good"]
    train_defect_idx = [i for i in train_indices if samples[i]["label"] == "defect"]
    n_good = len(train_good_idx)
    n_defect = len(train_defect_idx)

    if n_defect > 0 and n_good > n_defect:
        oversample_factor = n_good // n_defect
        remainder = n_good % n_defect
        # Repeat defect indices to match good count
        balanced_defect_idx = train_defect_idx * oversample_factor
        # Add a few more to match exactly
        rng2 = np.random.RandomState(config.seed + 1)
        if remainder > 0:
            balanced_defect_idx += list(rng2.choice(train_defect_idx, remainder, replace=False))
        balanced_train_indices = train_good_idx + balanced_defect_idx
        rng2.shuffle(balanced_train_indices)
        print(f"Class balance: {n_good} good, {n_defect} defect → oversampled to {len(balanced_defect_idx)} defect")
    else:
        balanced_train_indices = train_indices

    train_samples = [_build_conversation(samples[i]) for i in balanced_train_indices]
    val_samples = [_build_conversation(samples[i]) for i in val_indices]

    print(f"Split: {len(train_samples)} train ({n_good} good + {len(balanced_train_indices) - n_good} defect), {len(val_samples)} val")
    return train_samples, val_samples, train_indices, val_indices


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_adapter(
    train_samples: list[dict],
    val_samples: list[dict],
    config: ExperimentConfig,
    output_dir: Path,
) -> Path:
    """QLoRA fine-tune Gemma 4 via Unsloth.

    Maps to future Training Agent. Loads the model with 4-bit quantization,
    applies LoRA, and trains on the conversation-format dataset.

    Args:
        train_samples: Training data in conversation format.
        val_samples: Validation data in conversation format.
        config: Experiment configuration.
        output_dir: Directory to save the trained adapter.

    Returns:
        Path to the saved adapter directory.
    """
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    print(f"\n{'='*60}")
    print(f"Loading model: {config.model_name}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3
    print(f"VRAM total: {total_vram:.1f} GB, free: {free_vram:.1f} GB")
    print(f"{'='*60}\n")

    # Check available VRAM
    gpu_free, gpu_total = torch.cuda.mem_get_info()
    gpu_free_gb = gpu_free / 1024**3
    print(f"GPU free memory: {gpu_free_gb:.1f} GB")

    # Force device_map to single GPU to prevent CPU offloading.
    # bitsandbytes 4-bit doesn't support mixed CPU/GPU device_map,
    # and CPU-offloaded bf16 models can't be trained with accelerate.
    # For small models (E4B ~3GB at 4-bit), this always fits.
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_16bit=not config.load_in_4bit,
        use_gradient_checkpointing="unsloth",
        device_map="cuda:0",
    )

    # Apply LoRA
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=config.finetune_vision_layers,
        finetune_language_layers=config.finetune_language_layers,
        finetune_attention_modules=config.finetune_attention_modules,
        finetune_mlp_modules=config.finetune_mlp_modules,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=config.seed,
        use_rslora=False,
        target_modules="all-linear",
    )

    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # Build HuggingFace datasets
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)

    # Setup trainer
    adapter_output = output_dir / "adapter"
    adapter_output.mkdir(parents=True, exist_ok=True)

    # Skip eval during training if VRAM is tight. The eval step's fp32 logit
    # conversion in accelerate needs ~2.5GB extra. We need that plus optimizer
    # states headroom, so require 6GB+ free after model+LoRA loading.
    vram_free = torch.cuda.mem_get_info()[0] / (1024**3)
    if vram_free < 6.0:
        print(f"  VRAM headroom: {vram_free:.1f} GB free — skipping eval during training to avoid OOM")
        eval_strategy = "no"
        load_best = False
    else:
        eval_strategy = "epoch"
        load_best = True

    training_args = SFTConfig(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=max(1, int(config.warmup_ratio * 150)),  # ~5% warmup
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        fp16=False,
        bf16=config.bf16,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        output_dir=str(adapter_output),
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

    # Subclass SFTTrainer to skip logits-dependent metrics (entropy, token accuracy)
    # that crash with Unsloth's compiled lazy logits on Gemma 4.
    class SafeSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, num_items_in_batch=None,
                         return_outputs=False, **kwargs):
            # Use base Trainer.compute_loss which only returns the model's loss
            # without accessing outputs.logits for entropy/accuracy computation.
            from transformers import Trainer
            return Trainer.compute_loss(
                self, model, inputs,
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

    # Train
    print(f"\nStarting training: {config.num_train_epochs} epochs, "
          f"batch_size={config.per_device_train_batch_size}, "
          f"grad_accum={config.gradient_accumulation_steps}")

    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Final train loss: {train_result.training_loss:.4f}")

    # Log VRAM usage
    if torch.cuda.is_available():
        vram_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak VRAM usage: {vram_used:.1f} GB")

    # Save adapter (not merged — avoids Gemma4ClippableLinear bug)
    adapter_save_path = output_dir / "final_adapter"
    model.save_pretrained(str(adapter_save_path))
    tokenizer.save_pretrained(str(adapter_save_path))
    print(f"Adapter saved to {adapter_save_path}")

    # Save training log
    log_history = trainer.state.log_history
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)

    # Free GPU memory — model will be reloaded for evaluation
    del trainer
    del model
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory freed for evaluation")

    return adapter_save_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    adapter_path: Path,
    val_samples: list[dict],
    config: ExperimentConfig,
) -> EvalMetrics:
    """Evaluate trained adapter: compute AUROC, F1, confusion matrix.

    Maps to future Validation Agent. Loads the adapter, runs inference on
    the validation set, parses PASS/FAIL from responses, and computes metrics.

    Args:
        adapter_path: Path to the saved adapter directory.
        val_samples: Validation data in conversation format.
        config: Experiment configuration.

    Returns:
        EvalMetrics with AUROC, F1, precision, recall, accuracy, and confusion matrix.
    """
    import torch

    # Disable caching_allocator_warmup — it tries to allocate the full FP16 model
    # size as a warmup buffer, which OOMs when loading large quantized models.
    # This is a performance optimization, not required for correctness.
    import transformers.modeling_utils
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    transformers.modeling_utils.caching_allocator_warmup = lambda *a, **kw: None

    from unsloth import FastVisionModel

    print(f"\n{'='*60}")
    print("Evaluating adapter...")
    print(f"{'='*60}\n")

    # Load model with adapter
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_16bit=not config.load_in_4bit,
        device_map="cuda:0",
    )

    FastVisionModel.for_inference(model)

    # Run inference on validation set
    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores: list[float] = []
    predictions: list[dict] = []

    from fsvlm.prompts.verdict import resolve_inspection_prompt, verdict_token_ids, verdict_tokens

    pass_str, fail_str = verdict_tokens()
    pass_token_id, fail_token_id = verdict_token_ids(tokenizer)

    for i, sample in enumerate(val_samples):
        messages = sample["messages"]

        # Extract ground truth from the assistant response
        assistant_text = messages[1]["content"][0]["text"]
        true_label = 0 if assistant_text.startswith(pass_str) else 1

        # Build inference input (user turn only)
        user_content = messages[0]["content"]
        # Extract image and text from the user message
        image = None
        text = ""
        for part in user_content:
            if part["type"] == "image":
                image = part["image"]
            elif part["type"] == "text":
                text = part["text"]

        # Format as chat with image marker for vision models
        device = next(model.parameters()).device
        if image is not None:
            chat_messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ]}]
        else:
            chat_messages = [{"role": "user", "content": text}]

        prompt = tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Process text + image together
        if image is not None:
            inputs = tokenizer(
                text=prompt,
                images=[image],
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = tokenizer(
                text=prompt,
                return_tensors="pt",
                padding=True,
            )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            # Generate with scores to get token probabilities
            gen_output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
            output = gen_output.sequences

            # Extract verdict probabilities from the first generated token's scores
            prob_pass = 0.5
            prob_fail = 0.5
            if hasattr(gen_output, 'scores') and gen_output.scores:
                first_token_scores = gen_output.scores[0][0]  # [vocab_size]
                pass_logit = first_token_scores[pass_token_id].float()
                fail_logit = first_token_scores[fail_token_id].float()
                probs = torch.softmax(torch.stack([pass_logit, fail_logit]), dim=0)
                prob_pass = probs[0].item()
                prob_fail = probs[1].item()

                if i == 0:
                    print(f"  Token probs working: P({pass_str})={prob_pass:.4f}, "
                          f"P({fail_str})={prob_fail:.4f}")

        # Decode response (skip the input tokens)
        input_len = inputs["input_ids"].shape[-1]
        response = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)

        # Use token probability for classification and confidence
        # defect_score: probability that the image is defective (0 = good, 1 = defect)
        defect_score = prob_fail

        # Parse verdict from response text as primary classification
        response_upper = response.strip().upper()
        if response_upper.startswith(pass_str.upper()):
            pred_label = 0
            confidence = prob_pass
        elif response_upper.startswith(fail_str.upper()):
            pred_label = 1
            confidence = prob_fail
        else:
            # Ambiguous response — use probability to decide
            if prob_fail > prob_pass:
                pred_label = 1
                confidence = prob_fail
            elif fail_str.upper() in response_upper or "DEFECT" in response_upper:
                pred_label = 1
                confidence = max(prob_fail, 0.6)
            else:
                pred_label = 0
                confidence = max(prob_pass, 0.5)

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_scores.append(defect_score)  # continuous prob for AUROC

        predictions.append({
            "index": i,
            "true_label": "defect" if true_label == 1 else "good",
            "pred_label": "defect" if pred_label == 1 else "good",
            "confidence": confidence,
            "defect_score": defect_score,
            "prob_pass": prob_pass,
            "prob_fail": prob_fail,
            "response": response[:200],
        })

        if (i + 1) % 10 == 0 or i == len(val_samples) - 1:
            print(f"  Evaluated {i + 1}/{len(val_samples)} samples")

    # Compute metrics
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_scores_arr = np.array(y_scores)

    # Handle edge case: if all predictions are the same class, AUROC is undefined
    if len(np.unique(y_true_arr)) < 2:
        print("WARNING: Only one class in ground truth — AUROC undefined, setting to 0.5")
        auroc = 0.5
    elif len(np.unique(y_pred_arr)) < 2:
        print("WARNING: Model predicts same class for everything — degenerate model")
        auroc = roc_auc_score(y_true_arr, y_scores_arr)
    else:
        auroc = roc_auc_score(y_true_arr, y_scores_arr)

    # Find optimal threshold on defect_score where both P >= 0.75 and R >= 0.75
    # (or best F1 if that's not achievable)
    from sklearn.metrics import precision_recall_curve
    precisions_curve, recalls_curve, thresholds_curve = precision_recall_curve(
        y_true_arr, y_scores_arr,
    )

    best_threshold = 0.5
    best_f1_at_threshold = 0.0
    balanced_threshold = None  # threshold where P >= 0.75 AND R >= 0.75

    for t_idx, threshold in enumerate(thresholds_curve):
        p = precisions_curve[t_idx]
        r = recalls_curve[t_idx]
        if p + r > 0:
            f1_t = 2 * p * r / (p + r)
        else:
            f1_t = 0.0
        if f1_t > best_f1_at_threshold:
            best_f1_at_threshold = f1_t
            best_threshold = threshold
        if p >= 0.75 and r >= 0.75 and balanced_threshold is None:
            balanced_threshold = threshold

    # Apply best threshold to get optimized predictions
    opt_threshold = balanced_threshold if balanced_threshold is not None else best_threshold
    y_pred_opt = (y_scores_arr >= opt_threshold).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred_opt).tolist()

    metrics = EvalMetrics(
        auroc=float(auroc),
        f1=float(f1_score(y_true_arr, y_pred_opt, zero_division=0)),
        precision=float(precision_score(y_true_arr, y_pred_opt, zero_division=0)),
        recall=float(recall_score(y_true_arr, y_pred_opt, zero_division=0)),
        accuracy=float(accuracy_score(y_true_arr, y_pred_opt)),
        confusion_matrix=cm,
        num_test_samples=len(val_samples),
        predictions=predictions,
    )

    # Also compute raw (text-based) metrics for comparison (computed for side-effect parity)
    confusion_matrix(y_true_arr, y_pred_arr).tolist()

    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  AUROC:     {metrics.auroc:.4f}")
    print(f"\n  At optimal threshold ({opt_threshold:.4f}):")
    print(f"  F1:        {metrics.f1:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print("\n  Confusion Matrix (threshold-optimized):")
    print("              Pred Good  Pred Defect")
    print(f"  True Good   {cm[0][0]:>9}  {cm[0][1]:>11}")
    print(f"  True Defect {cm[1][0]:>9}  {cm[1][1]:>11}")
    if balanced_threshold is not None:
        print("\n  Balanced threshold found: P >= 0.75 AND R >= 0.75")
    else:
        print("\n  No threshold achieves P >= 0.75 AND R >= 0.75; using best F1 threshold")
    print("\n  Raw text-based results (before threshold optimization):")
    print(f"  F1:        {f1_score(y_true_arr, y_pred_arr, zero_division=0):.4f}")
    print(f"  Precision: {precision_score(y_true_arr, y_pred_arr, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true_arr, y_pred_arr, zero_division=0):.4f}")
    print(f"\n  Gate: AUROC >= 0.90 → {'PASS' if metrics.auroc >= 0.90 else 'FAIL'}")
    print(f"{'='*60}\n")

    # Print failure cases
    failures = [p for p in predictions if p["true_label"] != p["pred_label"]]
    if failures:
        print(f"\nFailure gallery ({len(failures)} misclassifications):")
        for f in failures[:10]:
            print(f"  [{f['index']}] True: {f['true_label']}, Pred: {f['pred_label']} "
                  f"(conf={f['confidence']:.2f})")
            print(f"       Response: {f['response'][:100]}...")

    return metrics


# ---------------------------------------------------------------------------
# Reproducibility Baseline
# ---------------------------------------------------------------------------

def save_baseline(
    config: ExperimentConfig,
    metrics: EvalMetrics,
    train_indices: list[int],
    val_indices: list[int],
    output_path: Path,
) -> None:
    """Save reproducibility baseline for Phase 1 regression testing.

    Args:
        config: The exact configuration used.
        metrics: The evaluation results achieved.
        train_indices: Which sample indices went to training.
        val_indices: Which sample indices went to validation.
        output_path: Where to save the baseline JSON.
    """
    baseline = {
        "phase": 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": config.to_dict(),
        "metrics": {
            "auroc": metrics.auroc,
            "f1": metrics.f1,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "accuracy": metrics.accuracy,
            "confusion_matrix": metrics.confusion_matrix,
            "num_test_samples": metrics.num_test_samples,
        },
        "data_split": {
            "train_indices": train_indices,
            "val_indices": val_indices,
        },
        "gate": {
            "target": "AUROC >= 0.90",
            "passed": metrics.auroc >= 0.90,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"Baseline saved to {output_path}")


# ---------------------------------------------------------------------------
# Loss Curve Plotting
# ---------------------------------------------------------------------------

def plot_loss_curve(output_dir: Path) -> None:
    """Plot training and eval loss from the training log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log_path = output_dir / "training_log.json"
    if not log_path.exists():
        print("No training log found, skipping loss curve plot")
        return

    with open(log_path) as f:
        log_history = json.load(f)

    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    if not train_steps:
        print("No training loss entries found in log")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_steps, train_losses, label="Train Loss", alpha=0.7)
    if eval_steps:
        ax.plot(eval_steps, eval_losses, label="Eval Loss", marker="o", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Phase 0: Training Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = output_dir / "loss_curve.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0: Validate Gemma 4 for defect detection on MVTec AD hazelnut",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("./mvtec_hazelnut"),
        help="Path to dataset directory with good/ and defect/ subdirectories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./phase0_output"),
        help="Directory for training outputs (adapter, logs, baseline)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/gemma-4-26B-A4B-it",
        help="HuggingFace model ID (default: unsloth/gemma-4-26B-A4B-it)",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download MVTec AD hazelnut dataset before training",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed (default: 3407)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only run evaluation on existing adapter",
    )

    args = parser.parse_args()

    # Build config from args
    config = ExperimentConfig(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_rank,  # alpha == rank
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print("FSVLM Phase 0: ML Validation")
    print(f"{'='*60}")
    print(f"Model:    {config.model_name}")
    print(f"LoRA:     rank={config.lora_rank}, alpha={config.lora_alpha}")
    print(f"Training: {config.num_train_epochs} epochs, lr={config.learning_rate}")
    print(f"Seed:     {config.seed}")
    print(f"Data:     {args.data}")
    print(f"Output:   {args.output}")
    print(f"{'='*60}\n")

    # Step 0: Download data if requested
    if args.download_data:
        data_dir = download_mvtec_hazelnut(args.data.parent)
        args.data = data_dir
    elif not args.data.exists():
        print(f"ERROR: Data directory not found: {args.data}")
        print("Run with --download-data to download MVTec AD hazelnut, or provide --data path")
        sys.exit(1)

    # Step 1: Prepare data
    print("\n[1/4] Preparing data...")
    train_samples, val_samples, train_indices, val_indices = prepare_data(args.data, config)

    # Step 2: Train adapter (skip if --eval-only)
    adapter_path = args.output / "final_adapter"
    if args.eval_only:
        if not adapter_path.exists():
            print(f"ERROR: --eval-only but no adapter found at {adapter_path}")
            sys.exit(1)
        print(f"\n[2/4] Skipping training (--eval-only), using adapter at {adapter_path}")
    else:
        print("\n[2/4] Training adapter...")
        adapter_path = train_adapter(train_samples, val_samples, config, args.output)

    # Step 3: Evaluate
    print("\n[3/4] Evaluating...")
    if args.eval_only:
        # Direct evaluation — GPU memory is clean (no prior training in this process)
        metrics = evaluate(adapter_path, val_samples, config)
    else:
        # After training, GPU memory is polluted (PyTorch CUDA context persists).
        # Run evaluation in a subprocess to get a clean CUDA state.
        print("Running evaluation in subprocess for clean GPU memory...")
        import subprocess as _sp
        eval_result = _sp.run(
            [
                sys.executable, __file__,
                "--data", str(args.data),
                "--output", str(args.output),
                "--model-name", config.model_name,
                "--lora-rank", str(config.lora_rank),
                "--epochs", str(config.num_train_epochs),
                "--lr", str(config.learning_rate),
                "--seed", str(config.seed),
                "--eval-only",
            ],
            capture_output=True, text=True,
        )
        print(eval_result.stdout)
        if eval_result.returncode != 0:
            print("Evaluation subprocess failed:")
            print(eval_result.stderr[-2000:] if len(eval_result.stderr) > 2000 else eval_result.stderr)
            sys.exit(1)

        # Read metrics from the baseline file that --eval-only saved
        baseline_path = args.output / "phase0_baseline.json"
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        metrics = EvalMetrics(
            auroc=baseline_data["metrics"]["auroc"],
            f1=baseline_data["metrics"]["f1"],
            precision=baseline_data["metrics"]["precision"],
            recall=baseline_data["metrics"]["recall"],
            accuracy=baseline_data["metrics"]["accuracy"],
            confusion_matrix=baseline_data["metrics"]["confusion_matrix"],
            num_test_samples=baseline_data["metrics"]["num_test_samples"],
        )

    # Step 4: Save baseline
    print("\n[4/4] Saving baseline...")
    baseline_path = args.output / "phase0_baseline.json"
    save_baseline(config, metrics, train_indices, val_indices, baseline_path)

    # Also copy baseline to experiments/ for easy access
    experiments_baseline = Path(__file__).parent / "phase0_baseline.json"
    with open(baseline_path) as f:
        baseline_data = json.load(f)
    with open(experiments_baseline, "w") as f:
        json.dump(baseline_data, f, indent=2)
    print(f"Baseline also saved to {experiments_baseline}")

    # Plot loss curve
    plot_loss_curve(args.output)

    # Final verdict
    print(f"\n{'='*60}")
    if metrics.auroc >= 0.90:
        print("GATE PASSED: AUROC >= 0.90")
        print("Proceed to Phase 1: Build the agent architecture.")
    else:
        print(f"GATE FAILED: AUROC = {metrics.auroc:.4f} (need >= 0.90)")
        print("Options:")
        print("  1. Try different hyperparameters (--lora-rank, --lr, --epochs)")
        print("  2. Try fallback model: --model-name unsloth/gemma-4-E4B-it")
        print("  3. Reassess model choice entirely")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
