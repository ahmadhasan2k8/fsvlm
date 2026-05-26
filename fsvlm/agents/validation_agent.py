"""Validation Agent — evaluates trained adapters.

Cython-compatible: no module-level torch/transformers/sklearn imports,
no exec/eval, no complex metaclasses. Heavy imports inside function bodies.
Dependencies injected via constructor.
"""

from __future__ import annotations

from pathlib import Path

from fsvlm.config import FSVLMConfig
from fsvlm.types import (
    ConfusionMatrixData,
    FailureCase,
    LabeledSample,
    TrainingConfig,
    ValidationMetrics,
    ValidationReport,
)


class ValidationAgent:
    """Evaluates a trained adapter on a held-out validation set.

    Computes AUROC, F1, precision, recall, accuracy, confusion matrix,
    and identifies failure cases. Console output only (no HTML — Phase 2).

    Args:
        config: FSVLM configuration.
    """

    def __init__(self, config: FSVLMConfig) -> None:
        self._config = config

    def validate(
        self,
        adapter_path: Path,
        val_samples: list[LabeledSample],
        training_config: TrainingConfig | None = None,
    ) -> ValidationReport:
        """Run evaluation on the validation set.

        Loads the adapter, runs inference, computes metrics with threshold
        optimization, and collects failure cases.

        Args:
            adapter_path: Path to the saved adapter directory.
            val_samples: Validation samples with ground truth labels.
            training_config: Config used during training (for prompt/model info).

        Returns:
            ValidationReport with metrics, confusion matrix, and failure cases.
        """
        tc = training_config or TrainingConfig(model_name=self._config.default_model)

        predictions = self._run_inference(adapter_path, val_samples, tc)
        report = self._compute_metrics(predictions, val_samples)
        return report

    def _run_inference(
        self,
        adapter_path: Path,
        val_samples: list[LabeledSample],
        tc: TrainingConfig,
    ) -> list[dict]:
        """Load adapter and run inference on all validation samples."""
        import torch

        # Disable caching_allocator_warmup — OOMs on large quantized models
        import transformers.modeling_utils
        from loguru import logger

        transformers.modeling_utils.caching_allocator_warmup = lambda *a, **kw: None

        from unsloth import FastVisionModel

        from fsvlm.utils.image import load_image

        logger.info(f"Loading adapter from {adapter_path}")

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=tc.max_seq_length,
            load_in_4bit=tc.load_in_4bit,
            load_in_16bit=not tc.load_in_4bit,
            device_map="cuda:0",
        )
        FastVisionModel.for_inference(model)

        device = next(model.parameters()).device

        from fsvlm.prompts.verdict import (
            resolve_inspection_prompt,
            verdict_token_ids,
            verdict_tokens,
        )

        pass_token_id, fail_token_id = verdict_token_ids(tokenizer, tc.model_name)
        pass_str, fail_str = verdict_tokens(tc.model_name)
        resolved_prompt = resolve_inspection_prompt(tc.inspection_prompt, tc.model_name)

        predictions: list[dict] = []

        for i, sample in enumerate(val_samples):
            img = load_image(sample.image_path, max_size=tc.max_image_size)

            chat_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": resolved_prompt},
                    ],
                }
            ]

            prompt = tokenizer.apply_chat_template(
                chat_messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            inputs = tokenizer(
                text=prompt,
                images=[img],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                gen_output = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Extract token probabilities
            prob_pass = 0.5
            prob_fail = 0.5
            if hasattr(gen_output, "scores") and gen_output.scores:
                first_scores = gen_output.scores[0][0]
                pass_logit = first_scores[pass_token_id].float()
                fail_logit = first_scores[fail_token_id].float()
                probs = torch.softmax(torch.stack([pass_logit, fail_logit]), dim=0)
                prob_pass = probs[0].item()
                prob_fail = probs[1].item()

            # Decode response text
            input_len = inputs["input_ids"].shape[-1]
            response = tokenizer.decode(gen_output.sequences[0][input_len:], skip_special_tokens=True)

            # Classify based on response text
            response_upper = response.strip().upper()
            if response_upper.startswith(pass_str.upper()):
                text_pred = 0
                confidence = prob_pass
            elif response_upper.startswith(fail_str.upper()):
                text_pred = 1
                confidence = prob_fail
            else:
                if prob_fail > prob_pass:
                    text_pred = 1
                    confidence = prob_fail
                else:
                    text_pred = 0
                    confidence = prob_pass

            true_label = 0 if sample.label == "good" else 1

            predictions.append(
                {
                    "index": i,
                    "image_path": sample.image_path,
                    "true_label": true_label,
                    "text_pred": text_pred,
                    "confidence": confidence,
                    "defect_score": prob_fail,
                    "response": response[:200],
                }
            )

            if (i + 1) % 10 == 0 or i == len(val_samples) - 1:
                logger.info(f"Evaluated {i + 1}/{len(val_samples)} samples")

        # Free GPU memory
        del model, tokenizer
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        return predictions

    def _compute_metrics(
        self,
        predictions: list[dict],
        val_samples: list[LabeledSample],
    ) -> ValidationReport:
        """Compute metrics with threshold optimization."""
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_true = np.array([p["true_label"] for p in predictions])
        y_scores = np.array([p["defect_score"] for p in predictions])

        # AUROC
        if len(np.unique(y_true)) < 2:
            auroc = 0.5
        else:
            auroc = float(roc_auc_score(y_true, y_scores))

        # Threshold optimization: find threshold where P >= min_p AND R >= min_r
        precisions_c, recalls_c, thresholds_c = precision_recall_curve(y_true, y_scores)

        best_threshold = 0.5
        best_f1 = 0.0
        balanced_threshold = None
        min_p = self._config.min_precision
        min_r = self._config.min_recall

        for t_idx, threshold in enumerate(thresholds_c):
            p = precisions_c[t_idx]
            r = recalls_c[t_idx]
            f1_t = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = threshold
            if p >= min_p and r >= min_r and balanced_threshold is None:
                balanced_threshold = threshold

        opt_threshold = balanced_threshold if balanced_threshold is not None else best_threshold
        y_pred = (y_scores >= opt_threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred).tolist()

        metrics = ValidationMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            auroc=auroc,
            optimal_threshold=float(opt_threshold),
        )

        # Collect failure cases
        failures: list[FailureCase] = []
        for p in predictions:
            pred_at_threshold = 1 if p["defect_score"] >= opt_threshold else 0
            if pred_at_threshold != p["true_label"]:
                actual = "good" if p["true_label"] == 0 else "defect"
                predicted = "good" if pred_at_threshold == 0 else "defect"
                failures.append(
                    FailureCase(
                        image_path=p["image_path"],
                        predicted=predicted,
                        actual=actual,
                        confidence=p["confidence"],
                        model_reasoning=p["response"],
                    )
                )

        scores_list = [p["defect_score"] for p in predictions]

        return ValidationReport(
            metrics=metrics,
            confusion_matrix=ConfusionMatrixData(matrix=cm),
            failure_cases=failures,
            num_test_samples=len(predictions),
            confidence_scores=scores_list,
        )
