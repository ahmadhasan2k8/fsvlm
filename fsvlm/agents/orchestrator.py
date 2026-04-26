"""Orchestrator Agent — routes commands to specialist agents.

Sequential pipeline: data_agent -> training_agent -> validation_agent.
This is the only agent users interact with directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from fsvlm.config import FSVLMConfig
from fsvlm.types import (
    InspectionResult,
    LoRAConfig,
    PreparedDataset,
    SweepConfig,
    SweepResult,
    TrainingConfig,
    TrainingResult,
    ValidationReport,
)

if TYPE_CHECKING:
    from fsvlm.events import EventBus

# Default sweep configs — rank 16/32/64 with appropriate LRs
DEFAULT_SWEEP_CONFIGS = [
    SweepConfig(rank=16, alpha=16, learning_rate=2e-4, max_epochs=10),
    SweepConfig(rank=32, alpha=32, learning_rate=2e-4, max_epochs=10),
    SweepConfig(rank=64, alpha=64, learning_rate=1e-4, max_epochs=10),
]


class Orchestrator:
    """Wires agents together and manages the train/inspect workflow.

    Args:
        config: FSVLM configuration.
        event_bus: Optional EventBus for progress events.
    """

    def __init__(
        self,
        config: FSVLMConfig,
        event_bus: EventBus | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus

    def train(
        self,
        data_path: Path,
        output_dir: Path | None = None,
        training_config: TrainingConfig | None = None,
        sweep: bool = True,
    ) -> tuple[TrainingResult, ValidationReport]:
        """Run the full training pipeline: data -> train -> validate.

        With sweep=True (default), runs multiple configurations and picks
        the best by F1 score. With sweep=False, trains a single config.

        Args:
            data_path: Directory, CSV, or JSON with labeled data.
            output_dir: Where to save the adapter. Auto-generated if None.
            training_config: Optional training config override.
            sweep: Whether to run auto-research sweep (multiple configs).

        Returns:
            Tuple of (TrainingResult, ValidationReport).
        """
        from fsvlm.agents.data_agent import DataAgent
        from fsvlm.models.adapter import save_adapter_metadata
        from fsvlm.types import AdapterMetadata

        self._config.ensure_dirs()
        tc = training_config or TrainingConfig()
        out = output_dir or self._config.adapters_dir / "latest"

        # Step 1: Prepare data
        logger.info("Step 1: Preparing data")
        data_agent = DataAgent(self._config)
        dataset = data_agent.prepare(data_path)

        logger.info(
            f"Dataset: {dataset.report.total_images} images "
            f"({dataset.report.good_count} good, {dataset.report.defect_count} defect), "
            f"{len(dataset.train_samples)} train, {len(dataset.val_samples)} val"
        )

        # Emit data prep event
        if self._event_bus is not None:
            from fsvlm.events import DataPrepCompleteEvent

            self._event_bus.emit(
                DataPrepCompleteEvent(
                    total_images=dataset.report.total_images,
                    good_count=dataset.report.good_count,
                    defect_count=dataset.report.defect_count,
                    train_count=len(dataset.train_samples),
                    val_count=len(dataset.val_samples),
                )
            )

        # Step 2: Train (single or sweep)
        if sweep and self._config.sweep_enabled and dataset.report.total_images >= 100:
            result, report = self._run_sweep(dataset, tc, out)
        else:
            if sweep and not self._config.sweep_enabled:
                logger.info("Sweep disabled in config, training single config")
            elif sweep and dataset.report.total_images < 100:
                logger.info(
                    f"Dataset too small for sweep ({dataset.report.total_images} < 100), "
                    "training single config"
                )
            result, report = self._train_and_validate(dataset, tc, out)

        # Save adapter metadata
        from fsvlm.models.adapter import next_adapter_version

        version = next_adapter_version(self._config.adapters_dir)

        save_adapter_metadata(
            result.adapter_path,
            AdapterMetadata(
                adapter_name=result.adapter_path.name,
                base_model=tc.model_name,
                lora_rank=tc.lora.rank,
                lora_alpha=tc.lora.alpha,
                training_images=len(dataset.train_samples),
                training_epochs=tc.num_train_epochs,
                validation_metrics=report.metrics,
                prompt_template=tc.inspection_prompt,
                adapter_version=version,
            ),
        )

        # Emit validation event
        if self._event_bus is not None:
            from fsvlm.events import ValidationCompleteEvent

            self._event_bus.emit(
                ValidationCompleteEvent(
                    num_samples=report.num_test_samples,
                    f1=report.metrics.f1,
                    auroc=report.metrics.auroc,
                    accuracy=report.metrics.accuracy,
                )
            )

        # Generate reports
        self._generate_reports(report, out)
        self._print_report(report, result)
        return result, report

    def retrain(
        self,
        data_path: Path,
        output_dir: Path | None = None,
        training_config: TrainingConfig | None = None,
    ) -> tuple[TrainingResult, ValidationReport]:
        """Retrain with accumulated corrections merged into original data.

        Args:
            data_path: Original training data path.
            output_dir: Where to save the new adapter version.
            training_config: Optional training config override.

        Returns:
            Tuple of (TrainingResult, ValidationReport).
        """
        from fsvlm.agents.data_agent import DataAgent
        from fsvlm.agents.feedback_agent import FeedbackAgent

        feedback = FeedbackAgent(self._config)
        corrections = feedback.corrections_as_samples()

        if not corrections:
            logger.warning("No corrections to retrain with")
            raise RuntimeError(
                "No corrections found. Log corrections first: fsvlm correct <image> --actual <label>"
            )

        logger.info(f"Retraining with {len(corrections)} corrections")

        # Prepare original data
        data_agent = DataAgent(self._config)
        dataset = data_agent.prepare(data_path)

        # Merge corrections into training data
        dataset.train_samples.extend(corrections)
        logger.info(f"Merged dataset: {len(dataset.train_samples)} train samples")

        # Find previous adapter for lineage
        tc = training_config or TrainingConfig()
        out = output_dir or self._config.adapters_dir / "retrained"

        result, report = self._train_and_validate(dataset, tc, out)

        # Save metadata with version increment and parent lineage
        from fsvlm.models.adapter import next_adapter_version, save_adapter_metadata
        from fsvlm.types import AdapterMetadata

        version = next_adapter_version(self._config.adapters_dir)
        try:
            parent = str(self._find_latest_adapter())
        except Exception:
            parent = ""

        save_adapter_metadata(
            result.adapter_path,
            AdapterMetadata(
                adapter_name=result.adapter_path.name,
                base_model=tc.model_name,
                lora_rank=tc.lora.rank,
                lora_alpha=tc.lora.alpha,
                training_images=len(dataset.train_samples),
                training_epochs=tc.num_train_epochs,
                validation_metrics=report.metrics,
                prompt_template=tc.inspection_prompt,
                adapter_version=version,
                parent_adapter=parent,
            ),
        )

        # Archive corrections after successful retrain
        feedback.clear_corrections()

        self._generate_reports(report, out)
        self._print_report(report, result)
        return result, report

    def _train_and_validate(
        self,
        dataset: PreparedDataset,
        tc: TrainingConfig,
        output_dir: Path,
    ) -> tuple[TrainingResult, ValidationReport]:
        """Train one config and validate."""
        logger.info(f"Training: rank={tc.lora.rank}, lr={tc.learning_rate}, epochs={tc.num_train_epochs}")
        result = self._train_subprocess(dataset, tc, output_dir)

        logger.info("Validating adapter")
        report = self._validate_subprocess(result.adapter_path, dataset.val_samples, result.config)
        return result, report

    def _run_sweep(
        self,
        dataset: PreparedDataset,
        base_tc: TrainingConfig,
        output_dir: Path,
    ) -> tuple[TrainingResult, ValidationReport]:
        """Run auto-research sweep: try multiple configs, pick best by F1."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        sweep_configs = DEFAULT_SWEEP_CONFIGS
        sweep_results: list[SweepResult] = []
        best_result: TrainingResult | None = None
        best_report: ValidationReport | None = None
        best_f1 = -1.0

        logger.info(f"Starting auto-research sweep with {len(sweep_configs)} configs")

        for i, sc in enumerate(sweep_configs):
            logger.info(f"Sweep {i + 1}/{len(sweep_configs)}: rank={sc.rank}, lr={sc.learning_rate}")

            # Emit sweep progress event
            if self._event_bus is not None:
                from fsvlm.events import SweepProgressEvent

                self._event_bus.emit(
                    SweepProgressEvent(
                        candidate_index=i,
                        total_candidates=len(sweep_configs),
                        current_config={"rank": sc.rank, "lr": sc.learning_rate},
                        best_f1_so_far=best_f1 if best_f1 >= 0 else 0.0,
                    )
                )

            # Build config for this sweep candidate
            tc = TrainingConfig(
                model_name=base_tc.model_name,
                load_in_4bit=base_tc.load_in_4bit,
                max_seq_length=base_tc.max_seq_length,
                lora=LoRAConfig(rank=sc.rank, alpha=sc.alpha),
                num_train_epochs=sc.max_epochs,
                per_device_train_batch_size=base_tc.per_device_train_batch_size,
                gradient_accumulation_steps=base_tc.gradient_accumulation_steps,
                learning_rate=sc.learning_rate,
                warmup_ratio=base_tc.warmup_ratio,
                weight_decay=base_tc.weight_decay,
                lr_scheduler_type=base_tc.lr_scheduler_type,
                optim=base_tc.optim,
                bf16=base_tc.bf16,
                seed=base_tc.seed,
                max_image_size=base_tc.max_image_size,
                output_dir=output_dir / f"sweep_{i}",
                inspection_prompt=base_tc.inspection_prompt,
            )

            candidate_dir = output_dir / f"sweep_{i}"

            try:
                result, report = self._train_and_validate(dataset, tc, candidate_dir)

                notes: list[str] = []
                f1 = report.metrics.f1

                sr = SweepResult(
                    config=sc,
                    metrics=report.metrics,
                    train_loss=result.train_loss_history[-1] if result.train_loss_history else 0.0,
                    elapsed_seconds=result.elapsed_seconds,
                    notes=notes,
                )
                sweep_results.append(sr)

                if f1 > best_f1:
                    best_f1 = f1
                    best_result = result
                    best_report = report
                    sr.selected = True

                logger.info(f"Sweep {i + 1}: F1={f1:.4f}, AUROC={report.metrics.auroc:.4f}")

                # Early termination: if F1 > 0.98, skip remaining configs
                if f1 > 0.98:
                    logger.info(f"Early termination: F1={f1:.4f} > 0.98")
                    break

            except Exception as e:
                logger.error(f"Sweep {i + 1} failed: {e}")
                sweep_results.append(
                    SweepResult(
                        config=sc,
                        notes=[f"Failed: {e}"],
                    )
                )

        if best_result is None or best_report is None:
            raise RuntimeError("All sweep configurations failed")

        # Print sweep summary
        table = Table(title="Sweep Results")
        table.add_column("Rank", justify="right")
        table.add_column("LR", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("AUROC", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("", justify="center")

        for sr in sweep_results:
            if sr.metrics:
                table.add_row(
                    str(sr.config.rank),
                    f"{sr.config.learning_rate:.0e}",
                    f"{sr.metrics.f1:.4f}",
                    f"{sr.metrics.auroc:.4f}",
                    f"{sr.elapsed_seconds:.0f}s",
                    "[green]BEST[/green]" if sr.selected else "",
                )
            else:
                table.add_row(
                    str(sr.config.rank),
                    f"{sr.config.learning_rate:.0e}",
                    "FAIL",
                    "FAIL",
                    "-",
                    "",
                )
        console.print(table)

        # Copy best adapter to main output dir
        best_result.sweep_results = sweep_results
        best_result.sweep_reasoning = (
            f"Selected rank={best_result.config.lora.rank} with F1={best_f1:.4f} "
            f"from {len(sweep_results)} candidates"
        )

        return best_result, best_report

    def inspect(
        self,
        image_path: Path,
        adapter_path: Path | None = None,
    ) -> InspectionResult:
        """Run inference on a single image.

        Args:
            image_path: Path to the image.
            adapter_path: Path to a trained adapter. Uses latest if None.

        Returns:
            InspectionResult with pass/fail and description.
        """
        from fsvlm.agents.inspector_agent import InspectorAgent

        if adapter_path is None:
            adapter_path = self._find_latest_adapter()

        inspector = InspectorAgent(self._config)
        inspector.load_adapter(adapter_path)
        try:
            result = inspector.inspect(image_path)
        finally:
            inspector.unload()

        return result

    def inspect_batch(
        self,
        image_paths: list[Path],
        adapter_path: Path | None = None,
    ) -> list[InspectionResult]:
        """Run inference on multiple images (model loaded once).

        Args:
            image_paths: List of image file paths.
            adapter_path: Path to a trained adapter. Uses latest if None.

        Returns:
            List of InspectionResult, one per image.
        """
        from fsvlm.agents.inspector_agent import InspectorAgent

        if adapter_path is None:
            adapter_path = self._find_latest_adapter()

        inspector = InspectorAgent(self._config)
        inspector.load_adapter(adapter_path)
        try:
            results = inspector.inspect_batch(image_paths)
        finally:
            inspector.unload()

        return results

    def _train_subprocess(
        self,
        dataset: PreparedDataset,
        training_config: TrainingConfig,
        output_dir: Path,
    ) -> TrainingResult:
        """Run training in a subprocess so GPU memory is fully freed on exit."""
        import json
        import subprocess
        import sys
        import tempfile
        from dataclasses import asdict

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Serialize dataset
            samples_file = tmp / "samples.json"
            samples_data = {
                "train": [
                    {"image_path": str(s.image_path), "label": s.label, "description": s.description}
                    for s in dataset.train_samples
                ],
                "val": [
                    {"image_path": str(s.image_path), "label": s.label, "description": s.description}
                    for s in dataset.val_samples
                ],
                "report": {
                    "total_images": dataset.report.total_images,
                    "good_count": dataset.report.good_count,
                    "defect_count": dataset.report.defect_count,
                },
                "seed": dataset.seed,
            }
            samples_file.write_text(json.dumps(samples_data))

            # Serialize config
            config_file = tmp / "config.json"
            tc_dict = asdict(training_config)
            tc_dict["output_dir"] = str(tc_dict["output_dir"])
            config_file.write_text(json.dumps(tc_dict))

            result_file = tmp / "result.json"
            output_dir.mkdir(parents=True, exist_ok=True)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fsvlm.agents._train_subprocess",
                    "--samples",
                    str(samples_file),
                    "--config",
                    str(config_file),
                    "--output-dir",
                    str(output_dir),
                    "--result-file",
                    str(result_file),
                ],
                capture_output=False,  # let output stream to console
                timeout=3600,
            )

            if proc.returncode != 0:
                raise RuntimeError("Training subprocess failed")

            result_data = json.loads(result_file.read_text())

        return TrainingResult(
            adapter_path=Path(result_data["adapter_path"]),
            config=training_config,
            train_loss_history=result_data.get("train_loss_history", []),
            elapsed_seconds=result_data.get("elapsed_seconds", 0.0),
        )

    def _validate_subprocess(
        self,
        adapter_path: Path,
        val_samples: list,
        training_config: TrainingConfig | None = None,
    ) -> ValidationReport:
        """Run validation in a subprocess for clean CUDA state.

        After training, the CUDA context retains ~10GB even after
        del model + gc.collect() + empty_cache(). Running validation
        in a subprocess gets a fresh CUDA context.
        """
        import json
        import subprocess
        import sys
        import tempfile
        from dataclasses import asdict

        from fsvlm.types import (
            ConfusionMatrixData,
            FailureCase,
            ValidationMetrics,
        )

        # Serialize inputs to temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            samples_file = tmp / "samples.json"
            samples_data = [
                {"image_path": str(s.image_path), "label": s.label, "description": s.description}
                for s in val_samples
            ]
            samples_file.write_text(json.dumps(samples_data))

            config_file = tmp / "config.json"
            tc_for_val = training_config or TrainingConfig()
            tc_dict = asdict(tc_for_val)
            tc_dict["output_dir"] = str(tc_dict["output_dir"])
            config_file.write_text(json.dumps(tc_dict))

            output_file = tmp / "report.json"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fsvlm.agents._validate_subprocess",
                    "--adapter",
                    str(adapter_path),
                    "--samples",
                    str(samples_file),
                    "--config",
                    str(config_file),
                    "--output",
                    str(output_file),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.error(f"Validation subprocess failed:\n{result.stderr}")
                raise RuntimeError(f"Validation failed: {result.stderr[-500:]}")

            report_data = json.loads(output_file.read_text())

        # Reconstruct ValidationReport
        metrics = ValidationMetrics(**report_data["metrics"])
        cm = ConfusionMatrixData(**report_data["confusion_matrix"])
        failures = [
            FailureCase(
                image_path=Path(f["image_path"]),
                predicted=f["predicted"],
                actual=f["actual"],
                confidence=f["confidence"],
                model_reasoning=f.get("model_reasoning", ""),
            )
            for f in report_data.get("failure_cases", [])
        ]

        return ValidationReport(
            metrics=metrics,
            confusion_matrix=cm,
            failure_cases=failures,
            num_test_samples=report_data.get("num_test_samples", 0),
            confidence_scores=report_data.get("confidence_scores", []),
            summary=report_data.get("summary", ""),
        )

    def _find_latest_adapter(self) -> Path:
        """Find the most recently trained adapter."""
        from fsvlm.exceptions import InvalidAdapterError

        adapters_dir = self._config.adapters_dir
        if not adapters_dir.exists():
            raise InvalidAdapterError(
                str(adapters_dir),
                reason="No adapters directory found. Train a model first.",
            )

        # Look for adapter subdirectories
        candidates = [
            d for d in adapters_dir.iterdir() if d.is_dir() and (d / "adapter_config.json").exists()
        ]

        if not candidates:
            # Check one level deeper (e.g., adapters/latest/adapter/)
            for parent in adapters_dir.iterdir():
                if parent.is_dir():
                    for child in parent.iterdir():
                        if child.is_dir() and (child / "adapter_config.json").exists():
                            candidates.append(child)

        if not candidates:
            raise InvalidAdapterError(
                str(adapters_dir),
                reason="No trained adapters found. Run: fsvlm train --images <path>",
            )

        # Return most recently modified
        return max(candidates, key=lambda d: d.stat().st_mtime)

    def _generate_reports(self, report: ValidationReport, output_dir: Path) -> None:
        """Generate HTML and JSON validation reports."""
        from fsvlm.reports.html_report import HTMLReportGenerator
        from fsvlm.reports.json_report import JSONReportGenerator

        try:
            html_path = HTMLReportGenerator().generate(report, output_dir / "report")
            logger.info(f"HTML report: {html_path}")
        except Exception as e:
            logger.warning(f"Failed to generate HTML report: {e}")

        try:
            json_path = JSONReportGenerator().generate(report, output_dir / "report")
            logger.info(f"JSON report: {json_path}")
        except Exception as e:
            logger.warning(f"Failed to generate JSON report: {e}")

    def _print_report(
        self,
        report: ValidationReport,
        result: TrainingResult | None = None,
    ) -> None:
        """Print validation results to console."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        m = report.metrics

        console.print("\n[bold]Validation Results[/bold]")

        table = Table(show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("AUROC", f"{m.auroc:.4f}")
        table.add_row("F1", f"{m.f1:.4f}")
        table.add_row("Precision", f"{m.precision:.4f}")
        table.add_row("Recall", f"{m.recall:.4f}")
        table.add_row("Accuracy", f"{m.accuracy:.4f}")
        table.add_row("Threshold", f"{m.optimal_threshold:.4f}")
        console.print(table)

        cm = report.confusion_matrix.matrix
        if len(cm) == 2:
            console.print("\n[bold]Confusion Matrix[/bold]")
            cm_table = Table(show_header=True)
            cm_table.add_column("", style="bold")
            cm_table.add_column("Pred Good", justify="right")
            cm_table.add_column("Pred Defect", justify="right")
            cm_table.add_row("True Good", str(cm[0][0]), str(cm[0][1]))
            cm_table.add_row("True Defect", str(cm[1][0]), str(cm[1][1]))
            console.print(cm_table)

        if report.failure_cases:
            console.print(f"\n[yellow]{len(report.failure_cases)} misclassified samples[/yellow]")

        if result and result.sweep_reasoning:
            console.print(f"\n[dim]{result.sweep_reasoning}[/dim]")
