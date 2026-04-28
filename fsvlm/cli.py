"""CLI entry point for FSVLM.

Commands: setup, train, inspect, validate, watch, serve, ui, correct, retrain, version.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

import fsvlm

app = typer.Typer(
    name="fsvlm",
    help="Fine-tune Gemma 4 for visual defect detection. Locally.",
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"fsvlm {fsvlm.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    _version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show the fsvlm version and exit.",
    ),
) -> None:
    """fsvlm — few-shot fine-tuning benchmarker for vision-language models."""


@app.command()
def version() -> None:
    """Show FSVLM version."""
    console.print(f"fsvlm {fsvlm.__version__}")


@app.command()
def setup(
    model: str | None = typer.Option(None, help="Model to download: small, medium, large, or HF repo name"),
    check: bool = typer.Option(False, "--check", help="Verify installation (GPU, model, deps)"),
) -> None:
    """Detect GPU, recommend and download a model."""
    from fsvlm.config import load_config
    from fsvlm.models.hardware import (
        KNOWN_MODELS,
        detect_gpu,
        get_model_by_name,
        recommend_model,
    )

    config = load_config()
    gpu = detect_gpu()

    if check:
        console.print("[bold]System Check[/bold]")
        if gpu.is_available:
            console.print(f"  GPU: [green]{gpu.name}[/green]")
            console.print(f"  VRAM: {gpu.vram_total_gb:.1f}GB total, {gpu.vram_free_gb:.1f}GB free")
            console.print(f"  CUDA: {gpu.cuda_version}")
            console.print(f"  Compute: sm_{gpu.compute_capability[0]}{gpu.compute_capability[1]}")
        else:
            console.print("  GPU: [red]Not found[/red]")
            console.print("  FSVLM requires a CUDA-capable GPU for training.")
        return

    console.print("[bold]FSVLM Setup[/bold]\n")

    if gpu.is_available:
        console.print(f"GPU: [green]{gpu.name}[/green] ({gpu.vram_total_gb:.1f}GB VRAM)")
    else:
        console.print("[red]No NVIDIA GPU detected.[/red]")
        console.print("FSVLM requires a CUDA-capable GPU for training.")
        console.print("Inference can run on CPU (slower).")
        raise typer.Exit(1)

    # Select model
    if model:
        model_info = get_model_by_name(model)
        if model_info is None:
            size_map = {m.size_label: m for m in KNOWN_MODELS}
            model_info = size_map.get(model)
        if model_info is None:
            console.print(f"[red]Unknown model: {model}[/red]")
            console.print(f"Available: {', '.join(m.name for m in KNOWN_MODELS)}")
            raise typer.Exit(1)
    else:
        model_info = recommend_model(gpu)
        console.print(f"Recommended model: [bold]{model_info.name}[/bold] ({model_info.size_label})")

    from fsvlm.models.downloader import download_model, is_model_cached

    if is_model_cached(model_info):
        console.print(f"[green]Model already downloaded: {model_info.name}[/green]")
    else:
        console.print(f"\nDownloading {model_info.name} (~{model_info.vram_required_gb:.0f}GB)...")
        download_model(model_info)

    config.ensure_dirs()
    console.print("\n[green]Setup complete![/green]")


@app.command()
def train(
    images: Path = typer.Option(..., "--images", help="Path to image dir, CSV, or JSON file"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory for adapter"),
    epochs: int | None = typer.Option(None, "--epochs", help="Number of training epochs"),
    model: str | None = typer.Option(None, "--model", help="Model name or size label"),
    lora_rank: int | None = typer.Option(None, "--lora-rank", help="LoRA rank (overrides config default)"),
    lora_alpha: int | None = typer.Option(None, "--lora-alpha", help="LoRA alpha (overrides config default)"),
    learning_rate: float | None = typer.Option(None, "--learning-rate", "--lr", help="Learning rate (overrides config default)"),
    no_sweep: bool = typer.Option(False, "--no-sweep", help="Skip auto-research sweep, use single config"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
) -> None:
    """Train a defect detection adapter on labeled images."""
    from fsvlm.agents.orchestrator import Orchestrator
    from fsvlm.config import load_config
    from fsvlm.models.downloader import download_model, is_model_cached
    from fsvlm.models.hardware import get_model_by_name
    from fsvlm.types import LoRAConfig, TrainingConfig

    config = load_config()
    tc = TrainingConfig(
        model_name=config.default_model,
        lora=LoRAConfig(rank=config.default_lora_rank, alpha=config.default_lora_alpha),
        learning_rate=config.default_learning_rate,
        num_train_epochs=config.default_max_epochs,
    )

    if model:
        model_info = get_model_by_name(model)
        if model_info:
            tc.model_name = model_info.hf_repo
        else:
            tc.model_name = model

    if epochs:
        tc.num_train_epochs = epochs

    if lora_rank is not None:
        tc.lora.rank = lora_rank
    if lora_alpha is not None:
        tc.lora.alpha = lora_alpha
    if learning_rate is not None:
        tc.learning_rate = learning_rate

    # Check model availability
    model_info = get_model_by_name(tc.model_name)
    if model_info and not is_model_cached(model_info):
        console.print(
            f"Model not found locally. Downloading {model_info.name} (~{model_info.vram_required_gb:.0f}GB)."
        )
        if not yes:
            typer.confirm("Continue?", abort=True)
        download_model(model_info)

    orchestrator = Orchestrator(config)
    try:
        result, report = orchestrator.train(
            images,
            output_dir=output,
            training_config=tc,
            sweep=not no_sweep,
        )
        console.print(f"\n[green]Adapter saved to: {result.adapter_path}[/green]")
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def inspect(
    image: Path = typer.Argument(..., help="Path to image file or directory for batch"),
    adapter: Path | None = typer.Option(None, "--adapter", help="Path to trained adapter"),
    output: str = typer.Option("text", "--output", "-o", help="Output format: text or json"),
) -> None:
    """Run defect detection on an image or batch of images."""
    import json as json_mod

    from fsvlm.agents.orchestrator import Orchestrator
    from fsvlm.config import load_config

    if not image.exists():
        console.print(f"[red]Not found: {image}[/red]")
        raise typer.Exit(1)

    config = load_config()
    orchestrator = Orchestrator(config)

    # Batch mode: directory of images
    if image.is_dir():
        from fsvlm.agents.inspector_agent import IMAGE_EXTENSIONS

        image_files = sorted(
            f for f in image.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_files:
            console.print(f"[red]No images found in {image}[/red]")
            raise typer.Exit(1)

        # Load model once for all images
        results = orchestrator.inspect_batch(image_files, adapter_path=adapter)

        if output == "json":
            data = [
                {
                    "image": str(r.image_path),
                    "pass_fail": "PASS" if r.pass_fail else "FAIL",
                    "confidence": round(r.confidence, 4),
                    "description": r.description,
                    "inference_time_ms": round(r.inference_time_ms, 1),
                }
                for r in results
            ]
            console.print(json_mod.dumps(data, indent=2))
        else:
            for r in results:
                status = "[green]PASS[/green]" if r.pass_fail else "[red]FAIL[/red]"
                console.print(f"{r.image_path.name}: {status} ({r.confidence:.1%})")
        return

    # Single image mode
    try:
        result = orchestrator.inspect(image, adapter_path=adapter)
    except Exception as e:
        console.print(f"[red]Inspection failed: {e}[/red]")
        raise typer.Exit(1)

    if output == "json":
        data = {
            "image": str(result.image_path),
            "pass_fail": "PASS" if result.pass_fail else "FAIL",
            "confidence": round(result.confidence, 4),
            "description": result.description,
            "inference_time_ms": round(result.inference_time_ms, 1),
        }
        console.print(json_mod.dumps(data, indent=2))
    else:
        status = "[green]PASS[/green]" if result.pass_fail else "[red]FAIL[/red]"
        console.print(f"\nResult: {status}")
        console.print(f"Confidence: {result.confidence:.1%}")
        console.print(f"Description: {result.description}")
        console.print(f"Time: {result.inference_time_ms:.0f}ms")


@app.command()
def validate(
    adapter: Path = typer.Option(..., "--adapter", help="Path to trained adapter"),
    images: Path = typer.Option(..., "--images", help="Path to test image dir, CSV, or JSON"),
) -> None:
    """Evaluate a trained adapter on a test dataset."""
    from fsvlm.agents.data_agent import DataAgent
    from fsvlm.agents.orchestrator import Orchestrator
    from fsvlm.config import load_config

    config = load_config()
    data_agent = DataAgent(config)

    # Read test data
    dataset = data_agent.prepare(images, test_split=1.0)  # all samples go to val

    orchestrator = Orchestrator(config)
    report = orchestrator._validate_subprocess(adapter, dataset.val_samples, None)
    orchestrator._print_report(report)


@app.command()
def watch(
    directory: Path = typer.Argument(..., help="Directory to watch for new images"),
    adapter: Path | None = typer.Option(None, "--adapter", help="Path to trained adapter"),
) -> None:
    """Watch a directory for new images and inspect them automatically."""
    from fsvlm.agents.inspector_agent import InspectorAgent
    from fsvlm.agents.orchestrator import Orchestrator
    from fsvlm.config import load_config
    from fsvlm.types import InspectionResult

    if not directory.exists():
        console.print(f"[red]Directory not found: {directory}[/red]")
        raise typer.Exit(1)

    config = load_config()

    if adapter is None:
        orchestrator = Orchestrator(config)
        try:
            adapter = orchestrator._find_latest_adapter()
        except Exception as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    console.print(f"Watching [bold]{directory}[/bold] for new images")
    console.print(f"Using adapter: {adapter}")
    console.print("Press Ctrl+C to stop\n")

    def on_result(result: InspectionResult) -> None:
        status = "[green]PASS[/green]" if result.pass_fail else "[red]FAIL[/red]"
        console.print(f"{result.image_path.name}: {status} ({result.confidence:.1%})")

    inspector = InspectorAgent(config)
    try:
        inspector.watch(directory, adapter, callback=on_result)
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    finally:
        inspector.unload()


@app.command()
def serve(
    adapter: Path | None = typer.Option(None, "--adapter", help="Path to trained adapter"),
    host: str = typer.Option("0.0.0.0", "--host", help="Server bind host"),
    port: int = typer.Option(8080, "--port", help="Server bind port"),
) -> None:
    """Launch REST API server for inference."""
    from fsvlm.config import load_config

    config = load_config()

    if adapter is None:
        from fsvlm.agents.orchestrator import Orchestrator

        orchestrator = Orchestrator(config)
        try:
            adapter = orchestrator._find_latest_adapter()
        except Exception as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    console.print(f"Starting server on {host}:{port}")
    console.print(f"Adapter: {adapter}")

    from fsvlm.server.api import run_server

    run_server(config=config, adapter_path=adapter, host=host, port=port)


@app.command()
def ui(
    port: int = typer.Option(7860, "--port", help="Gradio server port"),
    share: bool = typer.Option(False, "--share", help="Create public Gradio link"),
) -> None:
    """Launch Gradio web interface."""
    from fsvlm.config import load_config
    from fsvlm.ui.gradio_app import launch

    config = load_config()
    console.print(f"Launching Gradio UI on port {port}")
    launch(config=config, port=port, share=share)


@app.command()
def correct(
    image: Path = typer.Argument(..., help="Path to the misclassified image"),
    actual: str = typer.Option(..., "--actual", help="Correct label: good or defect"),
    adapter: Path | None = typer.Option(None, "--adapter", help="Adapter that made the error"),
) -> None:
    """Log a correction for a previous inspection result."""
    from fsvlm.agents.feedback_agent import FeedbackAgent
    from fsvlm.config import load_config
    from fsvlm.types import Correction

    if actual not in ("good", "defect"):
        console.print(f"[red]Invalid label: {actual}. Must be 'good' or 'defect'.[/red]")
        raise typer.Exit(1)

    if not image.exists():
        console.print(f"[red]Image not found: {image}[/red]")
        raise typer.Exit(1)

    config = load_config()
    config.ensure_dirs()

    predicted = "defect" if actual == "good" else "good"
    adapter_name = adapter.name if adapter else ""

    correction = Correction(
        image_path=image.resolve(),
        predicted_label=predicted,
        actual_label=actual,
        adapter_name=adapter_name,
    )

    feedback = FeedbackAgent(config)
    count = feedback.log_correction(correction)

    console.print(f"[green]Correction logged[/green] ({count} total pending)")

    if feedback.should_retrain():
        console.print(
            f"\n[yellow]Retrain threshold reached ({count} corrections). "
            f"Run: fsvlm retrain --images <original_data_path>[/yellow]"
        )


@app.command()
def retrain(
    images: Path = typer.Option(..., "--images", help="Path to original training data"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory for new adapter"),
    epochs: int | None = typer.Option(None, "--epochs", help="Number of training epochs"),
) -> None:
    """Retrain with accumulated corrections merged into original data."""
    from fsvlm.agents.orchestrator import Orchestrator
    from fsvlm.config import load_config
    from fsvlm.types import TrainingConfig

    config = load_config()
    tc = TrainingConfig(model_name=config.default_model)

    if epochs:
        tc.num_train_epochs = epochs

    orchestrator = Orchestrator(config)
    try:
        result, report = orchestrator.retrain(
            images,
            output_dir=output,
            training_config=tc,
        )
        console.print(f"\n[green]Retrained adapter saved to: {result.adapter_path}[/green]")
    except Exception as e:
        console.print(f"[red]Retrain failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
