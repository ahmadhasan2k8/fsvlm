"""Inspector Agent — runs inference on images.

Supports single image, batch, watch mode, and serves as the inference
engine for the REST API. Model is loaded once and kept warm for fast
sequential inference.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from fsvlm.config import FSVLMConfig
from fsvlm.types import InspectionResult

if TYPE_CHECKING:
    pass


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class InspectorAgent:
    """Runs defect detection inference on images using a trained adapter.

    Args:
        config: FSVLM configuration.
    """

    def __init__(self, config: FSVLMConfig) -> None:
        self._config = config
        self._model = None
        self._tokenizer = None
        self._adapter_path: Path | None = None
        self._pass_token_id: int = 0
        self._fail_token_id: int = 0

    @property
    def is_loaded(self) -> bool:
        """Whether a model/adapter is currently loaded."""
        return self._model is not None

    def load_adapter(self, adapter_path: Path) -> None:
        """Load a trained adapter for inference.

        Args:
            adapter_path: Path to the adapter directory.
        """
        import transformers.modeling_utils
        from loguru import logger

        transformers.modeling_utils.caching_allocator_warmup = lambda *a, **kw: None

        from unsloth import FastVisionModel

        logger.info(f"Loading adapter from {adapter_path}")

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=self._config.max_seq_length,
            load_in_4bit=self._config.load_in_4bit,
            load_in_16bit=not self._config.load_in_4bit,
            device_map="cuda:0",
        )
        FastVisionModel.for_inference(model)

        _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        self._pass_token_id = _tok.encode("PASS", add_special_tokens=False)[0]
        self._fail_token_id = _tok.encode("FAIL", add_special_tokens=False)[0]

        self._model = model
        self._tokenizer = tokenizer
        self._adapter_path = adapter_path

    def unload(self) -> None:
        """Free GPU memory by unloading the model."""
        if self._model is not None:
            import gc

            import torch

            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

    def inspect(
        self,
        image_path: Path,
        prompt: str | None = None,
        threshold: float = 0.5,
    ) -> InspectionResult:
        """Run inference on a single image.

        Args:
            image_path: Path to the image file.
            prompt: Inspection prompt override.
            threshold: Defect score threshold (above = FAIL).

        Returns:
            InspectionResult with pass/fail, confidence, and description.

        Raises:
            RuntimeError: If no adapter is loaded.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No adapter loaded. Call load_adapter() first.")

        import torch

        from fsvlm.prompts.generic import INSPECTION_PROMPT
        from fsvlm.utils.image import load_image

        start = time.perf_counter()

        img = load_image(image_path, max_size=self._config.max_image_size)
        inspection_prompt = prompt or INSPECTION_PROMPT

        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": inspection_prompt},
                ],
            }
        ]

        prompt_text = self._tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            text=prompt_text,
            images=[img],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            gen_output = self._model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Token probabilities
        prob_pass = 0.5
        prob_fail = 0.5
        if hasattr(gen_output, "scores") and gen_output.scores:
            first_scores = gen_output.scores[0][0]
            pass_logit = first_scores[self._pass_token_id].float()
            fail_logit = first_scores[self._fail_token_id].float()
            probs = torch.softmax(torch.stack([pass_logit, fail_logit]), dim=0)
            prob_pass = probs[0].item()
            prob_fail = probs[1].item()

        # Decode response
        input_len = inputs["input_ids"].shape[-1]
        response = self._tokenizer.decode(gen_output.sequences[0][input_len:], skip_special_tokens=True)

        # Classify
        is_defect = prob_fail >= threshold
        confidence = prob_fail if is_defect else prob_pass

        elapsed_ms = (time.perf_counter() - start) * 1000

        adapter_name = self._adapter_path.name if self._adapter_path else ""

        return InspectionResult(
            image_path=image_path,
            pass_fail=not is_defect,  # True = PASS
            confidence=confidence,
            description=response.strip(),
            model_name=self._config.default_model,
            adapter_name=adapter_name,
            inference_time_ms=elapsed_ms,
        )

    def inspect_batch(
        self,
        image_paths: list[Path],
        prompt: str | None = None,
        threshold: float = 0.5,
    ) -> list[InspectionResult]:
        """Run inference on multiple images (model loaded once).

        Args:
            image_paths: List of image file paths.
            prompt: Inspection prompt override.
            threshold: Defect score threshold.

        Returns:
            List of InspectionResult, one per image.
        """
        results: list[InspectionResult] = []
        for path in image_paths:
            try:
                results.append(self.inspect(path, prompt=prompt, threshold=threshold))
            except Exception as e:
                from loguru import logger

                logger.error(f"Failed to inspect {path}: {e}")
        return results

    def watch(
        self,
        watch_dir: Path,
        adapter_path: Path,
        threshold: float = 0.5,
        callback: Callable[[InspectionResult], None] | None = None,
    ) -> None:
        """Watch a directory for new images and inspect them.

        Blocks until interrupted (Ctrl+C / SIGINT).

        Args:
            watch_dir: Directory to monitor for new images.
            adapter_path: Path to the trained adapter.
            threshold: Defect score threshold.
            callback: Optional function called with each result.
        """
        from loguru import logger

        try:
            from watchdog.events import FileCreatedEvent, FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            raise ImportError(
                "Watch mode requires watchdog. Install with: "
                "pip install 'fsvlm[watch]' or pip install watchdog"
            )

        if not self.is_loaded:
            self.load_adapter(adapter_path)

        debounce = self._config.watch_debounce_seconds
        logger.info(f"Watching {watch_dir} for new images (debounce={debounce}s)")

        agent = self

        class _Handler(FileSystemEventHandler):
            def __init__(self) -> None:
                self._last_event: float = 0

            def on_created(self, event: FileCreatedEvent) -> None:
                if event.is_directory:
                    return
                path = Path(event.src_path)
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    return

                # Debounce
                now = time.time()
                if now - self._last_event < debounce:
                    return
                self._last_event = now

                # Small delay to let the file finish writing
                time.sleep(0.5)

                try:
                    result = agent.inspect(path, threshold=threshold)
                    status = "PASS" if result.pass_fail else "FAIL"
                    logger.info(f"{path.name}: {status} ({result.confidence:.1%})")
                    if callback:
                        callback(result)
                except Exception as e:
                    logger.error(f"Failed to inspect {path.name}: {e}")

        observer = Observer()
        observer.schedule(_Handler(), str(watch_dir), recursive=False)
        observer.start()

        try:
            while observer.is_alive():
                observer.join(timeout=1)
        except KeyboardInterrupt:
            logger.info("Watch mode stopped")
        finally:
            observer.stop()
            observer.join()


class InspectorSession:
    """Context manager that loads model on enter and frees GPU on exit.

    Usage::

        with InspectorSession(config, adapter_path) as inspector:
            result = inspector.inspect(Path("image.jpg"))
        # GPU memory freed here
    """

    def __init__(self, config: FSVLMConfig, adapter_path: Path) -> None:
        self._inspector = InspectorAgent(config)
        self._adapter_path = adapter_path

    def __enter__(self) -> InspectorAgent:
        self._inspector.load_adapter(self._adapter_path)
        return self._inspector

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        self._inspector.unload()
