"""Gradio web interface for FSVLM.

5 tabs: Annotate (SAM-assisted), Train, Inspect, Validate, Settings.
Annotate is the primary UX — click on defects, describe them, train.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fsvlm.config import FSVLMConfig
from fsvlm.events import EventBus, TrainingProgressEvent


def create_app(
    config: FSVLMConfig | None = None,
    event_bus: EventBus | None = None,
) -> Any:
    """Create the Gradio Blocks application.

    Args:
        config: FSVLM configuration.
        event_bus: Optional EventBus for training progress updates.

    Returns:
        Configured Gradio Blocks instance.
    """
    import gradio as gr
    import numpy as np
    from PIL import Image as PILImage

    if config is None:
        from fsvlm.config import load_config

        config = load_config()

    if event_bus is None:
        event_bus = EventBus()

    # Shared state
    _progress: dict[str, Any] = {"text": "", "training": False}

    def _on_training_progress(event: TrainingProgressEvent) -> None:
        _progress["text"] = (
            f"Epoch {event.epoch}/{event.total_epochs} | "
            f"Step {event.step}/{event.total_steps} | "
            f"Loss: {event.loss:.4f} | "
            f"LR: {event.learning_rate:.2e} | "
            f"Elapsed: {event.elapsed_seconds:.0f}s"
        )

    event_bus.subscribe(TrainingProgressEvent, _on_training_progress)

    # ---- Annotate Tab State ----
    # Annotation state stored as dicts (Gradio State requires JSON-serializable)
    # Format: {"images": [{"path": str, "annotations": [...], "is_good": bool}]}
    _sam_model: dict[str, Any] = {"instance": None}

    def _get_sam() -> Any:
        """Lazy-load SAM segmenter."""
        if _sam_model["instance"] is None:
            from fsvlm.models.sam import SAMSegmenter

            seg = SAMSegmenter()
            seg.load()
            _sam_model["instance"] = seg
        return _sam_model["instance"]

    def on_image_upload(
        image: np.ndarray | None,
        state: dict,
    ) -> tuple[Any, dict, str]:
        """Handle new image upload — reset annotations for this image."""
        if image is None:
            return None, state, "Upload an image to start annotating."

        # Save uploaded image to temp file
        pil_img = PILImage.fromarray(image)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="dvlm_ann_")
        pil_img.save(tmp.name)
        tmp.close()

        # Store original pixels so we can re-composite overlays on clean image
        state["current_image"] = tmp.name
        state["current_annotations"] = []
        state["original_pixels"] = image.tolist()

        return image, state, _format_annotation_status(state)

    def on_image_click(
        state: dict,
        evt: gr.SelectData,
    ) -> tuple[Any, dict, str]:
        """Handle click on image — run SAM segmentation at click point."""
        if "original_pixels" not in state or "current_image" not in state:
            return None, state, "Upload an image first."

        original = np.array(state["original_pixels"], dtype=np.uint8)
        x, y = evt.index
        sam = _get_sam()

        # Run SAM segmentation on the ORIGINAL image (no overlays)
        from fsvlm.models.sam import mask_to_location

        pil_img = PILImage.fromarray(original)
        result = sam.segment_at_point(pil_img, x, y)

        # Store annotation (without numpy — just serializable data)
        ann = {
            "click_x": int(x),
            "click_y": int(y),
            "mask_rle": result.mask_rle,
            "iou_score": float(result.iou_score),
            "location": mask_to_location(result.mask),
            "description": "",
            "defect_type": "",
        }
        state["current_annotations"].append(ann)

        # Build overlay on the original image
        overlay = _build_overlay(original, state["current_annotations"])

        return overlay, state, _format_annotation_status(state)

    def on_add_description(
        description: str,
        state: dict,
    ) -> tuple[Any, dict, str, str]:
        """Add description to an annotation.

        Supports targeting a specific annotation with '#N' prefix:
          '#2 scratch on edge' → sets annotation #2's description.
        Without a prefix, applies to the first undescribed annotation,
        or the most recent one if all are described.
        """
        import re

        if not state.get("current_annotations"):
            original = (
                np.array(state["original_pixels"], dtype=np.uint8) if "original_pixels" in state else None
            )
            return original, state, "Click on a defect first.", ""

        if not description.strip():
            original = (
                np.array(state["original_pixels"], dtype=np.uint8) if "original_pixels" in state else None
            )
            overlay = _build_overlay(original, state["current_annotations"]) if original is not None else None
            return overlay, state, _format_annotation_status(state), ""

        annotations = state["current_annotations"]

        # Check for #N prefix to target a specific annotation
        match = re.match(r"^#(\d+)\s+(.+)", description.strip())
        if match:
            target_idx = int(match.group(1)) - 1
            desc_text = match.group(2).strip()
            if 0 <= target_idx < len(annotations):
                annotations[target_idx]["description"] = desc_text
            else:
                original = (
                    np.array(state["original_pixels"], dtype=np.uint8) if "original_pixels" in state else None
                )
                overlay = _build_overlay(original, annotations) if original is not None else None
                return (
                    overlay,
                    state,
                    f"No annotation #{target_idx + 1}. You have {len(annotations)} annotations.",
                    "",
                )
        else:
            # Find first undescribed annotation, or fall back to last
            target_idx = len(annotations) - 1
            for i, ann in enumerate(annotations):
                if not ann.get("description"):
                    target_idx = i
                    break
            annotations[target_idx]["description"] = description.strip()

        original = np.array(state["original_pixels"], dtype=np.uint8) if "original_pixels" in state else None
        overlay = _build_overlay(original, annotations) if original is not None else None
        return overlay, state, _format_annotation_status(state), ""

    def on_undo_last(
        state: dict,
    ) -> tuple[Any, dict, str]:
        """Remove the last annotation."""
        if state.get("current_annotations"):
            state["current_annotations"].pop()

        original = np.array(state["original_pixels"], dtype=np.uint8) if "original_pixels" in state else None
        if original is not None and state.get("current_annotations"):
            overlay = _build_overlay(original, state["current_annotations"])
        else:
            overlay = original
        return overlay, state, _format_annotation_status(state)

    def on_mark_good(
        state: dict,
    ) -> tuple[Any, dict, str]:
        """Mark current image as good (no defects)."""
        if "current_image" not in state:
            return None, state, "Upload an image first."

        state["current_annotations"] = []

        # Save to session as good image
        session_images = state.get("session_images", [])
        session_images.append(
            {
                "path": state["current_image"],
                "annotations": [],
                "is_good": True,
            }
        )
        state["session_images"] = session_images

        original = np.array(state["original_pixels"], dtype=np.uint8) if "original_pixels" in state else None
        return original, state, _format_session_status(state)

    def on_save_image(
        state: dict,
    ) -> tuple[Any, dict, str]:
        """Save current image's annotations to session and prepare for next."""
        if "current_image" not in state:
            return None, state, "Upload an image first."

        annotations = state.get("current_annotations", [])
        if not annotations:
            original = (
                np.array(state["original_pixels"], dtype=np.uint8) if "original_pixels" in state else None
            )
            return original, state, "Add at least one annotation or mark as good."

        # Check all annotations have descriptions
        undescribed = [a for a in annotations if not a.get("description")]
        if undescribed:
            original = (
                np.array(state["original_pixels"], dtype=np.uint8) if "original_pixels" in state else None
            )
            overlay = _build_overlay(original, annotations) if original is not None else None
            return (
                overlay,
                state,
                f"{len(undescribed)} annotation(s) missing descriptions. Describe each defect before saving.",
            )

        # Save to session
        session_images = state.get("session_images", [])
        session_images.append(
            {
                "path": state["current_image"],
                "annotations": annotations,
                "is_good": False,
            }
        )
        state["session_images"] = session_images
        state["current_annotations"] = []
        state.pop("current_image", None)
        state.pop("original_pixels", None)

        return None, state, _format_session_status(state)

    def on_classify_and_train(
        state: dict,
        epochs: int,
    ) -> str:
        """Classify annotations via LLM, then train adapter."""
        session_images = state.get("session_images", [])
        if not session_images:
            return "No annotated images. Upload and annotate images first."

        defect_count = sum(1 for img in session_images if not img["is_good"])
        good_count = sum(1 for img in session_images if img["is_good"])
        total_anns = sum(len(img.get("annotations", [])) for img in session_images if not img["is_good"])

        if defect_count == 0:
            return "No defect annotations found. Click on defects and describe them."

        # Build sample size guidance
        sample_note = ""
        total = defect_count + good_count
        if total < 5:
            sample_note = (
                f"\n⚠ Very small dataset ({total} images, {total_anns} "
                f"annotations). Results will be limited. "
                f"Recommend 10-15 images minimum.\n"
            )
        elif total < 15:
            sample_note = (
                f"\nNote: {total} images ({total_anns} annotations). More images will improve accuracy.\n"
            )

        # Convert state dicts to domain types
        from fsvlm.types import (
            AnnotatedImage,
            AnnotationSession,
            DefectAnnotation,
        )

        ann_images = []
        for img_data in session_images:
            annotations = []
            for a in img_data.get("annotations", []):
                annotations.append(
                    DefectAnnotation(
                        click_x=a["click_x"],
                        click_y=a["click_y"],
                        mask_rle=a["mask_rle"],
                        iou_score=a["iou_score"],
                        user_description=a.get("description", ""),
                        location_description=a.get("location", ""),
                    )
                )
            ann_images.append(
                AnnotatedImage(
                    image_path=Path(img_data["path"]),
                    annotations=annotations,
                    is_good=img_data.get("is_good", False),
                )
            )

        session = AnnotationSession(
            images=ann_images,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Step 1: Classify with LLM
        from fsvlm.agents.annotation_agent import AnnotationAgent

        agent = AnnotationAgent(config)
        try:
            session = agent.classify_annotations(session)
        except Exception as e:
            return f"Classification failed: {e}"

        taxonomy_text = "\n".join(f"  - {k}: {v}" for k, v in session.defect_taxonomy.items())

        # Step 2: Convert to training samples
        samples = agent.annotations_to_samples(session)

        if not samples:
            return "No training samples generated."

        # Step 3: Write CSV for the training pipeline
        tmpdir = Path(tempfile.mkdtemp(prefix="dvlm_annotated_"))
        csv_path = tmpdir / "labels.csv"
        lines = ["image_path,label,description"]
        for s in samples:
            # Escape commas in description
            desc = s.description.replace('"', '""')
            lines.append(f'{s.image_path},"{s.label}","{desc}"')
        csv_path.write_text("\n".join(lines))

        # Step 4: Train
        _progress["training"] = True
        _progress["text"] = "Starting training on annotated data..."

        from fsvlm.agents.orchestrator import Orchestrator
        from fsvlm.types import TrainingConfig

        try:
            tc = TrainingConfig(model_name=config.default_model)
            tc.num_train_epochs = epochs

            orch = Orchestrator(config, event_bus=event_bus)
            result, report = orch.train(
                csv_path,
                training_config=tc,
                sweep=False,
            )

            m = report.metrics
            return (
                f"Training complete!\n"
                f"{sample_note}\n"
                f"Defect taxonomy discovered:\n{taxonomy_text}\n\n"
                f"Training data: {len(samples)} samples "
                f"({total_anns} defect annotations across "
                f"{defect_count} images, {good_count} good images)\n\n"
                f"Results:\n"
                f"  Adapter: {result.adapter_path}\n"
                f"  AUROC: {m.auroc:.4f}\n"
                f"  F1: {m.f1:.4f}\n"
                f"  Precision: {m.precision:.4f}\n"
                f"  Recall: {m.recall:.4f}\n"
                f"  Time: {result.elapsed_seconds:.0f}s"
            )
        except Exception as e:
            return f"Training failed: {e}"
        finally:
            _progress["training"] = False

    # Colors for mask overlays (semi-transparent)
    _OVERLAY_COLORS = [
        (255, 50, 50),  # red
        (50, 180, 255),  # blue
        (50, 255, 50),  # green
        (255, 200, 50),  # yellow
        (200, 50, 255),  # purple
        (255, 128, 0),  # orange
    ]

    def _build_overlay(
        image: np.ndarray | None,
        annotations: list[dict],
    ) -> np.ndarray | None:
        """Render mask overlays with numbered labels onto image pixels."""
        if image is None:
            return None

        from fsvlm.models.sam import rle_to_mask

        result = image.copy()

        for i, ann in enumerate(annotations):
            if not ann.get("mask_rle"):
                continue

            mask = rle_to_mask(ann["mask_rle"])

            if mask.shape[:2] != result.shape[:2]:
                continue

            color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
            alpha = 0.4

            # Blend color onto masked region
            for c in range(3):
                result[:, :, c] = np.where(
                    mask,
                    (result[:, :, c] * (1 - alpha) + color[c] * alpha).astype(np.uint8),
                    result[:, :, c],
                )

            # Draw click point crosshair
            cx, cy = ann["click_x"], ann["click_y"]
            size = max(5, min(result.shape[0], result.shape[1]) // 50)
            y1, y2 = max(0, cy - size), min(result.shape[0], cy + size + 1)
            x1, x2 = max(0, cx - size), min(result.shape[1], cx + size + 1)
            result[max(0, cy - 1) : min(result.shape[0], cy + 2), x1:x2] = color
            result[y1:y2, max(0, cx - 1) : min(result.shape[1], cx + 2)] = color

            # Draw numbered label badge near click point
            _draw_number_badge(result, cx, cy, i + 1, color)

        return result

    def _draw_number_badge(
        image: np.ndarray,
        cx: int,
        cy: int,
        number: int,
        color: tuple,
    ) -> None:
        """Draw a numbered circle badge at the given position.

        Uses a simple pixel-based approach (no PIL/font dependency).
        """
        h, w = image.shape[:2]
        radius = max(10, min(h, w) // 40)

        # Position badge offset from click point (upper-right)
        bx = min(cx + radius + 4, w - radius - 2)
        by = max(cy - radius - 4, radius + 2)

        # Draw filled circle background (white with colored border)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist_sq = dx * dx + dy * dy
                px, py = bx + dx, by + dy
                if 0 <= px < w and 0 <= py < h:
                    if dist_sq <= radius * radius:
                        if dist_sq > (radius - 2) * (radius - 2):
                            image[py, px] = color  # border
                        else:
                            image[py, px] = (255, 255, 255)  # fill

        # Draw number using simple 5x7 pixel font
        _draw_digit(image, bx, by, number, color)

    # Simple 5x7 pixel font for digits 0-9
    _DIGIT_PATTERNS: dict[str, list[str]] = {
        "0": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        "2": ["01110", "10001", "00010", "00100", "01000", "10000", "11111"],
        "3": ["01110", "10001", "00001", "00110", "00001", "10001", "01110"],
        "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
        "6": ["01110", "10000", "11110", "10001", "10001", "10001", "01110"],
        "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
        "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
        "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    }

    def _draw_digit(
        image: np.ndarray,
        cx: int,
        cy: int,
        number: int,
        color: tuple,
    ) -> None:
        """Draw a number (1-99) centered at (cx, cy) using pixel font."""
        h, w = image.shape[:2]
        digits = str(number)
        char_w = 5
        char_h = 7
        gap = 1
        total_w = len(digits) * char_w + (len(digits) - 1) * gap
        start_x = cx - total_w // 2
        start_y = cy - char_h // 2

        for di, ch in enumerate(digits):
            pattern = _DIGIT_PATTERNS.get(ch, _DIGIT_PATTERNS["0"])
            ox = start_x + di * (char_w + gap)
            for row_i, row in enumerate(pattern):
                for col_i, pixel in enumerate(row):
                    if pixel == "1":
                        px = ox + col_i
                        py = start_y + row_i
                        if 0 <= px < w and 0 <= py < h:
                            image[py, px] = color

    def _format_annotation_status(state: dict) -> str:
        """Format status text for current image annotations."""
        annotations = state.get("current_annotations", [])
        session_images = state.get("session_images", [])

        lines = []
        if session_images:
            good = sum(1 for img in session_images if img["is_good"])
            defect = len(session_images) - good
            lines.append(f"Session: {len(session_images)} images saved ({defect} defect, {good} good)")
            lines.append("")

        if not annotations:
            lines.append("Click on a defect to segment it, or mark as 'Good'.")
        else:
            lines.append(f"Annotations on this image: {len(annotations)}")
            undescribed = []
            for i, ann in enumerate(annotations):
                desc = ann.get("description")
                loc = ann.get("location", "unknown")
                iou = ann.get("iou_score", 0)
                if desc:
                    lines.append(f"  #{i + 1} [{loc}] IoU {iou:.2f} — {desc}")
                else:
                    lines.append(f"  #{i + 1} [{loc}] IoU {iou:.2f} — ⚠ needs description")
                    undescribed.append(i + 1)

            lines.append("")
            if undescribed:
                nums = ", ".join(f"#{n}" for n in undescribed)
                lines.append(f"→ Describe annotation {nums}, then click 'Add Description'.")
                lines.append("  Tip: type '#2 scratch on edge' to describe a specific annotation.")
            else:
                lines.append("All annotations described. Click more defects, or 'Save & Next Image'.")

        return "\n".join(lines)

    def _format_session_status(state: dict) -> str:
        """Format status for the overall annotation session."""
        session_images = state.get("session_images", [])
        if not session_images:
            return "No images annotated yet. Upload an image to start."

        defect_count = sum(1 for img in session_images if not img["is_good"])
        good_count = sum(1 for img in session_images if img["is_good"])
        total_anns = sum(len(img.get("annotations", [])) for img in session_images if not img["is_good"])

        lines = [
            f"Image saved! Session: {len(session_images)} images "
            f"({defect_count} defect, {good_count} good), "
            f"{total_anns} defect annotations total.",
            "",
        ]

        # Minimum sample guidance
        if defect_count + good_count < 10:
            remaining = 10 - (defect_count + good_count)
            lines.append(
                f"Tip: Add {remaining} more images for better results. "
                f"Aim for 10-15 minimum (mix of good and defect)."
            )
        else:
            lines.append("Good dataset size. You can train now or add more.")

        lines.append("")
        lines.append("→ Upload the next image above, or click 'Classify & Train' when done.")

        return "\n".join(lines)

    # ---- Inspect Tab ----
    def inspect_image(
        image_path: str | None,
        adapter_dir: str,
        threshold: float,
    ) -> str:
        if not image_path:
            return "No image provided."
        if not adapter_dir:
            return "No adapter path provided."

        from fsvlm.agents.inspector_agent import InspectorSession

        path = Path(image_path)
        adapter = Path(adapter_dir)

        if not adapter.exists():
            return f"Adapter not found: {adapter}"

        try:
            with InspectorSession(config, adapter) as inspector:
                result = inspector.inspect(path, threshold=threshold)
        except Exception as e:
            return f"Error: {e}"

        status = "PASS" if result.pass_fail else "FAIL"
        return (
            f"Result: {status}\n"
            f"Confidence: {result.confidence:.1%}\n"
            f"Description: {result.description}\n"
            f"Time: {result.inference_time_ms:.0f}ms"
        )

    # ---- Train Tab ----
    def start_training(
        data_path: str,
        output_dir: str,
        epochs: int,
        no_sweep: bool,
    ) -> str:
        if not data_path:
            return "No data path provided."

        from fsvlm.agents.orchestrator import Orchestrator
        from fsvlm.types import TrainingConfig

        _progress["training"] = True
        _progress["text"] = "Starting training..."

        try:
            tc = TrainingConfig(model_name=config.default_model)
            tc.num_train_epochs = epochs

            orch = Orchestrator(config, event_bus=event_bus)
            out = Path(output_dir) if output_dir else None
            result, report = orch.train(
                Path(data_path),
                output_dir=out,
                training_config=tc,
                sweep=not no_sweep,
            )

            m = report.metrics
            return (
                f"Training complete!\n\n"
                f"Adapter: {result.adapter_path}\n"
                f"AUROC: {m.auroc:.4f}\n"
                f"F1: {m.f1:.4f}\n"
                f"Precision: {m.precision:.4f}\n"
                f"Recall: {m.recall:.4f}\n"
                f"Accuracy: {m.accuracy:.4f}\n"
                f"Time: {result.elapsed_seconds:.0f}s"
            )
        except Exception as e:
            return f"Training failed: {e}"
        finally:
            _progress["training"] = False

    def get_progress() -> str:
        return _progress.get("text", "")

    # ---- Validate Tab ----
    def run_validation(adapter_dir: str, data_path: str) -> str:
        if not adapter_dir or not data_path:
            return "Provide both adapter and data paths."

        from fsvlm.agents.data_agent import DataAgent
        from fsvlm.agents.orchestrator import Orchestrator

        try:
            data_agent = DataAgent(config)
            dataset = data_agent.prepare(Path(data_path), test_split=1.0)

            orch = Orchestrator(config)
            report = orch._validate_subprocess(Path(adapter_dir), dataset.val_samples, None)

            m = report.metrics
            cm = report.confusion_matrix.matrix
            cm_text = ""
            if len(cm) == 2:
                cm_text = (
                    f"\nConfusion Matrix:\n  TN={cm[0][0]}  FP={cm[0][1]}\n  FN={cm[1][0]}  TP={cm[1][1]}"
                )

            return (
                f"Validation Results ({report.num_test_samples} samples)\n\n"
                f"AUROC: {m.auroc:.4f}\n"
                f"F1: {m.f1:.4f}\n"
                f"Precision: {m.precision:.4f}\n"
                f"Recall: {m.recall:.4f}\n"
                f"Accuracy: {m.accuracy:.4f}\n"
                f"Threshold: {m.optimal_threshold:.4f}"
                f"{cm_text}"
            )
        except Exception as e:
            return f"Validation failed: {e}"

    # ---- Build UI ----
    with gr.Blocks(title="FSVLM") as app:
        gr.Markdown("# FSVLM\nVisual defect detection powered by Gemma 4 VLM + SAM segmentation")

        # ---- Annotate Tab (primary) ----
        with gr.Tab("Annotate", id="annotate"):
            gr.Markdown(
                "### Click on defects, describe them, train a detector\n\n"
                "1. **Upload** an image\n"
                "2. **Click** on each defect — SAM segments the region "
                "(multiple clicks per image supported)\n"
                "3. **Describe** each defect — type in the box and click "
                "'Add Description'\n"
                "   - Descriptions auto-target the first undescribed "
                "annotation\n"
                "   - Use `#2 scratch` to target a specific annotation\n"
                "4. **Save & Next Image** — repeat for more images, or "
                "'Mark as Good' for OK items\n"
                "5. **Classify & Train** when you have enough images "
                "(10-15 recommended)"
            )

            ann_state = gr.State(
                {
                    "session_images": [],
                    "current_annotations": [],
                }
            )

            with gr.Row():
                with gr.Column(scale=2):
                    # Single gr.Image for upload, display, AND click events
                    ann_canvas = gr.Image(
                        type="numpy",
                        label="Upload image, then click on defects to segment",
                        height=500,
                        interactive=True,
                    )

                with gr.Column(scale=1):
                    ann_status = gr.Textbox(
                        label="Status",
                        lines=8,
                        interactive=False,
                        value="Upload an image to start annotating.",
                    )
                    ann_desc_input = gr.Textbox(
                        label="What's wrong here?",
                        placeholder="e.g. 'crack in the surface', 'dark stain'",
                        lines=2,
                    )
                    ann_add_desc_btn = gr.Button("Add Description", variant="secondary")

                    with gr.Row():
                        ann_undo_btn = gr.Button("Undo Last Click")
                        ann_good_btn = gr.Button("Mark as Good")

                    ann_save_btn = gr.Button("Save & Next Image", variant="primary")

            gr.Markdown("---")

            with gr.Row():
                ann_epochs = gr.Slider(
                    1,
                    10,
                    value=3,
                    step=1,
                    label="Training Epochs",
                )
                ann_train_btn = gr.Button(
                    "Classify & Train",
                    variant="primary",
                    size="lg",
                )

            ann_train_output = gr.Textbox(
                label="Training Result",
                lines=15,
                interactive=False,
            )

            # Wire up events
            # Upload triggers state init
            ann_canvas.upload(
                on_image_upload,
                inputs=[ann_canvas, ann_state],
                outputs=[ann_canvas, ann_state, ann_status],
            )

            # Click on canvas triggers SAM segmentation
            ann_canvas.select(
                on_image_click,
                inputs=[ann_state],
                outputs=[ann_canvas, ann_state, ann_status],
            )

            ann_add_desc_btn.click(
                on_add_description,
                inputs=[ann_desc_input, ann_state],
                outputs=[ann_canvas, ann_state, ann_status, ann_desc_input],
            )

            ann_undo_btn.click(
                on_undo_last,
                inputs=[ann_state],
                outputs=[ann_canvas, ann_state, ann_status],
            )

            ann_good_btn.click(
                on_mark_good,
                inputs=[ann_state],
                outputs=[ann_canvas, ann_state, ann_status],
            )

            ann_save_btn.click(
                on_save_image,
                inputs=[ann_state],
                outputs=[ann_canvas, ann_state, ann_status],
            )

            ann_train_btn.click(
                on_classify_and_train,
                inputs=[ann_state, ann_epochs],
                outputs=ann_train_output,
            )

        # ---- Inspect Tab ----
        with gr.Tab("Inspect"):
            with gr.Row():
                with gr.Column():
                    inspect_img = gr.Image(type="filepath", label="Upload Image")
                    inspect_adapter = gr.Textbox(
                        label="Adapter Path",
                        placeholder="e.g. ~/.fsvlm/adapters/latest/adapter",
                    )
                    inspect_threshold = gr.Slider(0, 1, value=0.5, step=0.05, label="Threshold")
                    inspect_btn = gr.Button("Inspect", variant="primary")
                with gr.Column():
                    inspect_output = gr.Textbox(label="Result", lines=6)

            inspect_btn.click(
                inspect_image,
                inputs=[inspect_img, inspect_adapter, inspect_threshold],
                outputs=inspect_output,
            )

        # ---- Train Tab (file-based) ----
        with gr.Tab("Train"):
            with gr.Row():
                with gr.Column():
                    train_data = gr.Textbox(
                        label="Data Path",
                        placeholder="Path to image dir, CSV, or JSON",
                    )
                    train_output = gr.Textbox(
                        label="Output Directory (optional)",
                        placeholder="Leave blank for default",
                    )
                    train_epochs = gr.Slider(1, 30, value=3, step=1, label="Epochs")
                    train_no_sweep = gr.Checkbox(label="Skip sweep (single config)")
                    train_btn = gr.Button("Start Training", variant="primary")
                with gr.Column():
                    train_result = gr.Textbox(label="Training Result", lines=10)
                    train_progress = gr.Textbox(label="Progress", lines=2)
                    refresh_btn = gr.Button("Refresh Progress")

            train_btn.click(
                start_training,
                inputs=[train_data, train_output, train_epochs, train_no_sweep],
                outputs=train_result,
            )
            refresh_btn.click(get_progress, outputs=train_progress)

        # ---- Validate Tab ----
        with gr.Tab("Validate"):
            with gr.Row():
                with gr.Column():
                    val_adapter = gr.Textbox(
                        label="Adapter Path",
                        placeholder="Path to trained adapter",
                    )
                    val_data = gr.Textbox(
                        label="Test Data Path",
                        placeholder="Path to test image dir, CSV, or JSON",
                    )
                    val_btn = gr.Button("Validate", variant="primary")
                with gr.Column():
                    val_output = gr.Textbox(label="Validation Results", lines=12)

            val_btn.click(
                run_validation,
                inputs=[val_adapter, val_data],
                outputs=val_output,
            )

        # ---- Settings Tab ----
        with gr.Tab("Settings"):
            gr.Markdown(f"""
**Configuration**
- Base dir: `{config.base_dir}`
- Default model: `{config.default_model}`
- Sweep enabled: `{config.sweep_enabled}`
- Retrain threshold: `{config.correction_retrain_threshold} corrections`
- Server port: `{config.serve_port}`
""")

    return app


def launch(
    config: FSVLMConfig | None = None,
    port: int | None = None,
    share: bool = False,
) -> None:
    """Launch the Gradio interface.

    Args:
        config: FSVLM configuration.
        port: Port to serve on.
        share: Whether to create a public Gradio link.
    """
    if config is None:
        from fsvlm.config import load_config

        config = load_config()

    app = create_app(config)
    app.launch(
        server_port=port or config.ui_port,
        share=share,
    )
