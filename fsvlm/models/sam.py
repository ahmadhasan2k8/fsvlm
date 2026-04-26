"""SAM (Segment Anything Model) wrapper for interactive defect annotation.

Lazily loads SAM2.1-hiera-tiny (~117MB) for point-prompt segmentation.
User clicks on a defect → SAM returns a precise mask of that region.

Usage:
    sam = SAMSegmenter()
    sam.load()
    mask, score = sam.segment_at_point(image, x=450, y=300)
    sam.unload()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

DEFAULT_SAM_MODEL = "facebook/sam2.1-hiera-tiny"


@dataclass
class SegmentationResult:
    """Result of a single point-prompt segmentation."""

    mask: Any  # numpy ndarray (H, W) boolean mask
    iou_score: float
    mask_rle: str  # run-length encoded for compact storage


class SAMSegmenter:
    """Lazy-loading SAM wrapper for point-prompt segmentation.

    Args:
        model_name: HuggingFace model ID for SAM variant.
        device: Device to run on ("cuda", "cpu", or None for auto-detect).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SAM_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._model: Any = None
        self._processor: Any = None

    @property
    def is_loaded(self) -> bool:
        """Whether SAM model is currently loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load SAM model and processor into memory."""
        if self.is_loaded:
            return

        import torch
        from transformers import Sam2Model, Sam2Processor

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading SAM model: {} on {}", self.model_name, self._device)
        self._processor = Sam2Processor.from_pretrained(self.model_name)
        self._model = Sam2Model.from_pretrained(self.model_name).to(self._device)
        self._model.eval()
        logger.info("SAM model loaded ({:.0f}MB)", self._get_model_size_mb())

    def unload(self) -> None:
        """Free SAM model from memory."""
        if not self.is_loaded:
            return

        import gc

        del self._model
        del self._processor
        self._model = None
        self._processor = None
        gc.collect()

        if self._device and self._device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

        logger.info("SAM model unloaded")

    def segment_at_point(
        self,
        image: Any,
        x: int,
        y: int,
    ) -> SegmentationResult:
        """Segment the region containing the clicked point.

        Args:
            image: PIL Image or numpy array.
            x: Click x coordinate (pixels in original image).
            y: Click y coordinate (pixels in original image).

        Returns:
            SegmentationResult with boolean mask, confidence score, and RLE.
        """
        if not self.is_loaded:
            self.load()

        import numpy as np
        import torch
        from PIL import Image as PILImage

        # Ensure PIL Image
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)

        # Point prompt: [[[[x, y]]]] format (image, object, point, coords)
        # Labels: [[[1]]] = foreground (image, object, point)
        input_points = [[[[x, y]]]]
        input_labels = [[[1]]]

        inputs = self._processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process masks back to original image size
        # SAM2 pred_masks: (batch, num_objects, num_masks, H, W) at 256x256
        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
        )

        # Take best mask (highest IoU score)
        # iou_scores: (batch, num_objects, num_masks)
        iou_scores = outputs.iou_scores.cpu().squeeze()
        if iou_scores.dim() == 0:
            best_idx = 0
            best_iou = float(iou_scores)
        else:
            best_idx = int(iou_scores.argmax())
            best_iou = float(iou_scores[best_idx])

        # Extract boolean mask (H, W)
        # masks[0] is first batch, squeeze object dim, select best mask
        mask_tensor = masks[0].squeeze(0)  # (num_masks, H, W)
        mask = mask_tensor[best_idx].numpy().astype(bool)

        # Run-length encode for compact storage
        rle = _mask_to_rle(mask)

        return SegmentationResult(
            mask=mask,
            iou_score=best_iou,
            mask_rle=rle,
        )

    def _get_model_size_mb(self) -> float:
        """Estimate model size in MB."""
        if self._model is None:
            return 0.0
        total = sum(p.numel() * p.element_size() for p in self._model.parameters())
        return total / (1024 * 1024)


def _mask_to_rle(mask: Any) -> str:
    """Run-length encode a boolean mask to compact string.

    Args:
        mask: 2D boolean numpy array.

    Returns:
        RLE string: "height,width:start1 length1 start2 length2 ..."
    """
    import numpy as np

    h, w = mask.shape
    flat = mask.flatten()
    # Find runs of True values
    changes = np.diff(flat.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # Handle edge cases
    if flat[0]:
        starts = np.concatenate([[0], starts])
    if flat[-1]:
        ends = np.concatenate([ends, [len(flat)]])

    runs = []
    for s, e in zip(starts, ends):
        runs.append(f"{s} {e - s}")

    return f"{h},{w}:" + " ".join(runs)


def rle_to_mask(rle: str) -> Any:
    """Decode RLE string back to boolean mask.

    Args:
        rle: RLE string from _mask_to_rle.

    Returns:
        2D boolean numpy array.
    """
    import numpy as np

    header, runs_str = rle.split(":")
    h, w = [int(x) for x in header.split(",")]

    mask = np.zeros(h * w, dtype=bool)

    if runs_str.strip():
        parts = runs_str.strip().split()
        for i in range(0, len(parts), 2):
            start = int(parts[i])
            length = int(parts[i + 1])
            mask[start : start + length] = True

    return mask.reshape(h, w)


def mask_to_location(mask: Any) -> str:
    """Describe mask location in natural language.

    Args:
        mask: 2D boolean numpy array.

    Returns:
        Location string like "upper-left region" or "center".
    """
    import numpy as np

    h, w = mask.shape
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return "unknown"

    # Center of mass
    cx = float(xs.mean()) / w
    cy = float(ys.mean()) / h

    # Vertical position
    if cy < 0.33:
        v = "upper"
    elif cy > 0.66:
        v = "lower"
    else:
        v = "center"

    # Horizontal position
    if cx < 0.33:
        h_pos = "left"
    elif cx > 0.66:
        h_pos = "right"
    else:
        h_pos = "center"

    if v == "center" and h_pos == "center":
        return "center region"
    if v == "center":
        return f"{h_pos} region"
    if h_pos == "center":
        return f"{v} region"
    return f"{v}-{h_pos} region"
