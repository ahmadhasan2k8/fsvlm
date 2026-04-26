"""Image loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage


def load_image(path: Path, max_size: int = 560) -> PILImage.Image:
    """Load and resize an image, keeping aspect ratio.

    Args:
        path: Path to the image file.
        max_size: Maximum dimension (longest edge) in pixels.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the image doesn't exist.
        ValueError: If the file can't be opened as an image.
    """
    from PIL import Image

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image {path}: {e}") from e

    # Resize if larger than max_size
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img


def validate_image(path: Path) -> bool:
    """Check if a file is a valid, openable image.

    Returns True if the image can be opened and decoded, False otherwise.
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
