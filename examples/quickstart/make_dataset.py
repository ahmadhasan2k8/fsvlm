"""Generate a tiny synthetic dataset for the quickstart pipeline check.

Writes 20 images (10 good, 10 defect) to /tmp/fsvlm-quickstart/. No defect realism —
this exists to verify the data path works, not to train a real model.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    root = Path("/tmp/fsvlm-quickstart")
    (root / "good").mkdir(parents=True, exist_ok=True)
    (root / "defect").mkdir(parents=True, exist_ok=True)

    for i in range(10):
        # "Good" — uniform grey square
        img = Image.new("RGB", (224, 224), (180, 180, 180))
        img.save(root / "good" / f"good_{i:02d}.png")

        # "Defect" — same grey square with a red mark in a random-ish corner
        img = Image.new("RGB", (224, 224), (180, 180, 180))
        d = ImageDraw.Draw(img)
        x, y = 30 + (i * 19) % 150, 30 + (i * 23) % 150
        d.rectangle([x, y, x + 30, y + 30], fill=(220, 20, 20))
        img.save(root / "defect" / f"defect_{i:02d}.png")

    print(f"make_dataset: wrote 20 images to {root}/")
    print("  good/      → 10 PNGs")
    print("  defect/    → 10 PNGs")


if __name__ == "__main__":
    main()
