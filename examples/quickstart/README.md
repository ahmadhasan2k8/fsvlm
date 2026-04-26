# Quickstart — fsvlm in 30 seconds

Generate a tiny synthetic dataset, validate fsvlm reads it correctly, and inspect the
outputs — **all without a GPU and without downloading any model**. Confirms your install is
healthy before you commit to a real training run.

## Run it

After `pip install -e .` (or `pip install fsvlm[…]` once the package is on PyPI):

```bash
# from the fsvlm repo root
python examples/quickstart/make_dataset.py
python examples/quickstart/check_pipeline.py
```

Expected output:

```
make_dataset: wrote 20 images to /tmp/fsvlm-quickstart/
  good/      → 10 PNGs
  defect/    → 10 PNGs
check_pipeline:
  FolderLabelReader: 20 samples (10 good / 10 defect)
  DataAgent: train=18, val=2
  Image loader: (224, 224) RGB
  Inspection prompt: 151 chars, contains PASS/FAIL
PASS — pipeline is healthy
```

(The `(224, 224)` is the synthetic image size; `load_image`'s `max_size=560` only down-samples
larger images. Train/val split is stratified to keep one defect example in val.)

If those four checks pass, your install is good and you can move on to training on real
data — see the main README's "Three-command quickstart" or
[`docs/skills.md`](../../docs/skills.md) for the `0 → paper` path.

## What this does NOT do

- No model download (so no HuggingFace Hub round-trip)
- No GPU required
- No actual training (training requires a CUDA GPU; this verifies the data path)
- No real defects — the synthetic images are coloured rectangles for the data-path test

For real training, point `fsvlm train --images <your-folder>/` at a folder with `good/` and
`defect/` subdirectories of actual labeled images.

## Adapting to your own data

The synthetic generator (`make_dataset.py`) is intentionally minimal. To run the same
pipeline check against *your* dataset, just point the second script at your folder:

```bash
python examples/quickstart/check_pipeline.py /path/to/your/data/
```

Your folder should have `good/` and `defect/` subdirectories of images.
