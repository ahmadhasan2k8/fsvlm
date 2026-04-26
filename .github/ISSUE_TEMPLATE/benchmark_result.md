---
name: Benchmark result (contribution)
about: Report a new benchmark result you ran with fsvlm
title: "[Benchmark] "
labels: benchmark-result
---

## Why this issue type exists

fsvlm is a research tool that accumulates community-contributed benchmark numbers. If you ran a sweep on a dataset we don't currently cover, on a model we don't test, or on a category we haven't reported, we want to hear about it. With enough independent reproductions, the numbers in `docs/benchmarks.md` become trustworthy.

## Setup

- **fsvlm version / commit**: `fsvlm --version` + `git rev-parse HEAD` if built from source
- **Base model**: (e.g., `unsloth/gemma-4-E4B-it`)
- **Hardware**: (e.g., RTX 5080 Laptop, 16 GB VRAM)
- **OS + Python**:
- **Recipe version** used in the run: (find this in the results JSON's `recipe_version` field)

## Dataset & Category

- **Dataset**: MVTec AD / VisA / DeepPCB / other (cite source)
- **Category**: (e.g., `hazelnut`, `candle`, or a new category you're adding)
- **Split**: held-out / custom / please describe

## Protocol

- **N-shot**: (e.g., N=2, 30, or a range)
- **Seeds**: (e.g., {42, 1337, 7})
- **Label source**: thin / metadata / agent / custom
- **Sampling**: uniform / stratified / custom

## Results

Please paste the relevant rows from your `research/dataset_size_results.json` OR provide a summary table:

| N | Seed | AUROC | F1 | P | R |
|---|-----:|------:|---:|--:|--:|
| 2 | 42 | 0.xx | 0.xx | 0.xx | 0.xx |
| ... | | | | | |

## Notes & anomalies

Anything surprising? Did the extractor do something unusual? Did training diverge? Is there a failure mode worth documenting in the honest-failures table?

## Attachments (optional)

- Link to your full `dataset_size_results.json` fragment, or a gist / raw file URL
- Screenshots of the training log if relevant
