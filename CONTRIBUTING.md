# Contributing to fsvlm

fsvlm is an open-source research project exploring VLM data efficiency on industrial
anomaly detection. Contributions that improve the research — new dataset readers, more
rigorous benchmarks, honest failure-mode analysis, edge-deployment paths — are very welcome.

## Before you start

1. Read **[POSITIONING.md](POSITIONING.md)** first. It spells out what this project is, what it is
   not, what the benchmarks need to cover, and failure modes we explicitly want to avoid. Most
   discussion about scope, priorities, and "is this worth doing" is settled there.
2. Skim **[CLAUDE.md](CLAUDE.md)** for the architectural rules (SOLID layers, DI, ABCs, typing
   standards, testing pyramid). Non-negotiable.

## Quick start for contributors

```bash
git clone https://github.com/ahmadhasan2k8/fsvlm.git
cd fsvlm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,serve,watch,ui,annotate]"
pip install pytest pytest-cov hypothesis ruff mypy
pytest tests/unit -q
```

## Making a change

1. **One concern per PR.** New dataset reader → its own PR. New benchmark script → its own PR.
2. **Write tests.** Every file with logic gets unit tests in `tests/unit/`. See
   `tests/unit/test_visa_reader.py` for the pattern. CI enforces it.
3. **Type-hint everything.** `mypy --ignore-missing-imports fsvlm` should be clean.
4. **Format with ruff.** `ruff format .` then `ruff check .` before pushing.
5. **Preserve the append-only invariant** on `research/dataset_size_results.json`. Never
   rewrite historical rows — only append. See the Aether-style provenance described in
   `research/verdict.py`.
6. **Benchmark honesty.** If your change affects training or evaluation, re-run the affected
   sweep with ≥3 seeds and post the delta. No cherry-picked seeds.

## What we want

- **Dataset readers** for well-respected public industrial defect datasets (MVTec LOCO, Real-IAD,
  Severstal, etc.). Follow the `FolderLabelReader` / `VisAReader` / `DeepPCBReader` pattern.
- **Baseline comparisons** — wire in YOLOv11, Anomalib PatchCore, GPT-4o zero-shot, etc. so the
  benchmark table is complete per POSITIONING.md.
- **Edge-deployment paths** — ONNX / TensorRT / CoreML export scripts with reproducible latency
  numbers on Jetson / M-series / consumer GPUs.
- **Honest failure cases** — where fsvlm underperforms classical methods. Documented in
  `docs/benchmarks.md`. This is more valuable than polishing wins.
- **Bug fixes with reproducers.**

## What we do NOT want

Read POSITIONING.md's "What this project IS NOT" section. In particular:

- **No aerial, drone, wind, solar, or utility-infrastructure** examples, datasets, demos, or
  prompt templates. Out of project scope — hard line.
- **No medical imaging** datasets (regulated domain).
- **No framework bloat** (platform features, marketplaces, plugin systems beyond the current
  `LabelReader` / `ReportGenerator` ABCs) unless tied to a concrete research need.
- **No training-speed headlines.** Those are unsloth's wins, not ours — we credit them clearly.

## Reporting issues

Open a GitHub issue with:

- Exact `pip show fsvlm` version + OS + Python + GPU + VRAM
- The minimal reproducer (a few lines of code or the exact CLI command)
- Expected vs. actual behaviour
- Relevant log snippet from `~/.fsvlm/logs/`

## Code of conduct

Be kind. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License for contributors

By contributing code, documentation, benchmarks, or data to fsvlm, you agree your contribution
will be licensed under the project's [Apache License 2.0](LICENSE) — a permissive
open-source license that allows commercial use. Apache 2.0 includes an explicit patent
grant; if you can't distribute your contribution under Apache 2.0 (e.g. due to an
incompatible upstream license), please don't open the PR.

## Credit

Training kernels come from [unsloth](https://github.com/unslothai/unsloth). Please respect
their attribution requirements in any derivative work.

## Citation (academic norm, not a legal requirement)

If you publish work that uses fsvlm — or extends it into a new dataset reader, baseline, or
methodology — we'd appreciate a citation. See `CITATION.cff` for BibTeX. Apache 2.0 does
not legally mandate citation, but citation is how open research stays healthy.
