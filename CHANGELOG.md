# Changelog

## [Unreleased] — work toward v0.2

### Added
- **Pass 4 — full Tier A coverage on Gemma 4 E4B-it** (240 rows total): all 24 MVTec + VisA
  categories at N ∈ {0, 2, 10, 30} with 3 seeds. Twice held-out, pre-registered. Headline:
  Spearman ρ = -0.778 between zero-shot AUROC and best fine-tune lift, p < 10⁻⁵.
  ([commits 4292c20, e7d7856, 93a9fa6, 13f0a62](https://github.com/ahmadhasan2k8/fsvlm/commits/main))
- **ICL extension** on 6 high/low-lift categories — confirms FT > ICL at N=2 on most cats but
  not all (chewinggum is FT ≈ ICL; transistor is ICL > FT).
- **Recipe stability sub-study** — rule survives rank ∈ {8, 16, 32} and lr ∈ {1e-4, 2e-4} on
  Gemma 4 (all variants ρ = -1.000 on 5-cat subset).
- **Multi-model phase** — rule tested on Qwen3-VL-8B-Instruct (passes ρ = -1.000 on 5 cats)
  and Llama-3.2-11B-Vision (does not transfer under same recipe; reveals a model-architecture
  boundary worth investigating). 2-of-3 model families confirm the rule.
  ([commit fbb2f96](https://github.com/ahmadhasan2k8/fsvlm/commit/fbb2f96))
- `--lora-rank` / `--lora-alpha` / `--learning-rate` CLI flags on `fsvlm train` and forwarded
  through `research/dataset_size_sweep.py`. ([commit edbefc7](https://github.com/ahmadhasan2k8/fsvlm/commit/edbefc7))

### Fixed
- **TRL 0.24 + transformers 5.5 + unsloth + Qwen/Llama compatibility**: `TrainingArguments.to_dict()`
  obfuscates token-suffixed fields by replacing them with `<{NAME_UPPER}>` placeholders, and
  some code paths in TRL's SFTTrainer round-trip args through that method, leading to a
  `'<EOS_TOKEN>' is not in vocabulary` error on Qwen and Llama vision models. fsvlm now
  installs a runtime monkey-patch on `TrainingArguments.to_dict` that reverses the obfuscation
  for `*_token` fields, plus a defensive override on `SFTTrainer.__init__` to clear any
  literal `<EOS_TOKEN>` placeholder. ([commit 376a4fb](https://github.com/ahmadhasan2k8/fsvlm/commit/376a4fb))
- `fsvlm.cli train` now correctly pulls `lora_rank`, `lora_alpha`, `learning_rate`, and
  `num_train_epochs` from `FSVLMConfig` rather than falling back to hardcoded `TrainingConfig`
  dataclass defaults. Earlier behavior silently ignored env-var / config-file overrides on
  these fields. ([commit edbefc7](https://github.com/ahmadhasan2k8/fsvlm/commit/edbefc7))
- `research/tiered_validation.py` `_run_base_model_inference` now reads the `FSVLM_DEFAULT_MODEL`
  env var instead of hardcoding `"unsloth/gemma-4-E4B-it"`. ([commit e4d7c89](https://github.com/ahmadhasan2k8/fsvlm/commit/e4d7c89))

## [0.1.0] - 2026-04-06

### Added
- Initial project scaffold with domain types, config, and CLI
- Data Agent: folder-based image reader with stratified split and minority oversampling
- Training Agent: QLoRA fine-tuning via Unsloth (Cython-compatible)
- Validation Agent: AUROC, F1, precision, recall with threshold optimization
- Inspector Agent: single-image inference with token probability scoring
- Orchestrator: sequential data → train → validate pipeline
- CLI commands: `setup`, `train`, `inspect`, `version`
- Hardware detection and model recommendation
- Adapter metadata save/load with schema versioning
