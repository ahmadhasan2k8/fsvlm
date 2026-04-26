# Changelog

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
