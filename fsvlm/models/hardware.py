"""GPU hardware detection and model recommendation."""

from __future__ import annotations

from fsvlm.types import GPUInfo, ModelInfo

# Known model catalog — extend as new models are added
KNOWN_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="gemma-4-E2B-it",
        hf_repo="unsloth/gemma-4-E2B-it",
        size_label="small",
        params_billions=2.0,
        vram_required_gb=4.0,
    ),
    ModelInfo(
        name="gemma-4-E4B-it",
        hf_repo="unsloth/gemma-4-E4B-it",
        size_label="medium",
        params_billions=4.0,
        vram_required_gb=8.0,
    ),
    ModelInfo(
        name="gemma-4-12B-it",
        hf_repo="unsloth/gemma-4-12B-it",
        size_label="large",
        params_billions=12.0,
        vram_required_gb=16.0,
    ),
]


def detect_gpu() -> GPUInfo:
    """Detect NVIDIA GPU and return hardware info.

    Returns GPUInfo with is_available=False if no CUDA GPU is found.
    """
    try:
        import torch
    except ImportError:
        return GPUInfo(
            name="unknown",
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            cuda_version="",
            is_available=False,
        )

    if not torch.cuda.is_available():
        return GPUInfo(
            name="none",
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            cuda_version="",
            is_available=False,
        )

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)

    return GPUInfo(
        name=props.name,
        vram_total_gb=total / (1024**3),
        vram_free_gb=free / (1024**3),
        cuda_version=torch.version.cuda or "",
        compute_capability=(props.major, props.minor),
        is_available=True,
    )


def recommend_model(gpu: GPUInfo) -> ModelInfo:
    """Recommend the best model for the detected GPU.

    Picks the largest model that fits in available VRAM.
    """
    if not gpu.is_available:
        # CPU-only: smallest model
        return KNOWN_MODELS[0]

    best = KNOWN_MODELS[0]
    for model in KNOWN_MODELS:
        if model.vram_required_gb <= gpu.vram_free_gb:
            best = model
    return best


def get_model_by_name(name: str) -> ModelInfo | None:
    """Look up a model by name or HF repo."""
    for model in KNOWN_MODELS:
        if name in (model.name, model.hf_repo):
            return model
    return None
