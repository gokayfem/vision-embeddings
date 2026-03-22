"""HuggingFace vision-model encoder (SigLIP2, CLIP, DINOv2, DINOv3, ...).

Inspired by production patterns from fal.ai's SigLIP2 endpoint:
- Explicit warmup to trigger CUDA/JIT/cudnn initialization before real work
- channels_last memory format for tensor-core efficiency
- Simple, direct preprocessing path
"""

from __future__ import annotations

import logging
import sys
import types

import torch
from PIL import Image

from ..config import EncoderConfig
from ..dali_preprocessor import DALIPreprocessor, is_available as _dali_available
from .base import BaseEncoder

logger = logging.getLogger(__name__)

# Model types where we load the full model and extract .vision_model
# to avoid "UNEXPECTED key" warnings from loading a vision-only class
# against a checkpoint that contains both vision + text weights.
_FULL_MODEL_TYPES: set[str] = {"clip", "siglip", "siglip2"}

# Pure vision models — no text tower, safe to load directly.
_VISION_ONLY_MAP: dict[str, str] = {
    "dinov2": "Dinov2Model",
}


def _ensure_flash_attn_importable() -> None:
    """Install a no-op ``flash_attn`` shim so custom model files
    (e.g. InternViT) can be imported even when flash_attn is absent.

    The model will fall back to eager / sdpa attention at runtime.
    """
    try:
        import flash_attn  # noqa: F401
        return  # real package available, nothing to do
    except ImportError:
        pass

    stub = types.ModuleType("flash_attn")
    stub.__path__ = []  # type: ignore[attr-defined]

    def _not_installed(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "flash_attn is not installed — model should use eager attention"
        )

    stub.flash_attn_func = _not_installed  # type: ignore[attr-defined]
    stub.flash_attn_varlen_func = _not_installed  # type: ignore[attr-defined]
    sys.modules["flash_attn"] = stub

    iface = types.ModuleType("flash_attn.flash_attn_interface")
    iface.flash_attn_func = _not_installed  # type: ignore[attr-defined]
    iface.flash_attn_varlen_func = _not_installed  # type: ignore[attr-defined]
    sys.modules["flash_attn.flash_attn_interface"] = iface

    bert = types.ModuleType("flash_attn.bert_padding")
    bert.pad_input = _not_installed  # type: ignore[attr-defined]
    bert.unpad_input = _not_installed  # type: ignore[attr-defined]
    bert.index_first_axis = _not_installed  # type: ignore[attr-defined]
    sys.modules["flash_attn.bert_padding"] = bert

    logger.info("Installed flash_attn stub — models will use eager attention")


def _load_vision_model(model_id: str, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    # Pure vision architectures — load directly, all keys match.
    if model_type in _VISION_ONLY_MAP:
        import transformers
        cls = getattr(transformers, _VISION_ONLY_MAP[model_type])
        return cls.from_pretrained(model_id, torch_dtype=dtype)

    # CLIP / SigLIP — load the *full* model (vision+text), then detach
    # the vision tower.  This way every key in the checkpoint is expected
    # and the "UNEXPECTED" report is silenced.
    if model_type in _FULL_MODEL_TYPES:
        full = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        return full.vision_model

    # Models with custom code (InternViT, DINOv3, etc.)
    # Ensure flash_attn is importable so custom modeling files don't crash.
    _ensure_flash_attn_importable()

    try:
        full = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=None,
            low_cpu_mem_usage=False,
        )
    except RuntimeError as exc:
        if "meta tensor" not in str(exc).lower():
            raise
        # Fallback: construct model from config on CPU, load state dict manually.
        logger.info("Meta-tensor error — falling back to manual weight loading")
        full = _load_custom_model_manual(model_id, dtype)

    return full.vision_model if hasattr(full, "vision_model") else full


def _load_custom_model_manual(model_id: str, dtype: torch.dtype) -> torch.nn.Module:
    """Last-resort loader for custom models that break with lazy init."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as load_safetensors
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Instantiate with real CPU tensors — no meta device.
    with torch.device("cpu"):
        model = AutoModel.from_config(
            config, trust_remote_code=True, torch_dtype=dtype,
        )

    # Load weights from safetensors (preferred) or pytorch bin.
    try:
        weight_path = hf_hub_download(model_id, "model.safetensors")
        state_dict = load_safetensors(weight_path)
    except Exception:
        weight_path = hf_hub_download(model_id, "pytorch_model.bin")
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

    model.load_state_dict(state_dict, strict=False)
    return model.to(dtype=dtype)


class HFVisionEncoder(BaseEncoder):
    """Wraps any HuggingFace ``*VisionModel`` behind :meth:`encode_batch`.

    Performance features:
    - Explicit warmup to trigger CUDA kernels, JIT, and cudnn autotuning
    - channels_last memory format for tensor-core efficiency
    - Pinned memory + non-blocking H2D transfers
    - Optional DALI GPU preprocessing
    - Optional torch.compile with max-autotune
    """

    def __init__(
        self,
        config: EncoderConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        compile_model: bool = True,
        use_dali: bool = False,
    ) -> None:
        from transformers import AutoImageProcessor

        self.model_id = config.model_id
        self.embed_dim = config.embed_dim
        self.num_tokens = config.num_tokens
        self.device = device
        self.dtype = dtype

        self.processor = AutoImageProcessor.from_pretrained(
            config.model_id, trust_remote_code=True,
        )
        self.model = _load_vision_model(config.model_id, dtype)
        self.model = self.model.to(device).to(memory_format=torch.channels_last).eval()

        # Optional DALI GPU preprocessing
        self._dali: DALIPreprocessor | None = None
        if use_dali and _dali_available():
            proc_mean = getattr(self.processor, "image_mean", (0.485, 0.456, 0.406))
            proc_std = getattr(self.processor, "image_std", (0.229, 0.224, 0.225))
            device_id = 0 if device == "cuda" else int(device.split(":")[-1])
            self._dali = DALIPreprocessor(
                resize=config.resolution,
                crop=config.resolution,
                mean=tuple(proc_mean),
                std=tuple(proc_std),
                device_id=device_id,
            )
            logger.info("DALI GPU preprocessing enabled (resolution=%d)", config.resolution)
        elif use_dali:
            logger.warning("DALI requested but not installed — using CPU preprocessing")

        if compile_model:
            try:
                logger.info("Compiling model (first batch will be slow)...")
                self.model = torch.compile(self.model, mode="max-autotune")
            except Exception:
                logger.warning("torch.compile unavailable, continuing without")

        # Warmup: trigger CUDA kernels, JIT compilation, cudnn autotuning
        self._warmup(config.resolution)

    def _warmup(self, resolution: int) -> None:
        """Run a dummy batch to trigger all lazy initialization."""
        logger.info("Warming up encoder...")
        dummy = [Image.new("RGB", (resolution, resolution))]
        with torch.inference_mode():
            self.encode_batch(dummy)
        if self.device != "cpu":
            torch.cuda.synchronize()
        logger.info("Warmup complete")

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """Images -> pixel_values tensor. Uses DALI (GPU) if available,
        otherwise falls back to HF AutoImageProcessor (CPU, pinned)."""
        if self._dali is not None:
            pv = self._dali(images)
            if not isinstance(pv, torch.Tensor):
                pv = torch.as_tensor(pv, device=self.device)
            return pv.to(memory_format=torch.channels_last)
        inputs = self.processor(images=images, return_tensors="pt")
        pv = inputs["pixel_values"].to(memory_format=torch.channels_last)
        if self.device != "cpu":
            pv = pv.pin_memory()
        return pv

    @torch.inference_mode()
    def encode_preprocessed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """GPU: preprocessed pixel_values -> embeddings on CPU."""
        pv = pixel_values.to(self.device, dtype=self.dtype, non_blocking=True)
        out = self.model(pixel_values=pv).last_hidden_state
        return out.to(torch.float16).cpu()

    @torch.inference_mode()
    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor:
        return self.encode_preprocessed(self.preprocess(images))
