"""HuggingFace vision-model encoder (SigLIP2, CLIP, DINOv2, DINOv3, ...)."""

from __future__ import annotations

import logging
import sys
import types

import torch
from PIL import Image

from ..config import EncoderConfig
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
    # The model's own code detects flash_attn is missing and falls back to
    # standard attention — we do NOT pass attn_implementation here because
    # custom model code often doesn't handle that kwarg.
    _ensure_flash_attn_importable()

    full = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=False,
    )

    return full.vision_model if hasattr(full, "vision_model") else full


class HFVisionEncoder(BaseEncoder):
    """Wraps any HuggingFace ``*VisionModel`` behind :meth:`encode_batch`.

    Performance features:
    - Dedicated CUDA stream for async H2D transfers
    - Non-blocking device copies
    - Pinned CPU output buffer
    """

    def __init__(
        self,
        config: EncoderConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        compile_model: bool = True,
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
        self.model = self.model.to(device).eval()

        # dedicated stream for H2D overlap
        self._stream = (
            torch.cuda.Stream(device=device)
            if device != "cpu" else None
        )

        if compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                logger.warning("torch.compile unavailable, continuing without")

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """CPU-only: images -> pixel_values tensor (pinned memory)."""
        inputs = self.processor(images=images, return_tensors="pt")
        pv = inputs["pixel_values"]
        if self.device != "cpu":
            pv = pv.pin_memory()
        return pv

    @torch.no_grad()
    def encode_preprocessed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """GPU: preprocessed pixel_values -> embeddings on CPU."""
        stream = self._stream
        if stream is not None:
            with torch.cuda.stream(stream):
                pv = pixel_values.to(self.device, dtype=self.dtype, non_blocking=True)
                out = self.model(pixel_values=pv).last_hidden_state
            stream.synchronize()
        else:
            pv = pixel_values.to(self.device, dtype=self.dtype)
            out = self.model(pixel_values=pv).last_hidden_state
        return out.to(torch.float16).cpu()

    @torch.no_grad()
    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor:
        return self.encode_preprocessed(self.preprocess(images))
