"""HuggingFace vision-model encoder (SigLIP2, CLIP, DINOv2, DINOv3, ...)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image

from ..config import EncoderConfig
from .base import BaseEncoder

logger = logging.getLogger(__name__)

_VISION_CLASS_MAP: dict[str, str] = {
    "siglip2": "Siglip2VisionModel",
    "siglip": "SiglipVisionModel",
    "clip": "CLIPVisionModel",
    "dinov2": "Dinov2Model",
}


def _load_vision_model(model_id: str, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if model_type in _VISION_CLASS_MAP:
        import transformers
        cls = getattr(transformers, _VISION_CLASS_MAP[model_type])
        return cls.from_pretrained(model_id, torch_dtype=dtype)

    full = AutoModel.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True,
    )
    return full.vision_model if hasattr(full, "vision_model") else full


class HFVisionEncoder(BaseEncoder):
    """Wraps any HuggingFace ``*VisionModel`` behind :meth:`encode_batch`.

    Performance features:
    - Dedicated CUDA stream for async H2D transfers
    - Non-blocking device copies
    - Pinned CPU output buffer
    - Threaded image preprocessing
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

        self.processor = AutoImageProcessor.from_pretrained(config.model_id)
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
