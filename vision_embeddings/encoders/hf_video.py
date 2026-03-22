"""HuggingFace video-model encoder (V-JEPA 2) — encodes still images as
repeated-frame clips."""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

from ..config import EncoderConfig
from .base import BaseEncoder

logger = logging.getLogger(__name__)


def _image_to_video(image: Image.Image, frames: int) -> torch.Tensor:
    """PIL image -> ``T x C x H x W`` uint8 tensor (identical frames)."""
    arr = np.asarray(image, dtype=np.uint8)
    return (
        torch.from_numpy(arr)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .expand(frames, -1, -1, -1)
    )


class HFVideoEncoder(BaseEncoder):
    """V-JEPA 2 via HuggingFace — images are broadcast across frames."""

    def __init__(
        self,
        config: EncoderConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        compile_model: bool = True,
    ) -> None:
        from transformers import AutoModel, AutoVideoProcessor

        self.model_id = config.model_id
        self.embed_dim = config.embed_dim
        self.num_tokens = config.num_tokens
        self.device = device
        self._frames = config.frames_per_clip

        self.processor = AutoVideoProcessor.from_pretrained(config.model_id)
        self.model = AutoModel.from_pretrained(
            config.model_id, torch_dtype=dtype,
        ).to(device).eval()

        if compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                logger.warning("torch.compile unavailable, continuing without")

    @torch.no_grad()
    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor:
        videos = [_image_to_video(img, self._frames) for img in images]
        inputs = self.processor(videos, return_tensors="pt").to(self.device)

        if hasattr(self.model, "get_vision_features"):
            tokens = self.model.get_vision_features(**inputs)
        else:
            tokens = self.model(**inputs).last_hidden_state

        return tokens.to(torch.float16).cpu()
