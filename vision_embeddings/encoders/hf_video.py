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
    arr = np.array(image, dtype=np.uint8)  # np.array (not asarray) → writable copy
    return (
        torch.from_numpy(arr)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .expand(frames, -1, -1, -1)
        .contiguous()
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

        self._stream = (
            torch.cuda.Stream(device=device)
            if device != "cpu" else None
        )

        if compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                logger.warning("torch.compile unavailable, continuing without")

    def preprocess(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        """CPU-only: images -> processor output dict (pinned)."""
        videos = [_image_to_video(img, self._frames) for img in images]
        inputs = self.processor(videos, return_tensors="pt")
        if self.device != "cpu":
            for k in inputs:
                if torch.is_tensor(inputs[k]):
                    inputs[k] = inputs[k].pin_memory()
        return inputs

    @torch.no_grad()
    def encode_preprocessed(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """GPU: preprocessed inputs -> embeddings on CPU."""
        stream = self._stream
        if stream is not None:
            with torch.cuda.stream(stream):
                dev_inputs = {
                    k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in inputs.items()
                }
                if hasattr(self.model, "get_vision_features"):
                    tokens = self.model.get_vision_features(**dev_inputs)
                else:
                    tokens = self.model(**dev_inputs).last_hidden_state
            stream.synchronize()
        else:
            dev_inputs = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }
            if hasattr(self.model, "get_vision_features"):
                tokens = self.model.get_vision_features(**dev_inputs)
            else:
                tokens = self.model(**dev_inputs).last_hidden_state
        return tokens.to(torch.float16).cpu()

    @torch.no_grad()
    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor:
        return self.encode_preprocessed(self.preprocess(images))
