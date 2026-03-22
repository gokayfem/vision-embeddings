"""V-JEPA 2.1 encoder loaded via ``torch.hub``."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..config import EncoderConfig
from .base import BaseEncoder

logger = logging.getLogger(__name__)


def _clean_state_dict(
    sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in sd.items()
    }


def _extract_hidden(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], torch.Tensor):
        return x[0]
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    raise TypeError(f"Cannot extract hidden tensor from {type(x)}")


def _image_to_video(image: Image.Image, frames: int) -> torch.Tensor:
    """Expand single frame to video via stride-0 view (no memory duplication)."""
    arr = np.array(image, dtype=np.uint8)
    frame = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return frame.expand(frames, -1, -1, -1)


class TorchHubEncoder(BaseEncoder):
    """V-JEPA 2.1 — architecture from ``torch.hub``, weights from URL."""

    def __init__(
        self,
        config: EncoderConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        compile_model: bool = True,
    ) -> None:
        self.model_id = config.model_id
        self.embed_dim = config.embed_dim
        self.num_tokens = config.num_tokens
        self.device = device
        self._frames = config.frames_per_clip

        hub_kw: dict[str, Any] = dict(
            source="github", pretrained=False,
            trust_repo=True, skip_validation=True,
        )

        model = torch.hub.load(config.hub_repo, config.hub_name, **hub_kw)
        if isinstance(model, tuple):
            model = model[0]

        state = torch.hub.load_state_dict_from_url(
            config.ckpt_url, map_location="cpu",
        )
        enc_sd = _clean_state_dict(state[config.encoder_key])
        msg = model.load_state_dict(enc_sd, strict=False)
        if msg.unexpected_keys:
            logger.info("torch_hub: unexpected keys %s", msg.unexpected_keys)

        self.model = model.to(device).to(memory_format=torch.channels_last).eval()

        self._hub_processor = torch.hub.load(
            config.hub_repo, "vjepa2_preprocessor",
            crop_size=config.resolution, **hub_kw,
        )

        if compile_model:
            try:
                logger.info("Compiling model (first batch will be slow)...")
                self.model = torch.compile(self.model, mode="max-autotune")
            except Exception:
                logger.warning("torch.compile unavailable, continuing without")

        # Warmup
        self._warmup(config.resolution)

    def _warmup(self, resolution: int) -> None:
        logger.info("Warming up torch_hub encoder...")
        dummy = [Image.new("RGB", (resolution, resolution))]
        with torch.inference_mode():
            self.encode_batch(dummy)
        if self.device != "cpu":
            torch.cuda.synchronize()
        logger.info("Warmup complete")

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """CPU-only: images -> batched preprocessed tensor (pinned)."""
        clips = []
        for img in images:
            video = _image_to_video(img, self._frames)
            processed = self._hub_processor(video)
            if isinstance(processed, (list, tuple)):
                processed = processed[0]
            if not isinstance(processed, torch.Tensor):
                processed = torch.as_tensor(processed)
            if processed.ndim == 4:
                processed = processed.unsqueeze(0)
            clips.append(processed)
        batch = torch.cat(clips, dim=0)
        if self.device != "cpu":
            batch = batch.pin_memory()
        return batch

    @torch.inference_mode()
    def encode_preprocessed(self, batch: torch.Tensor) -> torch.Tensor:
        """GPU: preprocessed batch tensor -> embeddings on CPU."""
        batch = batch.to(self.device, non_blocking=True)
        return _extract_hidden(self.model(batch)).to(torch.float16).cpu()

    @torch.inference_mode()
    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor:
        return self.encode_preprocessed(self.preprocess(images))
