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
    arr = np.asarray(image, dtype=np.uint8)
    return (
        torch.from_numpy(arr)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .expand(frames, -1, -1, -1)
    )


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

        self.model = model.to(device).eval()

        self._hub_processor = torch.hub.load(
            config.hub_repo, "vjepa2_preprocessor",
            crop_size=config.resolution, **hub_kw,
        )

        if compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                logger.warning("torch.compile unavailable, continuing without")

    @torch.no_grad()
    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor:
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

        batch = torch.cat(clips, dim=0).to(self.device)
        return _extract_hidden(self.model(batch)).to(torch.float16).cpu()
