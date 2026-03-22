"""Shared configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderConfig:
    """Immutable description of a vision / video encoder."""

    model_id: str
    embed_dim: int
    num_tokens: int
    resolution: int
    loader: str = "hf_vision"          # hf_vision | hf_video | torch_hub
    frames_per_clip: int = 1
    hub_repo: str = ""
    hub_name: str = ""
    ckpt_url: str = ""
    encoder_key: str = ""


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable description of an image dataset on HuggingFace."""

    hf_id: str
    image_column: str
    split: str
    subset: str | None = None
    multi_image: bool = False
