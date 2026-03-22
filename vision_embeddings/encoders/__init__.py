"""Encoder subpackage — factory, registration, and built-in configs."""

from __future__ import annotations

import torch

from ..config import EncoderConfig
from .base import BaseEncoder
from .defaults import DEFAULT_ENCODERS
from .hf_video import HFVideoEncoder
from .hf_vision import HFVisionEncoder
from .torch_hub import TorchHubEncoder

__all__ = [
    "BaseEncoder",
    "HFVisionEncoder",
    "HFVideoEncoder",
    "TorchHubEncoder",
    "create_encoder",
    "register_encoder",
    "get_encoder_config",
    "list_encoders",
]

_REGISTRY: dict[str, EncoderConfig] = {**DEFAULT_ENCODERS}

_LOADER_MAP: dict[str, type[BaseEncoder]] = {
    "hf_vision": HFVisionEncoder,
    "hf_video": HFVideoEncoder,
    "torch_hub": TorchHubEncoder,
}


def register_encoder(name: str, config: EncoderConfig) -> None:
    """Add a custom encoder to the global registry."""
    _REGISTRY[name] = config


def get_encoder_config(name: str) -> EncoderConfig:
    """Look up an encoder config by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown encoder '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_encoders() -> list[str]:
    """Return sorted list of all registered encoder names."""
    return sorted(_REGISTRY.keys())


def create_encoder(
    name: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    compile_model: bool = True,
) -> BaseEncoder:
    """Instantiate an encoder by registry name."""
    config = get_encoder_config(name)
    cls = _LOADER_MAP.get(config.loader)
    if cls is None:
        raise ValueError(
            f"No loader for '{config.loader}'. "
            f"Available: {list(_LOADER_MAP.keys())}"
        )
    return cls(config, device=device, dtype=dtype, compile_model=compile_model)
