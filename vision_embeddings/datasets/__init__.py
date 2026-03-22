"""Dataset subpackage — registration and built-in configs."""

from __future__ import annotations

from ..config import DatasetConfig
from .defaults import CAULDRON_SUBSETS, DEFAULT_DATASETS

__all__ = [
    "CAULDRON_SUBSETS",
    "register_dataset",
    "get_dataset",
    "list_datasets",
]

_REGISTRY: dict[str, DatasetConfig] = {**DEFAULT_DATASETS}


def register_dataset(name: str, config: DatasetConfig) -> None:
    """Add a custom dataset to the global registry."""
    _REGISTRY[name] = config


def get_dataset(name: str) -> DatasetConfig:
    """Look up a dataset config by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_datasets() -> list[str]:
    """Return sorted list of all registered dataset names."""
    return sorted(_REGISTRY.keys())
