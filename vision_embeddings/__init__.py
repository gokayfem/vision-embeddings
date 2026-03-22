"""vision_embeddings — extract, cache, and upload vision encoder embeddings.

>>> from vision_embeddings import create_encoder, process_dataset, get_dataset
>>> encoder = create_encoder("siglip2-so400m-384")
>>> process_dataset(encoder, get_dataset("textvqa"), ...)
"""

from .config import DatasetConfig, EncoderConfig
from .datasets import CAULDRON_SUBSETS, get_dataset, list_datasets, register_dataset
from .encoders import (
    BaseEncoder,
    HFVideoEncoder,
    HFVisionEncoder,
    TorchHubEncoder,
    create_encoder,
    get_encoder_config,
    list_encoders,
    register_encoder,
)
from .pipeline import process_dataset

__all__ = [
    "EncoderConfig",
    "DatasetConfig",
    "BaseEncoder",
    "HFVisionEncoder",
    "HFVideoEncoder",
    "TorchHubEncoder",
    "create_encoder",
    "register_encoder",
    "get_encoder_config",
    "list_encoders",
    "CAULDRON_SUBSETS",
    "register_dataset",
    "get_dataset",
    "list_datasets",
    "process_dataset",
]
