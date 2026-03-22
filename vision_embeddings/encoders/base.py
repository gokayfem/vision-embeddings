"""Abstract base class that every encoder must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from PIL import Image


class BaseEncoder(ABC):
    """Public contract for all vision / video encoders.

    Subclasses must set ``model_id``, ``embed_dim``, ``num_tokens``
    and implement :meth:`encode_batch`.
    """

    model_id: str
    embed_dim: int
    num_tokens: int

    @abstractmethod
    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor:
        """Encode a list of PIL images.

        Returns
        -------
        torch.Tensor
            Shape ``[batch, seq_len, hidden_dim]``, dtype ``float16``, on CPU.
        """
        ...
