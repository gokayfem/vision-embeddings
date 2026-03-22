"""Abstract base class that every encoder must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from PIL import Image


class BaseEncoder(ABC):
    """Public contract for all vision / video encoders.

    Subclasses must set ``model_id``, ``embed_dim``, ``num_tokens``
    and implement :meth:`encode_batch`.

    For CPU-GPU overlap, subclasses should also override :meth:`preprocess`
    (CPU-only, thread-safe) and :meth:`encode_preprocessed` (GPU).
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

    def preprocess(self, images: list[Image.Image]) -> Any:
        """CPU-only preprocessing. Override for CPU-GPU overlap.

        Default returns images unchanged so :meth:`encode_preprocessed`
        falls back to :meth:`encode_batch`.
        """
        return images

    def encode_preprocessed(self, preprocessed: Any) -> torch.Tensor:
        """GPU encoding from preprocessed data. Override for CPU-GPU overlap.

        Default treats *preprocessed* as a list of PIL images.
        """
        return self.encode_batch(preprocessed)
