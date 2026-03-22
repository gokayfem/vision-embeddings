"""Optional NVIDIA DALI GPU-accelerated image preprocessing.

Decodes, resizes, and normalizes images entirely on GPU — bypasses
PIL and the CPU image processing bottleneck.

Requires: ``pip install nvidia-dali-cuda120`` (or your CUDA version).
Falls back gracefully if DALI is not installed.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

logger = logging.getLogger(__name__)

_DALI_AVAILABLE = False
try:
    from nvidia.dali import fn, pipeline_def, types  # type: ignore[import-untyped]
    _DALI_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    return _DALI_AVAILABLE


class DALIPreprocessor:
    """GPU-resident image decode + resize + normalize via NVIDIA DALI.

    Replaces the CPU-bound ``AutoImageProcessor`` for HF vision encoders.
    All work happens on GPU — the output is a CUDA tensor ready for the model.
    """

    def __init__(
        self,
        resize: int = 384,
        crop: int = 384,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
        device_id: int = 0,
    ) -> None:
        if not _DALI_AVAILABLE:
            raise ImportError(
                "NVIDIA DALI not installed. "
                "Install with: pip install nvidia-dali-cuda120"
            )
        self.resize = resize
        self.crop = crop
        self.mean = mean
        self.std = std
        self.device_id = device_id

    def __call__(
        self,
        images: list[Image.Image],
    ) -> torch.Tensor:
        """Preprocess PIL images on GPU.

        Returns ``[batch, 3, crop, crop]`` float32 CUDA tensor, normalized.
        """
        import numpy as np

        arrays = [np.asarray(img.convert("RGB"), dtype=np.uint8) for img in images]

        @pipeline_def(batch_size=len(arrays), num_threads=4, device_id=self.device_id)
        def _pipe():
            raw = fn.external_source(name="images", device="cpu")
            gpu = raw.gpu()
            resized = fn.resize(
                gpu,
                resize_shorter=self.resize,
                interp_type=types.INTERP_TRIANGULAR,
            )
            cropped = fn.crop(
                resized,
                crop=(self.crop, self.crop),
                crop_pos_x=0.5,
                crop_pos_y=0.5,
            )
            normalized = fn.crop_mirror_normalize(
                cropped,
                mean=[m * 255.0 for m in self.mean],
                std=[s * 255.0 for s in self.std],
                dtype=types.FLOAT,
                output_layout="CHW",
            )
            return normalized

        pipe = _pipe()
        pipe.build()
        pipe.feed_input("images", arrays)
        (out,) = pipe.run()

        return out.as_tensor()  # [batch, 3, crop, crop] on GPU
