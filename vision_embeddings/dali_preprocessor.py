"""Optional NVIDIA DALI GPU-accelerated image preprocessing.

Decodes, resizes, and normalizes images entirely on GPU — bypasses
PIL and the CPU image processing bottleneck.

Requires: ``pip install nvidia-dali-cuda120`` (or your CUDA version).
Falls back gracefully if DALI is not installed.
"""

from __future__ import annotations

import logging

import numpy as np
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

    The DALI pipeline is built once at a fixed max batch size and reused
    across calls. Smaller batches are padded with dummy images and the
    output is sliced.
    """

    def __init__(
        self,
        resize: int = 384,
        crop: int = 384,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
        max_batch_size: int = 64,
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
        self._max_batch_size = max_batch_size

        # Build the pipeline once with the max batch size
        self._pipe = self._build_pipeline(max_batch_size)
        # Dummy image for padding short batches
        self._dummy = np.zeros((10, 10, 3), dtype=np.uint8)

    def _build_pipeline(self, batch_size: int):
        resize = self.resize
        crop = self.crop
        mean = self.mean
        std = self.std
        device_id = self.device_id

        @pipeline_def(batch_size=batch_size, num_threads=4, device_id=device_id)
        def _pipe():
            raw = fn.external_source(name="images", device="cpu")
            gpu = raw.gpu()
            resized = fn.resize(
                gpu,
                resize_shorter=resize,
                interp_type=types.INTERP_TRIANGULAR,
            )
            cropped = fn.crop(
                resized,
                crop=(crop, crop),
                crop_pos_x=0.5,
                crop_pos_y=0.5,
            )
            normalized = fn.crop_mirror_normalize(
                cropped,
                mean=[m * 255.0 for m in mean],
                std=[s * 255.0 for s in std],
                dtype=types.FLOAT,
                output_layout="CHW",
            )
            return normalized

        pipe = _pipe()
        pipe.build()
        return pipe

    def __call__(
        self,
        images: list[Image.Image],
    ) -> torch.Tensor:
        """Preprocess PIL images on GPU.

        Returns ``[batch, 3, crop, crop]`` float32 CUDA tensor, normalized.
        """
        real_count = len(images)
        arrays = [np.asarray(img.convert("RGB"), dtype=np.uint8) for img in images]

        # Rebuild pipeline if batch exceeds max (rare)
        if real_count > self._max_batch_size:
            logger.info("DALI: rebuilding pipeline for batch_size=%d", real_count)
            self._pipe = self._build_pipeline(real_count)
            self._max_batch_size = real_count

        # Pad to max_batch_size with dummy images
        while len(arrays) < self._max_batch_size:
            arrays.append(self._dummy)

        self._pipe.feed_input("images", arrays)
        (out,) = self._pipe.run()

        tensor = out.as_tensor()  # [max_batch_size, 3, crop, crop] on GPU
        return tensor[:real_count]  # slice to actual batch size
