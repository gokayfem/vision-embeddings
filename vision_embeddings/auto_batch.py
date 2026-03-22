"""Auto batch-size tuning — find the largest batch that fits in VRAM.

Runs a binary search at startup with the actual model and image resolution.
Avoids OOM during the long extraction run.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

from .encoders.base import BaseEncoder

logger = logging.getLogger(__name__)

# absolute bounds
_MIN_BATCH = 1
_MAX_PROBE = 512


def find_optimal_batch_size(
    encoder: BaseEncoder,
    resolution: int = 384,
    max_batch: int = _MAX_PROBE,
    headroom: float = 0.9,
) -> int:
    """Binary-search for the largest batch size that fits in GPU memory.

    Parameters
    ----------
    encoder
        An instantiated encoder (model already on GPU).
    resolution
        Image resolution to probe with.
    max_batch
        Upper bound to search. Will be clamped by VRAM.
    headroom
        Fraction of free VRAM to target (0.9 = leave 10% headroom).

    Returns
    -------
    int
        Recommended batch size.
    """
    if not torch.cuda.is_available():
        return max_batch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dummy = Image.new("RGB", (resolution, resolution), color=(128, 128, 128))

    lo, hi = _MIN_BATCH, max_batch
    best = _MIN_BATCH

    while lo <= hi:
        mid = (lo + hi) // 2
        if _try_batch(encoder, dummy, mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # apply headroom — scale down slightly from the max that worked
    safe = max(1, int(best * headroom))
    logger.info("Auto batch size: probed max=%d, with %.0f%% headroom -> %d",
                best, headroom * 100, safe)
    return safe


def _try_batch(encoder: BaseEncoder, dummy: Image.Image, size: int) -> bool:
    """Try encoding a batch of *size* dummy images. Return True if it fits."""
    torch.cuda.empty_cache()
    try:
        batch = [dummy] * size
        encoder.encode_batch(batch)
        torch.cuda.synchronize()
        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        return False
    finally:
        torch.cuda.empty_cache()
