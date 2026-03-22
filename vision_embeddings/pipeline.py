"""Pipelined extraction: dataset -> encoder -> shards -> HF Hub.

Three-stage pipeline inspired by pegainfer's kernel-launch amortization:
  Stage 1 (CPU threads) : preprocess next batch of images
  Stage 2 (GPU stream)  : encode current batch
  Stage 3 (BG thread)   : save + upload previous shard (batched commit)
All three stages overlap — the GPU never waits on I/O.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image
from tqdm.auto import tqdm

from .batch_upload import upload_config_and_readme, upload_shard_batched
from .config import DatasetConfig, EncoderConfig
from .encoders.base import BaseEncoder

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Background uploader — never block the GPU on network I/O
# ------------------------------------------------------------------

class _BackgroundUploader:
    """Saves + uploads shards in a background thread using batched commits."""

    def __init__(
        self,
        api: HfApi,
        repo_id: str,
        delete_local: bool = True,
    ) -> None:
        self._api = api
        self._repo_id = repo_id
        self._delete = delete_local
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="upload")
        self._pending: Future | None = None
        self._lock = Lock()

    def submit(
        self,
        tensors: dict[str, torch.Tensor],
        metadata: list[dict],
        shard_idx: int,
        output_dir: Path,
    ) -> None:
        """Submit a shard for background save+upload. Back-pressure: blocks
        only if the previous upload hasn't finished yet."""
        self.wait()
        self._pending = self._pool.submit(
            upload_shard_batched,
            self._api,
            self._repo_id,
            tensors,
            metadata,
            shard_idx,
            output_dir,
            self._delete,
        )

    def wait(self) -> None:
        with self._lock:
            if self._pending is not None:
                self._pending.result()
                self._pending = None

    def shutdown(self) -> None:
        self.wait()
        self._pool.shutdown(wait=True)


# ------------------------------------------------------------------
# Parallel image preprocessing
# ------------------------------------------------------------------

def _prepare_image(img: Image.Image) -> Image.Image | None:
    try:
        rgb = img.convert("RGB")
        return rgb if min(rgb.size) >= 10 else None
    except Exception:
        return None


def _prepare_images_parallel(
    images: list[Image.Image],
    pool: ThreadPoolExecutor,
) -> list[Image.Image | None]:
    return list(pool.map(_prepare_image, images))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_images(sample: dict, config: DatasetConfig) -> list[Image.Image]:
    if config.multi_image:
        raw = sample.get(config.image_column) or []
        return [img for img in raw if img is not None]
    img = sample.get(config.image_column)
    return [img] if img is not None else []


def _encode_safe(
    encoder: BaseEncoder,
    images: list[Image.Image],
    metadata: list[dict],
) -> tuple[torch.Tensor | None, list[dict]]:
    try:
        return encoder.encode_batch(images), metadata
    except Exception as exc:
        logger.warning("Batch failed (%d imgs): %s — per-image fallback", len(images), exc)
        embs, metas = [], []
        for img, meta in zip(images, metadata):
            try:
                embs.append(encoder.encode_batch([img]))
                metas.append(meta)
            except Exception:
                continue
        if embs:
            return torch.cat(embs, dim=0), metas
        return None, []


def _existing_shard_count(api: HfApi, repo_id: str) -> int:
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        indices = sorted(
            int(f.split("shard_")[1].split(".")[0])
            for f in files
            if f.startswith("shards/shard_") and f.endswith(".safetensors")
        )
        count = 0
        for i, idx in enumerate(indices):
            if idx != i:
                break
            count += 1
        return count
    except Exception:
        return 0


def _build_shard_tensors(
    emb: torch.Tensor,
    save_mode: str,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    if save_mode == "pooled":
        tensors["embeddings"] = emb.mean(dim=1) if emb.ndim == 3 else emb
    elif save_mode == "both":
        tensors["embeddings"] = emb
        tensors["pooled"] = emb.mean(dim=1) if emb.ndim == 3 else emb
    else:
        tensors["embeddings"] = emb
    return tensors


def _generate_readme(encoder_id: str, name: str, cfg: dict) -> str:
    seq, dim = cfg.get("seq_len", "?"), cfg.get("embed_dim", "?")
    mode = cfg.get("save_mode", "tokens")
    shape = f"[shard, {dim}]" if mode == "pooled" else f"[shard, {seq}, {dim}]"
    return (
        "---\nlicense: apache-2.0\ntags:\n- vision-embeddings\n"
        f"- {name}\n---\n\n"
        f"# Vision embeddings: {name}\n\n"
        f"Pre-computed embeddings via `{encoder_id}`.\n\n"
        f"| Field | Value |\n| --- | --- |\n"
        f"| Encoder | `{encoder_id}` |\n"
        f"| Dataset | `{cfg.get('dataset', '')}` |\n"
        f"| Subset | `{cfg.get('subset', 'N/A')}` |\n"
        f"| Images | {cfg.get('total_images', 0)} |\n"
        f"| Shards | {cfg.get('total_shards', 0)} |\n"
        f"| Save mode | `{mode}` |\n"
        f"| Shape | `{shape}` |\n\n"
        "```python\n"
        "from safetensors.torch import load_file\n\n"
        "shard = load_file(\"shards/shard_000000.safetensors\")\n"
        f"emb = shard[\"embeddings\"]  # {shape}\n"
        "```\n"
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def process_dataset(
    encoder: BaseEncoder,
    dataset_config: DatasetConfig,
    encoder_config: EncoderConfig,
    dataset_name: str,
    repo_id: str,
    output_dir: str = "/tmp/embedding_cache",
    shard_size: int = 1000,
    batch_size: int = 64,
    max_images: int = 0,
    save_mode: str = "tokens",
    hf_token: str | None = None,
    delete_local: bool = True,
    num_prep_workers: int = 4,
) -> None:
    """Stream a dataset, encode images, shard, and upload to HF Hub.

    Three-stage pipeline:
      1. CPU thread-pool preprocesses images in parallel
      2. GPU encodes via dedicated CUDA stream
      3. Background thread saves + uploads shards via batched commits

    Resume-safe: detects existing shards in *repo_id* and skips ahead.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    api = HfApi(token=hf_token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    existing = _existing_shard_count(api, repo_id)
    skip_count = existing * shard_size
    if existing:
        print(f"Resuming {dataset_name}: {existing} shards, skipping {skip_count} imgs")

    ds = load_dataset(
        dataset_config.hf_id, name=dataset_config.subset,
        split=dataset_config.split, streaming=True, trust_remote_code=True,
    )

    out = Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    uploader = _BackgroundUploader(api, repo_id, delete_local)
    prep_pool = ThreadPoolExecutor(
        max_workers=num_prep_workers, thread_name_prefix="prep",
    )

    shard_idx = existing
    img_buf: list[Image.Image] = []
    meta_buf: list[dict] = []
    shard_embs: list[torch.Tensor] = []
    shard_meta: list[dict] = []
    global_idx = 0
    t0 = time.time()
    pbar = tqdm(desc=f"{dataset_name} [shard {shard_idx}]", unit="img")

    try:
        for sample_idx, sample in enumerate(ds):
            try:
                raw_images = _extract_images(sample, dataset_config)
            except Exception:
                continue

            prepared = _prepare_images_parallel(raw_images, prep_pool)

            for img_idx, image in enumerate(prepared):
                if global_idx < skip_count:
                    global_idx += 1
                    continue
                if image is None:
                    global_idx += 1
                    continue

                img_buf.append(image)
                meta_buf.append({
                    "global_idx": global_idx,
                    "sample_idx": sample_idx,
                    "image_idx": img_idx,
                })
                global_idx += 1

                if len(img_buf) >= batch_size:
                    embs, metas = _encode_safe(encoder, img_buf, meta_buf)
                    ok = embs.shape[0] if embs is not None else 0
                    if embs is not None:
                        shard_embs.append(embs)
                        shard_meta.extend(metas)
                    pbar.update(ok)
                    for im in img_buf:
                        im.close()
                    img_buf, meta_buf = [], []

                    while shard_embs:
                        total = sum(e.shape[0] for e in shard_embs)
                        if total < shard_size:
                            break
                        all_emb = torch.cat(shard_embs, dim=0)
                        shard_tensors = _build_shard_tensors(
                            all_emb[:shard_size], save_mode,
                        )
                        uploader.submit(
                            shard_tensors,
                            shard_meta[:shard_size],
                            shard_idx,
                            out,
                        )
                        left = all_emb.shape[0] - shard_size
                        if left > 0:
                            shard_embs = [all_emb[shard_size:].clone()]
                            shard_meta = shard_meta[shard_size:]
                        else:
                            shard_embs, shard_meta = [], []
                        del all_emb
                        shard_idx += 1
                        elapsed = time.time() - t0
                        speed = (global_idx - skip_count) / max(elapsed, 1)
                        pbar.set_description(
                            f"{dataset_name} [shard {shard_idx}] {speed:.0f} img/s"
                        )

                if max_images and (global_idx - skip_count) >= max_images:
                    break
            if max_images and (global_idx - skip_count) >= max_images:
                break

        if img_buf:
            embs, metas = _encode_safe(encoder, img_buf, meta_buf)
            ok = embs.shape[0] if embs is not None else 0
            if embs is not None:
                shard_embs.append(embs)
                shard_meta.extend(metas)
            pbar.update(ok)
            for im in img_buf:
                im.close()

        if shard_embs:
            all_emb = torch.cat(shard_embs, dim=0)
            shard_tensors = _build_shard_tensors(all_emb, save_mode)
            uploader.submit(shard_tensors, shard_meta, shard_idx, out)
            shard_idx += 1
            del all_emb

    finally:
        uploader.shutdown()
        prep_pool.shutdown(wait=False)

    pbar.close()

    config_data = {
        "encoder": encoder.model_id,
        "dataset": dataset_config.hf_id,
        "subset": dataset_config.subset,
        "split": dataset_config.split,
        "total_images": global_idx,
        "total_shards": shard_idx,
        "shard_size": shard_size,
        "embed_dim": encoder_config.embed_dim,
        "seq_len": encoder_config.num_tokens,
        "save_mode": save_mode,
    }
    cfg_path = out / "config.json"
    cfg_path.write_text(json.dumps(config_data, indent=2))
    readme = out / "README.md"
    readme.write_text(_generate_readme(encoder.model_id, dataset_name, config_data))
    upload_config_and_readme(api, repo_id, cfg_path, readme, delete_local)

    elapsed = time.time() - t0
    print(f"Done: {dataset_name} -> {repo_id}  "
          f"({shard_idx} shards, {global_idx} imgs, {elapsed:.0f}s)")
