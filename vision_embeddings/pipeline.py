"""Streaming extraction pipeline: dataset -> encoder -> shards -> HF Hub."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image
from safetensors.torch import save_file
from tqdm.auto import tqdm

from .config import DatasetConfig, EncoderConfig
from .encoders.base import BaseEncoder

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _extract_images(sample: dict, config: DatasetConfig) -> list[Image.Image]:
    if config.multi_image:
        raw = sample.get(config.image_column) or []
        return [img for img in raw if img is not None]
    img = sample.get(config.image_column)
    return [img] if img is not None else []


def _prepare_image(img: Image.Image) -> Image.Image | None:
    try:
        rgb = img.convert("RGB")
        return rgb if min(rgb.size) >= 10 else None
    except Exception:
        return None


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


def _save_shard(
    tensors: dict[str, torch.Tensor],
    metadata: list[dict],
    shard_idx: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    name = f"shard_{shard_idx:06d}"
    t_path = shard_dir / f"{name}.safetensors"
    m_path = shard_dir / f"{name}.json"
    save_file(tensors, str(t_path))
    m_path.write_text(json.dumps(metadata))
    return t_path, m_path


def _upload_files(
    api: HfApi, repo_id: str, paths: list[Path],
    prefix: str = "shards", delete_local: bool = True,
) -> None:
    for path in paths:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{prefix}/{path.name}",
            repo_id=repo_id, repo_type="dataset",
        )
        if delete_local:
            path.unlink(missing_ok=True)


def _flush_shard(
    all_emb: torch.Tensor,
    all_meta: list[dict],
    shard_idx: int,
    count: int,
    save_mode: str,
    out: Path,
    api: HfApi,
    repo_id: str,
    delete_local: bool,
) -> None:
    chunk = all_emb[:count]
    meta = all_meta[:count]

    tensors: dict[str, torch.Tensor] = {}
    if save_mode == "pooled":
        tensors["embeddings"] = chunk.mean(dim=1) if chunk.ndim == 3 else chunk
    elif save_mode == "both":
        tensors["embeddings"] = chunk
        tensors["pooled"] = chunk.mean(dim=1) if chunk.ndim == 3 else chunk
    else:
        tensors["embeddings"] = chunk

    t_path, m_path = _save_shard(tensors, meta, shard_idx, out)
    _upload_files(api, repo_id, [t_path, m_path], delete_local=delete_local)


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
) -> None:
    """Stream a dataset, encode images, shard, and upload to HF Hub.

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

    shard_idx = existing
    img_buf: list[Image.Image] = []
    meta_buf: list[dict] = []
    shard_embs: list[torch.Tensor] = []
    shard_meta: list[dict] = []
    global_idx = 0
    t0 = time.time()
    pbar = tqdm(desc=f"{dataset_name} [shard {shard_idx}]", unit="img")

    for sample_idx, sample in enumerate(ds):
        try:
            images = _extract_images(sample, dataset_config)
        except Exception:
            continue

        for img_idx, raw in enumerate(images):
            if global_idx < skip_count:
                global_idx += 1
                continue

            image = _prepare_image(raw)
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
                    _flush_shard(
                        all_emb, shard_meta, shard_idx, shard_size,
                        save_mode, out, api, repo_id, delete_local,
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
        _flush_shard(
            all_emb, shard_meta, shard_idx, all_emb.shape[0],
            save_mode, out, api, repo_id, delete_local,
        )
        shard_idx += 1
        del all_emb

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
    for p, dest in [(cfg_path, "config.json"), (readme, "README.md")]:
        api.upload_file(
            path_or_fileobj=str(p), path_in_repo=dest,
            repo_id=repo_id, repo_type="dataset",
        )
        if delete_local:
            p.unlink(missing_ok=True)

    elapsed = time.time() - t0
    print(f"Done: {dataset_name} -> {repo_id}  "
          f"({shard_idx} shards, {global_idx} imgs, {elapsed:.0f}s)")
