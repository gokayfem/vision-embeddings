"""Batched HF Hub upload via create_commit — one HTTP round-trip per shard pair.

The default ``upload_file`` makes one HTTP request per file. For a shard
(safetensors + json), that is 2 requests with full auth/handshake overhead.
``create_commit`` bundles both files into a single atomic commit, cutting
the number of HTTP round-trips in half and reducing latency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def upload_shard_batched(
    api: HfApi,
    repo_id: str,
    tensors: dict[str, torch.Tensor],
    metadata: list[dict],
    shard_idx: int,
    output_dir: Path,
    delete_local: bool = True,
) -> None:
    """Save a shard locally, then upload both files in one commit."""
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    name = f"shard_{shard_idx:06d}"
    t_path = shard_dir / f"{name}.safetensors"
    m_path = shard_dir / f"{name}.json"

    save_file(tensors, str(t_path))
    m_path.write_text(json.dumps(metadata))

    operations = [
        CommitOperationAdd(
            path_in_repo=f"shards/{t_path.name}",
            path_or_fileobj=str(t_path),
        ),
        CommitOperationAdd(
            path_in_repo=f"shards/{m_path.name}",
            path_or_fileobj=str(m_path),
        ),
    ]

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add shard {shard_idx:06d}",
        )
    except Exception as exc:
        logger.warning("Batched commit failed, falling back to per-file: %s", exc)
        for path in (t_path, m_path):
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"shards/{path.name}",
                repo_id=repo_id,
                repo_type="dataset",
            )

    if delete_local:
        t_path.unlink(missing_ok=True)
        m_path.unlink(missing_ok=True)


def upload_config_and_readme(
    api: HfApi,
    repo_id: str,
    config_path: Path,
    readme_path: Path,
    delete_local: bool = True,
) -> None:
    """Upload config.json + README.md in a single commit."""
    operations = [
        CommitOperationAdd(
            path_in_repo="config.json",
            path_or_fileobj=str(config_path),
        ),
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=str(readme_path),
        ),
    ]

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message="Update config and README",
        )
    except Exception as exc:
        logger.warning("Batched config commit failed, falling back: %s", exc)
        for p, dest in [(config_path, "config.json"), (readme_path, "README.md")]:
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=dest,
                repo_id=repo_id,
                repo_type="dataset",
            )

    if delete_local:
        config_path.unlink(missing_ok=True)
        readme_path.unlink(missing_ok=True)
