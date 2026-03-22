"""Local tests — no GPU, no big downloads, no HF uploads."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

# ------------------------------------------------------------------
# 1. Imports — verify the full package loads cleanly
# ------------------------------------------------------------------

def test_imports():
    from vision_embeddings import (
        BaseEncoder,
        CAULDRON_SUBSETS,
        DatasetConfig,
        EncoderConfig,
        HFVideoEncoder,
        HFVisionEncoder,
        TorchHubEncoder,
        create_encoder,
        find_optimal_batch_size,
        get_dataset,
        get_encoder_config,
        list_datasets,
        list_encoders,
        process_dataset,
        register_dataset,
        register_encoder,
    )
    print("OK  imports: all 16 public symbols load")


# ------------------------------------------------------------------
# 2. Registry — encoders and datasets
# ------------------------------------------------------------------

def test_encoder_registry():
    from vision_embeddings import get_encoder_config, list_encoders, EncoderConfig

    names = list_encoders()
    assert len(names) >= 15, f"Expected >=15 encoders, got {len(names)}"

    for name in names:
        cfg = get_encoder_config(name)
        assert isinstance(cfg, EncoderConfig)
        assert cfg.embed_dim > 0
        assert cfg.num_tokens > 0
        assert cfg.loader in ("hf_vision", "hf_video", "torch_hub")

    print(f"OK  encoder registry: {len(names)} encoders, all valid")


def test_dataset_registry():
    from vision_embeddings import get_dataset, list_datasets, DatasetConfig

    names = list_datasets()
    assert len(names) >= 57, f"Expected >=57 datasets, got {len(names)}"

    for name in names:
        cfg = get_dataset(name)
        assert isinstance(cfg, DatasetConfig)
        assert cfg.hf_id
        assert cfg.image_column
        assert cfg.split

    cauldron = [n for n in names if n.startswith("the_cauldron__")]
    assert len(cauldron) >= 50

    print(f"OK  dataset registry: {len(names)} datasets ({len(cauldron)} cauldron)")


# ------------------------------------------------------------------
# 3. Custom registration
# ------------------------------------------------------------------

def test_register_custom():
    from vision_embeddings import (
        register_encoder, register_dataset,
        get_encoder_config, get_dataset,
        EncoderConfig, DatasetConfig,
    )

    register_encoder("test-vit", EncoderConfig(
        model_id="test/vit", embed_dim=768, num_tokens=196, resolution=224,
    ))
    cfg = get_encoder_config("test-vit")
    assert cfg.model_id == "test/vit"
    assert cfg.embed_dim == 768

    register_dataset("test-data", DatasetConfig(
        hf_id="test/data", image_column="img", split="train",
    ))
    ds = get_dataset("test-data")
    assert ds.hf_id == "test/data"

    print("OK  custom registration: encoder + dataset registered and retrieved")


# ------------------------------------------------------------------
# 4. Config immutability
# ------------------------------------------------------------------

def test_config_frozen():
    from vision_embeddings import EncoderConfig
    cfg = EncoderConfig(model_id="x", embed_dim=1, num_tokens=1, resolution=1)
    try:
        cfg.embed_dim = 999  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass
    print("OK  config immutability: frozen dataclass works")


# ------------------------------------------------------------------
# 5. Mock encoder — test encode_batch contract
# ------------------------------------------------------------------

class _MockEncoder:
    """Fake encoder for pipeline tests — no model, no GPU."""

    model_id = "mock/encoder"
    embed_dim = 32
    num_tokens = 4

    def encode_batch(self, images: list[Image.Image]) -> torch.Tensor:
        n = len(images)
        return torch.randn(n, self.num_tokens, self.embed_dim, dtype=torch.float16)

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.randn(len(images), 3, 64, 64)

    def encode_preprocessed(self, pv: torch.Tensor) -> torch.Tensor:
        n = pv.shape[0]
        return torch.randn(n, self.num_tokens, self.embed_dim, dtype=torch.float16)


def test_mock_encoder():
    enc = _MockEncoder()
    imgs = [Image.new("RGB", (64, 64), color="red") for _ in range(8)]
    out = enc.encode_batch(imgs)
    assert out.shape == (8, 4, 32)
    assert out.dtype == torch.float16
    print("OK  mock encoder: shape and dtype correct")


# ------------------------------------------------------------------
# 6. Shard tensors — build + save modes
# ------------------------------------------------------------------

def test_build_shard_tensors():
    from vision_embeddings.pipeline import _build_shard_tensors

    emb = torch.randn(10, 4, 32)

    t = _build_shard_tensors(emb, "tokens")
    assert "embeddings" in t
    assert t["embeddings"].shape == (10, 4, 32)

    t = _build_shard_tensors(emb, "pooled")
    assert t["embeddings"].shape == (10, 32)

    t = _build_shard_tensors(emb, "both")
    assert t["embeddings"].shape == (10, 4, 32)
    assert t["pooled"].shape == (10, 32)

    print("OK  shard tensors: tokens/pooled/both modes correct")


# ------------------------------------------------------------------
# 7. Shard save to disk
# ------------------------------------------------------------------

def test_shard_save():
    from vision_embeddings.pipeline import _build_shard_tensors
    from safetensors.torch import save_file, load_file

    emb = torch.randn(5, 4, 32, dtype=torch.float16)
    tensors = _build_shard_tensors(emb, "tokens")
    meta = [{"global_idx": i, "sample_idx": i, "image_idx": 0} for i in range(5)]

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        t_path = p / "shard_000000.safetensors"
        m_path = p / "shard_000000.json"
        save_file(tensors, str(t_path))
        m_path.write_text(json.dumps(meta))

        loaded = load_file(str(t_path))
        assert loaded["embeddings"].shape == (5, 4, 32)
        assert loaded["embeddings"].dtype == torch.float16

        loaded_meta = json.loads(m_path.read_text())
        assert len(loaded_meta) == 5
        assert loaded_meta[0]["global_idx"] == 0

    print("OK  shard save: safetensors + json round-trip correct")


# ------------------------------------------------------------------
# 8. Batch upload — verify create_commit is called
# ------------------------------------------------------------------

def test_batch_upload():
    from vision_embeddings.batch_upload import upload_shard_batched

    emb = torch.randn(3, 4, 32, dtype=torch.float16)
    meta = [{"global_idx": i} for i in range(3)]

    mock_api = MagicMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        upload_shard_batched(
            mock_api, "test/repo",
            {"embeddings": emb}, meta,
            shard_idx=0, output_dir=Path(tmpdir),
            delete_local=True,
        )
        mock_api.create_commit.assert_called_once()
        call_kwargs = mock_api.create_commit.call_args
        assert call_kwargs.kwargs["repo_type"] == "dataset"
        ops = call_kwargs.kwargs["operations"]
        assert len(ops) == 2  # safetensors + json in one commit

    print("OK  batch upload: create_commit called with 2 operations")


# ------------------------------------------------------------------
# 9. Image preparation
# ------------------------------------------------------------------

def test_prepare_image():
    from vision_embeddings.pipeline import _prepare_image

    good = Image.new("RGB", (100, 100))
    assert _prepare_image(good) is not None

    tiny = Image.new("RGB", (5, 5))
    assert _prepare_image(tiny) is None

    rgba = Image.new("RGBA", (100, 100))
    result = _prepare_image(rgba)
    assert result is not None
    assert result.mode == "RGB"

    print("OK  image prep: valid/tiny/RGBA handled correctly")


# ------------------------------------------------------------------
# 10. Auto batch — CPU path (no CUDA)
# ------------------------------------------------------------------

def test_auto_batch_cpu():
    from vision_embeddings.auto_batch import find_optimal_batch_size

    enc = _MockEncoder()
    result = find_optimal_batch_size(enc, resolution=64, max_batch=128)
    # no CUDA available → returns max_batch directly
    assert result == 128
    print("OK  auto batch (CPU): returns max_batch when no CUDA")


# ------------------------------------------------------------------
# 11. Full pipeline with mock encoder + mock HF
# ------------------------------------------------------------------

def test_pipeline_mock():
    from vision_embeddings.pipeline import process_dataset
    from vision_embeddings.config import DatasetConfig, EncoderConfig

    enc = _MockEncoder()
    enc_config = EncoderConfig(
        model_id="mock/encoder", embed_dim=32, num_tokens=4, resolution=64,
    )

    # create a tiny fake streaming dataset
    fake_samples = [
        {"image": Image.new("RGB", (64, 64), color=(i * 30, 0, 0))}
        for i in range(8)
    ]

    ds_config = DatasetConfig(
        hf_id="fake/dataset", image_column="image", split="train",
    )

    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = []

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("vision_embeddings.pipeline.load_dataset") as mock_load, \
             patch("vision_embeddings.pipeline.HfApi", return_value=mock_api):
            mock_load.return_value = iter(fake_samples)

            process_dataset(
                encoder=enc,
                dataset_config=ds_config,
                encoder_config=enc_config,
                dataset_name="test_ds",
                repo_id="test/repo",
                output_dir=tmpdir,
                shard_size=5,
                batch_size=3,
                save_mode="tokens",
                delete_local=False,
            )

        # should have created shards
        shard_dir = Path(tmpdir) / "test_ds" / "shards"
        safetensor_files = list(shard_dir.glob("*.safetensors"))
        json_files = list(shard_dir.glob("*.json"))
        assert len(safetensor_files) >= 1, f"Expected shards, found {safetensor_files}"
        assert len(json_files) >= 1

        from safetensors.torch import load_file
        first = load_file(str(sorted(safetensor_files)[0]))
        assert "embeddings" in first
        assert first["embeddings"].ndim == 3  # [N, seq, dim]

        # verify HF interactions
        mock_api.create_repo.assert_called_once()
        assert mock_api.create_commit.call_count >= 1  # batched upload

    print(f"OK  pipeline mock: {len(safetensor_files)} shard(s) created, "
          f"HF create_commit called {mock_api.create_commit.call_count}x")


# ------------------------------------------------------------------
# Run all
# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_imports,
        test_encoder_registry,
        test_dataset_registry,
        test_register_custom,
        test_config_frozen,
        test_mock_encoder,
        test_build_shard_tensors,
        test_shard_save,
        test_batch_upload,
        test_prepare_image,
        test_auto_batch_cpu,
        test_pipeline_mock,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"{passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        raise SystemExit(1)
