"""Real end-to-end test — small model + 5 real images, no upload."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from safetensors.torch import load_file


def main() -> None:
    print("Loading dinov2-base on CPU (smallest encoder, ~86M params)...")
    from vision_embeddings import create_encoder, get_encoder_config

    enc_config = get_encoder_config("dinov2-base")
    encoder = create_encoder(
        "dinov2-base",
        device="cpu",
        compile_model=False,
    )
    print(f"  model_id  : {enc_config.model_id}")
    print(f"  embed_dim : {enc_config.embed_dim}")
    print(f"  num_tokens: {enc_config.num_tokens}")
    print(f"  resolution: {enc_config.resolution}")

    # --- test 1: raw encode_batch with dummy images ---
    print("\n--- Test 1: encode_batch with 3 dummy images ---")
    from PIL import Image
    dummy_imgs = [
        Image.new("RGB", (256, 256), color="red"),
        Image.new("RGB", (512, 300), color="green"),
        Image.new("RGB", (100, 400), color="blue"),
    ]
    emb = encoder.encode_batch(dummy_imgs)
    print(f"  output shape: {tuple(emb.shape)}")
    print(f"  dtype       : {emb.dtype}")
    print(f"  device      : {emb.device}")
    assert emb.ndim == 3
    assert emb.shape[0] == 3
    assert emb.shape[2] == enc_config.embed_dim
    print("  PASS")

    # --- test 2: stream 5 real images from The Cauldron (ai2d) ---
    print("\n--- Test 2: stream 5 real images from the_cauldron__ai2d ---")
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceM4/the_cauldron",
        name="ai2d",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    real_images = []
    for sample in ds:
        imgs = sample.get("images") or []
        for img in imgs:
            if img is not None:
                real_images.append(img.convert("RGB"))
            if len(real_images) >= 5:
                break
        if len(real_images) >= 5:
            break

    print(f"  collected {len(real_images)} images")
    for i, img in enumerate(real_images):
        print(f"    [{i}] size={img.size} mode={img.mode}")

    emb = encoder.encode_batch(real_images)
    print(f"  encoded shape: {tuple(emb.shape)}")
    assert emb.shape[0] == len(real_images)
    print("  PASS")

    # --- test 3: full pipeline with real images, mock upload ---
    print("\n--- Test 3: full pipeline (5 imgs, shard_size=3, mock upload) ---")
    from vision_embeddings import process_dataset
    from vision_embeddings.config import DatasetConfig

    ds_config = DatasetConfig(
        hf_id="HuggingFaceM4/the_cauldron",
        image_column="images",
        split="train",
        subset="ai2d",
        multi_image=True,
    )

    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = []

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("vision_embeddings.pipeline.HfApi", return_value=mock_api), \
             patch("vision_embeddings.batch_upload.HfApi", return_value=mock_api):
            process_dataset(
                encoder=encoder,
                dataset_config=ds_config,
                encoder_config=enc_config,
                dataset_name="cauldron_ai2d",
                repo_id="test/cauldron-ai2d",
                output_dir=tmpdir,
                shard_size=3,
                batch_size=2,
                max_images=5,
                save_mode="tokens",
                delete_local=False,
            )

        shard_dir = Path(tmpdir) / "cauldron_ai2d" / "shards"
        st_files = sorted(shard_dir.glob("*.safetensors"))
        json_files = sorted(shard_dir.glob("*.json"))

        print(f"\n  shards written: {len(st_files)}")
        for sf in st_files:
            data = load_file(str(sf))
            emb = data["embeddings"]
            jf = sf.with_suffix(".json")
            meta = json.loads(jf.read_text())
            print(f"    {sf.name}: embeddings {tuple(emb.shape)} dtype={emb.dtype}, "
                  f"meta entries={len(meta)}")
            assert emb.shape[0] == len(meta)
            assert emb.shape[2] == enc_config.embed_dim

        cfg_path = Path(tmpdir) / "cauldron_ai2d" / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            print(f"\n  config.json:")
            for k, v in cfg.items():
                print(f"    {k}: {v}")

        print("\n  HF API calls:")
        print(f"    create_repo: {mock_api.create_repo.call_count}x")
        print(f"    create_commit: {mock_api.create_commit.call_count}x (batched uploads)")
        print(f"    upload_file: {mock_api.upload_file.call_count}x (fallback uploads)")

    print("\n  PASS")
    print("\n" + "=" * 50)
    print("All 3 tests passed!")


if __name__ == "__main__":
    main()
