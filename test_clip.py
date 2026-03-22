"""Real test with CLIP ViT-L/14-336 + The Cauldron ai2d (5 images)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from safetensors.torch import load_file


def main() -> None:
    from vision_embeddings import create_encoder, get_encoder_config

    enc_config = get_encoder_config("clip-vit-l-336")
    print(f"Loading {enc_config.model_id} on CPU...")
    encoder = create_encoder("clip-vit-l-336", device="cpu", compile_model=False)
    print(f"  embed_dim : {enc_config.embed_dim}")
    print(f"  num_tokens: {enc_config.num_tokens}")
    print(f"  resolution: {enc_config.resolution}")

    # --- encode 5 real images ---
    print("\n--- Streaming 5 images from the_cauldron__ai2d ---")
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceM4/the_cauldron", name="ai2d",
        split="train", streaming=True,
    )

    images = []
    for sample in ds:
        for img in (sample.get("images") or []):
            if img is not None:
                images.append(img.convert("RGB"))
            if len(images) >= 5:
                break
        if len(images) >= 5:
            break

    for i, img in enumerate(images):
        print(f"  [{i}] {img.size}")

    print("\nEncoding (CPU, may take ~30s)...")
    emb = encoder.encode_batch(images)
    print(f"  shape: {tuple(emb.shape)}")
    print(f"  dtype: {emb.dtype}")
    assert emb.shape[0] == 5
    assert emb.shape[2] == enc_config.embed_dim
    print("  PASS")

    # --- full pipeline: tokens + pooled + both ---
    from vision_embeddings import process_dataset
    from vision_embeddings.config import DatasetConfig

    ds_config = DatasetConfig(
        hf_id="HuggingFaceM4/the_cauldron",
        image_column="images", split="train",
        subset="ai2d", multi_image=True,
    )

    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = []

    for mode in ("tokens", "pooled", "both"):
        print(f"\n--- Pipeline save_mode={mode} (5 imgs, shard_size=3) ---")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("vision_embeddings.pipeline.HfApi", return_value=mock_api):
                process_dataset(
                    encoder=encoder,
                    dataset_config=ds_config,
                    encoder_config=enc_config,
                    dataset_name="clip_test",
                    repo_id="test/clip",
                    output_dir=tmpdir,
                    shard_size=3,
                    batch_size=2,
                    max_images=5,
                    save_mode=mode,
                    delete_local=False,
                )

            shard_dir = Path(tmpdir) / "clip_test" / "shards"
            for sf in sorted(shard_dir.glob("*.safetensors")):
                data = load_file(str(sf))
                meta = json.loads(sf.with_suffix(".json").read_text())
                for key, tensor in data.items():
                    print(f"  {sf.name} [{key}]: {tuple(tensor.shape)} {tensor.dtype}")
                assert len(meta) == data["embeddings"].shape[0]

        print("  PASS")

    print("\n" + "=" * 50)
    print("All CLIP tests passed!")


if __name__ == "__main__":
    main()
