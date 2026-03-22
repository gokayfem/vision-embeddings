# vision-embeddings

Extract, cache, and upload vision encoder embeddings from VLM training datasets to HuggingFace Hub.

## Install

```bash
pip install vision-embeddings

# For V-JEPA 2.1 support
pip install "vision-embeddings[vjepa]"
```

## Supported Encoders

| Family | Names | Loader |
| --- | --- | --- |
| SigLIP2 | `siglip2-so400m-384` | hf_vision |
| SigLIP v1 | `siglip-so400m-384` | hf_vision |
| CLIP | `clip-vit-l-336` | hf_vision |
| DINOv2 | `dinov2-base`, `dinov2-large` | hf_vision |
| DINOv3 | `dinov3-vitb16` | hf_vision |
| InternViT | `internvit-300m-448` | hf_vision |
| V-JEPA 2 | `vjepa2-vitl-256`, `vjepa2-vith-256`, `vjepa2-vitg-256`, `vjepa2-vitg-384` | hf_video |
| V-JEPA 2.1 | `vjepa2.1-vitb-384`, `vjepa2.1-vitl-384`, `vjepa2.1-vitg-384`, `vjepa2.1-vitG-384` | torch_hub |

## Supported Datasets

57 datasets: The Cauldron (50 subsets), TextVQA, GQA, VQAv2, A-OKVQA, COCO, OK-VQA, Visual Genome.

## Usage

### Python API

```python
from vision_embeddings import create_encoder, get_dataset, process_dataset, get_encoder_config

# Create encoder
encoder = create_encoder("siglip2-so400m-384")
enc_config = get_encoder_config("siglip2-so400m-384")

# Process a dataset -> shards -> HuggingFace Hub
process_dataset(
    encoder=encoder,
    dataset_config=get_dataset("textvqa"),
    encoder_config=enc_config,
    dataset_name="textvqa",
    repo_id="your-org/siglip2-so400m-384--textvqa",
    batch_size=64,
    shard_size=1000,
    save_mode="tokens",   # "tokens" | "pooled" | "both"
)
```

### Custom Encoder

```python
from vision_embeddings import register_encoder, EncoderConfig

register_encoder("my-vit", EncoderConfig(
    model_id="my-org/my-vit",
    embed_dim=768,
    num_tokens=196,
    resolution=224,
))

encoder = create_encoder("my-vit")
```

### Custom Dataset

```python
from vision_embeddings import register_dataset, DatasetConfig

register_dataset("my-data", DatasetConfig(
    hf_id="my-org/my-dataset",
    image_column="image",
    split="train",
))
```

### CLI

```bash
# Image encoder
vision-embeddings \
    --encoder siglip2-so400m-384 \
    --datasets textvqa,the_cauldron__ai2d \
    --hf-org your-username \
    --batch-size 64

# Video encoder (use smaller batches + pooled mode)
vision-embeddings \
    --encoder vjepa2-vitg-384 \
    --datasets standard \
    --hf-org your-username \
    --batch-size 4 \
    --save-mode pooled

# List all available encoders / datasets
vision-embeddings --list-encoders --hf-org x --datasets x
vision-embeddings --list-datasets --hf-org x --datasets x
```

### Save Modes

| Mode | Shape | Size per 1K images (SigLIP2) |
| --- | --- | --- |
| `tokens` | `[N, 729, 1152]` | ~1.6 GB |
| `pooled` | `[N, 1152]` | ~2.3 MB |
| `both` | both tensors | ~1.6 GB |

## Output Format

Each dataset is uploaded to its own HuggingFace dataset repo:

```
shards/
  shard_000000.safetensors   # {"embeddings": [1000, 729, 1152]}
  shard_000000.json          # metadata per image
  shard_000001.safetensors
  ...
config.json
README.md
```

### Loading

```python
from safetensors.torch import load_file

shard = load_file("shards/shard_000000.safetensors")
embeddings = shard["embeddings"]  # [1000, 729, 1152]
```

## Features

- **Streaming** — no full dataset download needed
- **Resume-safe** — detects existing shards on HF Hub, picks up where it left off
- **Colab-friendly** — save-upload-delete pattern keeps local disk clean
- **Fast** — `torch.compile` + `tf32` + `float16`
- **Extensible** — `register_encoder()` / `register_dataset()` for custom additions

## License

Apache 2.0
