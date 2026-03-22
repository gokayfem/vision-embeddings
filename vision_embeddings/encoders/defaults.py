"""Built-in encoder configurations."""

from __future__ import annotations

from ..config import EncoderConfig

DEFAULT_ENCODERS: dict[str, EncoderConfig] = {
    # ---- SigLIP2 ----
    "siglip2-so400m-384": EncoderConfig(
        model_id="google/siglip2-so400m-patch14-384",
        embed_dim=1152, num_tokens=729, resolution=384,
    ),
    # ---- SigLIP v1 ----
    "siglip-so400m-384": EncoderConfig(
        model_id="google/siglip-so400m-patch14-384",
        embed_dim=1152, num_tokens=729, resolution=384,
    ),
    # ---- CLIP ----
    "clip-vit-l-336": EncoderConfig(
        model_id="openai/clip-vit-large-patch14-336",
        embed_dim=1024, num_tokens=577, resolution=336,
    ),
    # ---- DINOv2 ----
    "dinov2-base": EncoderConfig(
        model_id="facebook/dinov2-base",
        embed_dim=768, num_tokens=1370, resolution=518,
    ),
    "dinov2-large": EncoderConfig(
        model_id="facebook/dinov2-large",
        embed_dim=1024, num_tokens=1370, resolution=518,
    ),
    # ---- DINOv3 ----
    "dinov3-vitb16": EncoderConfig(
        model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        embed_dim=768, num_tokens=197, resolution=224,
    ),
    # ---- InternViT ----
    "internvit-300m-448": EncoderConfig(
        model_id="OpenGVLab/InternViT-300M-448px-V2_5",
        embed_dim=1024, num_tokens=1025, resolution=448,
    ),
    # ---- V-JEPA 2 (HuggingFace) ----
    "vjepa2-vitl-256": EncoderConfig(
        model_id="facebook/vjepa2-vitl-fpc64-256",
        embed_dim=1024, num_tokens=256, resolution=256,
        loader="hf_video", frames_per_clip=64,
    ),
    "vjepa2-vith-256": EncoderConfig(
        model_id="facebook/vjepa2-vith-fpc64-256",
        embed_dim=1280, num_tokens=256, resolution=256,
        loader="hf_video", frames_per_clip=64,
    ),
    "vjepa2-vitg-256": EncoderConfig(
        model_id="facebook/vjepa2-vitg-fpc64-256",
        embed_dim=1408, num_tokens=256, resolution=256,
        loader="hf_video", frames_per_clip=64,
    ),
    "vjepa2-vitg-384": EncoderConfig(
        model_id="facebook/vjepa2-vitg-fpc64-384",
        embed_dim=1408, num_tokens=576, resolution=384,
        loader="hf_video", frames_per_clip=64,
    ),
    # ---- V-JEPA 2.1 (torch.hub) ----
    "vjepa2.1-vitb-384": EncoderConfig(
        model_id="vjepa2.1-vitb-384",
        embed_dim=768, num_tokens=576, resolution=384,
        loader="torch_hub", frames_per_clip=64,
        hub_repo="facebookresearch/vjepa2",
        hub_name="vjepa2_1_vit_base_384",
        ckpt_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        encoder_key="ema_encoder",
    ),
    "vjepa2.1-vitl-384": EncoderConfig(
        model_id="vjepa2.1-vitl-384",
        embed_dim=1024, num_tokens=576, resolution=384,
        loader="torch_hub", frames_per_clip=64,
        hub_repo="facebookresearch/vjepa2",
        hub_name="vjepa2_1_vit_large_384",
        ckpt_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
        encoder_key="ema_encoder",
    ),
    "vjepa2.1-vitg-384": EncoderConfig(
        model_id="vjepa2.1-vitg-384",
        embed_dim=1408, num_tokens=576, resolution=384,
        loader="torch_hub", frames_per_clip=64,
        hub_repo="facebookresearch/vjepa2",
        hub_name="vjepa2_1_vit_giant_384",
        ckpt_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt",
        encoder_key="target_encoder",
    ),
    "vjepa2.1-vitG-384": EncoderConfig(
        model_id="vjepa2.1-vitG-384",
        embed_dim=1664, num_tokens=576, resolution=384,
        loader="torch_hub", frames_per_clip=64,
        hub_repo="facebookresearch/vjepa2",
        hub_name="vjepa2_1_vit_gigantic_384",
        ckpt_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt",
        encoder_key="target_encoder",
    ),
}
