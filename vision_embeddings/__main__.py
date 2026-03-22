"""CLI entry point: ``python -m vision_embeddings``."""

from __future__ import annotations

import argparse

import torch

from .datasets import CAULDRON_SUBSETS, get_dataset, list_datasets
from .encoders import create_encoder, get_encoder_config, list_encoders
from .pipeline import process_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache vision/video encoder embeddings for VLM datasets",
    )
    parser.add_argument(
        "--encoder", type=str, default="siglip2-so400m-384",
        choices=list_encoders(),
        help="Encoder name (default: siglip2-so400m-384)",
    )
    parser.add_argument(
        "--datasets", type=str, required=True,
        help="Comma-separated names, or: all | cauldron_all | standard",
    )
    parser.add_argument("--hf-org", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--save-mode", type=str, default="tokens",
        choices=["tokens", "pooled", "both"],
        help="tokens=full last_hidden_state, pooled=mean, both=both",
    )
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="/tmp/embedding_cache")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--list-encoders", action="store_true")

    args = parser.parse_args()

    if args.list_datasets:
        for d in list_datasets():
            print(d)
        return
    if args.list_encoders:
        for name in list_encoders():
            c = get_encoder_config(name)
            print(f"  {name:25s}  {c.model_id}  [{c.loader}]")
        return

    if args.shard_size <= 0 or args.batch_size <= 0:
        parser.error("--shard-size and --batch-size must be positive")

    standard = [
        "textvqa", "gqa", "a_okvqa", "vqav2",
        "coco", "okvqa", "visual_genome",
    ]
    if args.datasets == "all":
        ds_names = list_datasets()
    elif args.datasets == "cauldron_all":
        ds_names = [f"the_cauldron__{s}" for s in CAULDRON_SUBSETS]
    elif args.datasets == "standard":
        ds_names = standard
    else:
        ds_names = [d.strip() for d in args.datasets.split(",")]

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    enc_config = get_encoder_config(args.encoder)

    print(f"Encoder  : {enc_config.model_id}  [{enc_config.loader}]")
    print(f"Datasets : {len(ds_names)}")
    print(f"Save mode: {args.save_mode}")
    print(f"Upload   : {args.hf_org}/<encoder>--<dataset>\n")

    encoder = create_encoder(
        args.encoder,
        device=args.device,
        dtype=dtype,
        compile_model=not args.no_compile,
    )

    for ds_name in ds_names:
        ds_config = get_dataset(ds_name)
        repo_id = f"{args.hf_org}/{args.encoder}--{ds_name}"
        try:
            process_dataset(
                encoder=encoder,
                dataset_config=ds_config,
                encoder_config=enc_config,
                dataset_name=ds_name,
                repo_id=repo_id,
                output_dir=args.output_dir,
                shard_size=args.shard_size,
                batch_size=args.batch_size,
                max_images=args.max_images,
                save_mode=args.save_mode,
                hf_token=args.hf_token,
                delete_local=True,
            )
        except Exception as exc:
            print(f"ERROR {ds_name}: {exc}")
            continue


if __name__ == "__main__":
    main()
