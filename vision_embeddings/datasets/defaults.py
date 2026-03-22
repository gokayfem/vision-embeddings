"""Built-in dataset configurations."""

from __future__ import annotations

from ..config import DatasetConfig

CAULDRON_SUBSETS: list[str] = [
    "ai2d", "aokvqa", "chart2text", "chartqa", "clevr", "clevr_math",
    "cocoqa", "datikz", "diagram_image_to_text", "docvqa", "dvqa",
    "figureqa", "finqa", "geomverse", "hateful_memes", "hitab", "iam",
    "iconqa", "infographic_vqa", "intergps", "localized_narratives",
    "mapqa", "mimic_cgd", "multihiertt", "nlvr2", "ocrvqa", "okvqa",
    "plotqa", "raven", "rendered_text", "robut_sqa", "robut_wikisql",
    "robut_wtq", "scienceqa", "screen2words", "spot_the_diff", "st_vqa",
    "tabmwp", "tallyqa", "tat_qa", "textcaps", "textvqa", "tqa",
    "vistext", "visual7w", "visualmrc", "vqarad", "vqav2", "vsr",
    "websight",
]

DEFAULT_DATASETS: dict[str, DatasetConfig] = {}

for _s in CAULDRON_SUBSETS:
    DEFAULT_DATASETS[f"the_cauldron__{_s}"] = DatasetConfig(
        hf_id="HuggingFaceM4/the_cauldron",
        image_column="images", split="train",
        subset=_s, multi_image=True,
    )

DEFAULT_DATASETS.update({
    "textvqa": DatasetConfig(
        hf_id="facebook/textvqa", image_column="image", split="train",
    ),
    "gqa": DatasetConfig(
        hf_id="lmms-lab/GQA", image_column="image",
        split="train_all_images",
    ),
    "a_okvqa": DatasetConfig(
        hf_id="HuggingFaceM4/A-OKVQA", image_column="image", split="train",
    ),
    "vqav2": DatasetConfig(
        hf_id="HuggingFaceM4/VQAv2", image_column="image", split="train",
    ),
    "coco": DatasetConfig(
        hf_id="HuggingFaceM4/COCO", image_column="image", split="train",
    ),
    "okvqa": DatasetConfig(
        hf_id="lmms-lab/OK-VQA", image_column="image", split="val2014",
    ),
    "visual_genome": DatasetConfig(
        hf_id="visual_genome", image_column="image", split="train",
        subset="region_descriptions_v1.0.0",
    ),
})
