"""Part B: Local SegFormer fine-tuning on custom urban street imagery.

Can be run with:
    python fine_tune_segformer.py

If `data/images/` and `data/masks/` are not populated, this script automatically creates
a demonstration dataset of urban street scenes with ground-truth semantic masks,
fine-tunes `nvidia/mit-b0` directly to the ResiliCity 8-class taxonomy,
and saves the output to `backend/segformer-local-final`.
"""

import glob
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainingArguments,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
OUTPUT_DIR = HERE / "segformer-local-final"

# 8 ResiliCity target classes
LABELS = ["other", "road", "roof", "pavement", "wall", "vegetation", "water", "sky"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def generate_synthetic_samples(num_samples: int = 16) -> None:
    """Generate realistic demonstration urban street images and class-ID masks."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} demonstration urban street samples with ground-truth masks...")
    rng = np.random.default_rng(42)

    for i in range(num_samples):
        w, h = 512, 512
        img = Image.new("RGB", (w, h), (180, 210, 240))  # Sky base
        msk = Image.new("L", (w, h), LABEL2ID["sky"])
        draw_img = ImageDraw.Draw(img)
        draw_msk = ImageDraw.Draw(msk)

        # Buildings / Walls
        num_bldgs = rng.integers(2, 5)
        bldg_width = w // num_bldgs
        for b in range(num_bldgs):
            bx1 = b * bldg_width
            bx2 = (b + 1) * bldg_width
            by_top = rng.integers(120, 240)
            by_roof = by_top - rng.integers(25, 45)
            by_bottom = 400

            # Wall
            wall_color = (rng.integers(140, 210), rng.integers(130, 200), rng.integers(120, 190))
            draw_img.rectangle([bx1, by_top, bx2, by_bottom], fill=wall_color)
            draw_msk.rectangle([bx1, by_top, bx2, by_bottom], fill=LABEL2ID["wall"])

            # Roof
            roof_color = (rng.integers(120, 160), rng.integers(60, 90), rng.integers(50, 80))
            draw_img.polygon([(bx1, by_top), (bx2, by_top), ((bx1 + bx2) // 2, by_roof)], fill=roof_color)
            draw_msk.polygon([(bx1, by_top), (bx2, by_top), ((bx1 + bx2) // 2, by_roof)], fill=LABEL2ID["roof"])

        # Road
        road_top = 370
        road_color = (55, 60, 65)
        draw_img.rectangle([0, road_top, w, h], fill=road_color)
        draw_msk.rectangle([0, road_top, w, h], fill=LABEL2ID["road"])

        # Sidewalk / Pavements on left & right
        sw_l = rng.integers(40, 80)
        sw_r = rng.integers(40, 80)
        pave_color = (160, 165, 170)
        draw_img.polygon([(0, road_top), (sw_l, road_top), (sw_l * 1.5, h), (0, h)], fill=pave_color)
        draw_msk.polygon([(0, road_top), (sw_l, road_top), (sw_l * 1.5, h), (0, h)], fill=LABEL2ID["pavement"])
        draw_img.polygon([(w - sw_r, road_top), (w, road_top), (w, h), (w - sw_r * 1.5, h)], fill=pave_color)
        draw_msk.polygon([(w - sw_r, road_top), (w, road_top), (w, h), (w - sw_r * 1.5, h)], fill=LABEL2ID["pavement"])

        # Vegetation / Trees
        num_trees = rng.integers(2, 4)
        for _ in range(num_trees):
            tx = rng.integers(30, w - 30)
            ty = rng.integers(240, 360)
            trad = rng.integers(25, 45)
            veg_color = (rng.integers(30, 70), rng.integers(120, 180), rng.integers(40, 80))
            draw_img.ellipse([tx - trad, ty - trad, tx + trad, ty + trad], fill=veg_color)
            draw_msk.ellipse([tx - trad, ty - trad, tx + trad, ty + trad], fill=LABEL2ID["vegetation"])

        img.save(IMAGES_DIR / f"scene_{i:03d}.png")
        msk.save(MASKS_DIR / f"scene_{i:03d}.png")

    print(f"Sample dataset ready at {IMAGES_DIR} and {MASKS_DIR}")


class UrbanSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, processor):
        self.image_files = image_files
        self.processor = processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        stem = Path(img_path).stem
        mask_path = MASKS_DIR / f"{stem}.png"
        if not mask_path.exists():
            mask_path = MASKS_DIR / f"{stem}.jpg"

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        encoded = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}


def main():
    print("=" * 60)
    print("ResiliCity Part B: SegFormer Local Fine-Tuning")
    print("=" * 60)

    # Check for existing dataset or generate starter samples
    files = sorted(
        glob.glob(str(IMAGES_DIR / "*.jpg"))
        + glob.glob(str(IMAGES_DIR / "*.png"))
        + glob.glob(str(DATA_DIR / "images" / "*.jpg"))
        + glob.glob(str(DATA_DIR / "images" / "*.png"))
    )

    if not files:
        generate_synthetic_samples(num_samples=16)
        files = sorted(glob.glob(str(IMAGES_DIR / "*.png")))

    print(f"Total training image pairs available: {len(files)}")

    processor = SegformerImageProcessor(
        do_reduce_labels=False,
        size={"height": 512, "width": 512},
    )

    rng = np.random.default_rng(42)
    shuffled = list(files)
    rng.shuffle(shuffled)
    n_val = max(1, len(shuffled) // 5)
    train_files = shuffled[n_val:]
    val_files = shuffled[:n_val]

    train_ds = UrbanSegmentationDataset(train_files, processor)
    val_ds = UrbanSegmentationDataset(val_files, processor)
    print(f"Split: {len(train_ds)} train, {len(val_ds)} validation samples")

    print("Loading base SegFormer architecture (nvidia/mit-b0)...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(HERE / "segformer-checkpoints"),
        learning_rate=6e-5,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        logging_steps=2,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("Starting fine-tuning...")
    train_result = trainer.train()

    print(f"Fine-tuning complete. Saving model to: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    # Save fine-tuning metadata & metrics
    metrics = {
        "train_loss": float(train_result.training_loss),
        "num_classes": len(LABELS),
        "classes": LABELS,
        "base_model": "nvidia/mit-b0",
        "num_samples": len(files),
    }
    (OUTPUT_DIR / "fine_tune_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved: {metrics}")
    print("SUCCESS: Part B SegFormer fine-tuned model ready!")


if __name__ == "__main__":
    main()
