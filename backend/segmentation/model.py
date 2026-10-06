"""Singleton SegFormer model loader and inference engine."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from .contours import extract_polygons_for_mask
from .mapper import (
    CLASS_METADATA,
    RESILICITY_CLASSES,
    build_label_map,
)

HERE = Path(__file__).resolve().parent.parent
LOCAL_MODEL_DIR = HERE / "segformer-local-final"
DEFAULT_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"


class SegFormerEngine:
    _instance: Optional["SegFormerEngine"] = None

    def __init__(self):
        # Determine model path / identifier
        env_model = os.environ.get("SEGMENTER_MODEL")
        if env_model:
            self.model_name_or_path = env_model
            self.source = "custom_env"
        elif LOCAL_MODEL_DIR.exists():
            self.model_name_or_path = str(LOCAL_MODEL_DIR)
            self.source = "local_fine_tuned"
        else:
            self.model_name_or_path = DEFAULT_MODEL_ID
            self.source = "pretrained"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor: Optional[SegformerImageProcessor] = None
        self.model: Optional[SegformerForSemanticSegmentation] = None
        self.label_map: Dict[int, str] = {}
        self.is_loaded = False
        self.load_error: Optional[str] = None

        self._load_model()

    def _load_model(self) -> None:
        try:
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name_or_path)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_name_or_path
            ).to(self.device)
            self.model.eval()

            # Build label mapping from model's id2label
            id2label = {int(k): v for k, v in self.model.config.id2label.items()}
            self.label_map = build_label_map(id2label)
            self.is_loaded = True
            self.load_error = None
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            self.model = None
            self.processor = None

    @classmethod
    def get_instance(cls) -> "SegFormerEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_status(self) -> Dict[str, Any]:
        return {
            "available": self.is_loaded,
            "model_name": self.model_name_or_path,
            "device": str(self.device),
            "source": self.source,
            "error": self.load_error,
        }

    def segment_image(self, image: Image.Image) -> Dict[str, Any]:
        """Perform semantic segmentation on a PIL Image and return structured result."""
        if not self.is_loaded or self.model is None or self.processor is None:
            raise RuntimeError(
                f"SegFormer model not loaded: {self.load_error or 'Unknown initialization error'}"
            )

        orig_w, orig_h = image.size
        total_pixels = orig_w * orig_h

        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: (1, num_classes, H/4, W/4)

            # Upsample logits to original image dimensions
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
            raw_preds = upsampled_logits.argmax(dim=1)[0].cpu().numpy()  # shape: (orig_h, orig_w)

        # Count pixels per ResiliCity class
        unique_ids, counts = np.unique(raw_preds, return_counts=True)
        class_pixel_counts = {c: 0 for c in RESILICITY_CLASSES}

        for model_id, count in zip(unique_ids, counts):
            target_class = self.label_map.get(int(model_id), "other")
            class_pixel_counts[target_class] += int(count)

        # Construct masks and polygons
        classes_output = []
        masks_output = []

        # Map pixel positions to ResiliCity class masks where pixels exist
        for cls_name in RESILICITY_CLASSES:
            px_count = class_pixel_counts[cls_name]
            if px_count == 0:
                continue

            pct = round((px_count / total_pixels) * 100.0, 1)

            # Identify binary mask for this class
            # Find which raw prediction IDs map to this ResiliCity class
            matching_ids = [k for k, v in self.label_map.items() if v == cls_name]
            if not matching_ids:
                binary_mask = np.zeros((orig_h, orig_w), dtype=bool)
            elif len(matching_ids) == 1:
                binary_mask = (raw_preds == matching_ids[0])
            else:
                binary_mask = np.isin(raw_preds, matching_ids)

            polygons = extract_polygons_for_mask(binary_mask)
            primary_polygon = polygons[0] if polygons else None

            label, albedo, emis, color = CLASS_METADATA.get(
                cls_name, ("Other", 0.20, 0.90, "#94a3b8")
            )

            classes_output.append({
                "id": cls_name,
                "label": label,
                "percentage": pct,
                "pixel_count": px_count,
                "polygons": polygons,
            })

            masks_output.append({
                "id": f"seg-{cls_name}",
                "className": cls_name,
                "label": label,
                "areaPercentage": pct,
                "pixelCount": px_count,
                "albedo": albedo,
                "emissivity": emis,
                "color": color,
                "polygonPoints": primary_polygon,  # For backward-compatible single polygon consumers
                "polygons": polygons,              # For multi-polygon SVG overlay
            })

        # Sort by areaPercentage descending
        classes_output.sort(key=lambda x: -x["percentage"])
        masks_output.sort(key=lambda x: -x["areaPercentage"])

        return {
            "model": {
                "name": self.model_name_or_path,
                "source": self.source,
                "device": str(self.device),
            },
            "image": {
                "width": orig_w,
                "height": orig_h,
            },
            "classes": classes_output,
            "source": "segformer",
            "masks": masks_output,
        }
