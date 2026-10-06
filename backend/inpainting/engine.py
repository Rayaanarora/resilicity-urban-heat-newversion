"""Stage 4: Resilient Urban Inpainting Engine.

Generates realistic photo-level before/after cooling transformations strictly bounded
by segmentation masks and urban resilience interventions:
- Cool Roof / High-Albedo Coatings
- Green Roofs / Extensive Vegetative Sedum
- Cool Pavement / High-Albedo Surface Sealants
- Permeable Interlocking Pavers
- Tree Canopy Expansion with Realistic Shadows & Foliage
- Tensile Shade Structures
"""

import base64
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

HERE = Path(__file__).resolve().parent


class ResilientInpainter:
    """Procedural & depth-guided visual inpainter for urban cooling interventions."""

    def __init__(self):
        pass

    def inpaint(
        self,
        image: Image.Image,
        polygons_by_class: Dict[str, List[List[Tuple[float, float]]]],
        interventions: List[Dict[str, Any]],
    ) -> Image.Image:
        """Apply active interventions strictly bounded by segmentation polygons."""
        w, h = image.size
        # Work on float32 RGB array (0..255)
        np_img = np.array(image.convert("RGB")).astype(np.float32)

        # Standardize polygon keys so any class label / id matches
        norm_polys: Dict[str, List[List[Tuple[float, float]]]] = {}
        alias_map = {
            "asphalt / road": "road",
            "asphalt": "road",
            "road": "road",
            "street": "road",
            "seg-road": "road",
            "pavement": "pavement",
            "sidewalk": "pavement",
            "path": "pavement",
            "courtyard": "pavement",
            "seg-pavement": "pavement",
            "roof surfaces": "roof",
            "roof": "roof",
            "roofs": "roof",
            "seg-roof": "roof",
            "concrete / walls": "wall",
            "wall": "wall",
            "walls": "wall",
            "seg-wall": "wall",
            "vegetation": "vegetation",
            "veg": "vegetation",
            "seg-vegetation": "vegetation",
        }
        for k, v in polygons_by_class.items():
            clean_k = k.strip().lower()
            norm_k = alias_map.get(clean_k, clean_k)
            norm_polys.setdefault(norm_k, []).extend(v)
            norm_polys.setdefault(clean_k, []).extend(v)
            norm_polys.setdefault(k, []).extend(v)

        # Priority order of rendering: ground coatings first, then structures, then trees/canopies on top
        type_priority = {
            "cool_pavement": 1,
            "permeable_pave": 2,
            "cool_roof": 3,
            "green_roof": 4,
            "shade_structure": 5,
            "tree_canopy": 6,
        }
        sorted_interventions = sorted(
            interventions,
            key=lambda item: type_priority.get(item.get("type", ""), 0),
        )

        for item in sorted_interventions:
            itype = item.get("type")
            target = item.get("target_region") or item.get("target", "")
            target_norm = alias_map.get(str(target).strip().lower(), str(target).strip().lower())
            coverage = float(item.get("coverage", 0.7))

            # Retrieve polygons matching target region
            matching_polys = []
            if target_norm in norm_polys:
                matching_polys = norm_polys[target_norm]
            elif target in norm_polys:
                matching_polys = norm_polys[target]
            elif itype in ("cool_roof", "green_roof") and "roof" in norm_polys:
                matching_polys = norm_polys["roof"]
            elif itype == "cool_pavement" and "road" in norm_polys:
                matching_polys = norm_polys["road"]
            elif itype == "permeable_pave" and "pavement" in norm_polys:
                matching_polys = norm_polys["pavement"]
            elif itype == "tree_canopy":
                matching_polys = norm_polys.get("pavement", []) + norm_polys.get("road", [])

            # Fallback: if no road polygons found for cool_pavement, try pavement or vice-versa
            if not matching_polys:
                if itype in ("cool_pavement", "permeable_pave"):
                    matching_polys = norm_polys.get("pavement", []) or norm_polys.get("road", [])
                elif itype in ("cool_roof", "green_roof"):
                    matching_polys = norm_polys.get("roof", [])

            if itype == "cool_roof":
                np_img = self._apply_cool_roof(np_img, matching_polys, w, h, coverage)
            elif itype == "green_roof":
                np_img = self._apply_green_roof(np_img, matching_polys, w, h, coverage)
            elif itype == "cool_pavement":
                np_img = self._apply_cool_pavement(np_img, matching_polys, w, h, coverage)
            elif itype == "permeable_pave":
                np_img = self._apply_permeable_pavement(np_img, matching_polys, w, h, coverage)
            elif itype == "tree_canopy":
                np_img = self._apply_tree_canopy(np_img, matching_polys, w, h, coverage)
            elif itype == "shade_structure":
                np_img = self._apply_shade_structures(np_img, matching_polys, w, h, coverage)

        result_uint8 = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(result_uint8)

    def _create_polygon_mask(
        self, polys: List[List[Tuple[float, float]]], w: int, h: int, blur_k: int = 5
    ) -> np.ndarray:
        """Create a smooth binary float mask (0.0 to 1.0) from percentage polygons."""
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polys:
            if len(poly) < 3:
                continue
            pts = np.array(
                [[int(round((x / 100.0) * w)), int(round((y / 100.0) * h))] for x, y in poly],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts], 255)

        if blur_k > 0:
            mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
        return mask.astype(np.float32) / 255.0

    def _apply_cool_roof(
        self, img: np.ndarray, polys: List[List[Tuple[float, float]]], w: int, h: int, cov: float
    ) -> np.ndarray:
        """High-albedo cool roof: subtly lighten and desaturate roof surfaces."""
        if not polys:
            return img
        mask = self._create_polygon_mask(polys, w, h, blur_k=3) * cov
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Lighten the existing roof pixels (preserve texture, just raise brightness)
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.4   # desaturate
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + 50, 0, 255)  # brighten
        treated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        alpha = mask_3d * 0.45
        return img * (1.0 - alpha) + treated * alpha

    def _apply_green_roof(
        self, img: np.ndarray, polys: List[List[Tuple[float, float]]], w: int, h: int, cov: float
    ) -> np.ndarray:
        """Green roof: tint roof surfaces green while preserving texture."""
        if not polys:
            return img
        mask = self._create_polygon_mask(polys, w, h, blur_k=3) * cov
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Shift existing pixels toward green hue, preserve luminance structure
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = 60   # green hue
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + 80, 0, 255)  # boost saturation
        treated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        alpha = mask_3d * 0.35
        return img * (1.0 - alpha) + treated * alpha

    def _apply_cool_pavement(
        self, img: np.ndarray, polys: List[List[Tuple[float, float]]], w: int, h: int, cov: float
    ) -> np.ndarray:
        """Cool pavement: lighten road surfaces preserving all texture and markings."""
        if not polys:
            return img
        mask = self._create_polygon_mask(polys, w, h, blur_k=3) * cov
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Simply brighten and slightly desaturate — keeps lane markings, cracks, texture
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.5
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + 45, 0, 245)
        treated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        alpha = mask_3d * 0.40
        return img * (1.0 - alpha) + treated * alpha

    def _apply_permeable_pavement(
        self, img: np.ndarray, polys: List[List[Tuple[float, float]]], w: int, h: int, cov: float
    ) -> np.ndarray:
        """Permeable pavement: warm-lighten sidewalk surfaces."""
        if not polys:
            return img
        mask = self._create_polygon_mask(polys, w, h, blur_k=3) * cov
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Warm-shift and lighten existing pavement texture
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] * 0.5 + 15, 0, 179)  # nudge toward warm
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + 30, 0, 240)
        treated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        alpha = mask_3d * 0.30
        return img * (1.0 - alpha) + treated * alpha

    def _apply_tree_canopy(
        self, img: np.ndarray, polys: List[List[Tuple[float, float]]], w: int, h: int, cov: float
    ) -> np.ndarray:
        """Tree canopy: add dappled shade on ground surfaces and green verge at edges."""
        if not polys:
            return img
        ground_mask = self._create_polygon_mask(polys, w, h, blur_k=5)
        if np.max(ground_mask) < 0.05:
            return img

        # Dappled shade: subtle darkening patches simulating overhead leaf shadows
        rng = np.random.default_rng(2026)
        noise = cv2.resize(rng.uniform(0.0, 1.0, (h // 12, w // 12)).astype(np.float32), (w, h))
        noise = cv2.GaussianBlur(noise, (15, 15), 0)
        shade_spots = np.clip((noise - 0.4) * 2.5, 0.0, 1.0)
        shade_amount = ground_mask * shade_spots * cov * 0.20  # subtle 20% darkening max

        shade_3d = np.repeat(shade_amount[:, :, np.newaxis], 3, axis=2)
        result = img * (1.0 - shade_3d)

        # Thin green verge at ground mask edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        eroded = cv2.erode((ground_mask * 255).astype(np.uint8), kernel)
        edge = ((ground_mask * 255).astype(np.uint8) - eroded).astype(np.float32) / 255.0
        edge = cv2.GaussianBlur(edge, (7, 7), 0) * cov

        # Tint edge pixels green (preserve original brightness)
        hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        green_hsv = hsv.copy()
        green_hsv[:, :, 0] = 55  # green hue
        green_hsv[:, :, 1] = np.clip(green_hsv[:, :, 1] + 100, 0, 255)
        green_tinted = cv2.cvtColor(green_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        edge_3d = np.repeat(edge[:, :, np.newaxis], 3, axis=2) * 0.35
        return result * (1.0 - edge_3d) + green_tinted * edge_3d

    def _apply_shade_structures(
        self, img: np.ndarray, polys: List[List[Tuple[float, float]]], w: int, h: int, cov: float
    ) -> np.ndarray:
        """Shade structures: subtle geometric shadow pattern on ground."""
        if not polys:
            return img
        ground_mask = self._create_polygon_mask(polys, w, h, blur_k=3)
        if np.max(ground_mask) < 0.05:
            return img

        # Diagonal stripe shadow
        x = np.tile(np.linspace(0, 1, w), (h, 1)).astype(np.float32)
        y = np.tile(np.linspace(0, 1, h)[:, np.newaxis], (1, w)).astype(np.float32)
        stripes = (np.sin((x * 25 + y * 12) * math.pi) * 0.5 + 0.5)
        stripes = cv2.GaussianBlur(stripes, (5, 5), 0)

        shade = ground_mask * stripes * cov * 0.15  # very subtle 15%
        shade_3d = np.repeat(shade[:, :, np.newaxis], 3, axis=2)
        return img * (1.0 - shade_3d)



def encode_image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 92) -> str:
    """Encode PIL image to base64 data URI string."""
    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"
