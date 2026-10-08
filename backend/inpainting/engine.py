"""Stage 4 (DEPRECATED): Legacy Procedural Inpainting Engine.

NOTE: This is a legacy procedural engine preserved for backwards-compatibility test cases.
The primary production autonomous urban redesign pipeline is located in
`backend/image_generation/multi_pass.py` and served via `/api/v1/analyze-and-redesign`.
Do NOT use this legacy procedural path for primary user uploads.
"""

import base64
import io
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def encode_image_to_base64(image: Image.Image, quality: int = 88) -> str:
    """Encode PIL Image to JPEG base64 data URI."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


class ResilientInpainter:
    """Photorealistic architectural inpainter for urban cooling interventions."""

    def __init__(self):
        assets_dir = HERE / "assets"
        self.tree_sprites: List[np.ndarray] = []
        for name in ["tree_1.png", "tree_2.png"]:
            p = assets_dir / name
            if p.exists():
                sprite = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if sprite is not None and sprite.ndim == 3 and sprite.shape[2] == 4:
                    self.tree_sprites.append(sprite)

    def inpaint(
        self,
        image: Image.Image,
        polygons_by_class: Dict[str, List[List[Tuple[float, float]]]],
        interventions: List[Dict[str, Any]],
    ) -> Image.Image:
        """Apply active interventions strictly bounded by segmentation polygons."""
        w, h = image.size
        np_img = np.array(image.convert("RGB")).astype(np.float32)

        # Standardize polygon keys so any class label / id matches without duplicate loops
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
            clean_k = str(k).strip().lower()
            norm_k = alias_map.get(clean_k, clean_k)
            # Add uniquely to normalized key
            existing = norm_polys.setdefault(norm_k, [])
            for poly in v:
                existing.append(poly)

        # Rendering order: ground coatings first, then paving, roofs, shade structures, and trees last
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
            # Handle camelCase, snake_case, and target aliases
            raw_target = item.get("targetRegion") or item.get("target_region") or item.get("target") or ""
            target_norm = alias_map.get(str(raw_target).strip().lower(), str(raw_target).strip().lower())
            coverage = float(item.get("coverage", 0.7))

            # Retrieve polygons matching target region
            matching_polys = norm_polys.get(target_norm, [])

            if not matching_polys:
                if itype in ("cool_roof", "green_roof"):
                    matching_polys = norm_polys.get("roof", [])
                elif itype == "cool_pavement":
                    matching_polys = norm_polys.get("road", []) or norm_polys.get("pavement", [])
                elif itype == "permeable_pave":
                    matching_polys = norm_polys.get("pavement", []) or norm_polys.get("road", [])
                elif itype in ("tree_canopy", "shade_structure"):
                    matching_polys = norm_polys.get("pavement", []) + norm_polys.get("road", [])

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

    def _apply_cool_pavement(
        self, img: np.ndarray, polys: List[List[Tuple[float, float]]], w: int, h: int, cov: float
    ) -> np.ndarray:
        """High-albedo solar-reflective pavement sealcoat.
        
        Transforms weathered dark asphalt to clean light-gray architectural surfacing
        while preserving all lane markings, cracks, curb lines, and lighting.
        """
        if not polys:
            return img
        mask = self._create_polygon_mask(polys, w, h, blur_k=5) * cov
        if np.max(mask) < 0.05:
            return img

        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Convert to Lab to manipulate luminance independently of color/edges
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[:, :, 0]
        
        # High albedo solar sealcoat: lift luminance significantly while preserving contrast
        # Typical asphalt L is 40-70; cool coating brings it to 140-185
        boosted_L = np.clip(L * 1.35 + 45.0, 0.0, 235.0)
        
        # Desaturate pavement a/b channels toward neutral cool concrete grey
        new_a = lab[:, :, 1] * 0.4 + 128.0 * 0.6
        new_b = lab[:, :, 2] * 0.35 + 128.0 * 0.65

        coated_lab = np.stack([boosted_L, new_a, new_b], axis=2)
        coated_rgb = cv2.cvtColor(np.clip(coated_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

        # Blend with alpha proportional to coverage (clear, visible light-gray road)
        alpha = mask_3d * 0.70
        return img * (1.0 - alpha) + coated_rgb * alpha

    def _apply_permeable_pavement(
        self, img: np.ndarray, polys: List[List[Tuple[float, float]]], w: int, h: int, cov: float
    ) -> np.ndarray:
        """Permeable interlocking concrete / stone pavers with drainage joints."""
        if not polys:
            return img
        mask = self._create_polygon_mask(polys, w, h, blur_k=3) * cov
        if np.max(mask) < 0.05:
            return img

        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Generate interlocking paver grid
        paver_size = max(10, min(w, h) // 45)
        y_grid, x_grid = np.indices((h, w))
        
        # Staggered brick joint pattern
        row = y_grid // paver_size
        col = (x_grid + (row % 2) * (paver_size // 2)) // (paver_size * 2)
        
        joint_y = (y_grid % paver_size == 0)
        joint_x = ((x_grid + (row % 2) * (paver_size // 2)) % (paver_size * 2) == 0)
        joints = (joint_y | joint_x).astype(np.float32)
        joints = cv2.GaussianBlur(joints, (3, 3), 0)

        # Paver stone colors: warm natural interlocking stone
        stone_base = np.array([178.0, 172.0, 162.0], dtype=np.float32)
        joint_color = np.array([80.0, 75.0, 70.0], dtype=np.float32)

        # Paver surface with bevel joint shading
        paver_tex = stone_base * (1.0 - joints[:, :, np.newaxis] * 0.45) + joint_color * (joints[:, :, np.newaxis] * 0.45)

        # Modulate with original underlying lighting/shadows
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        lum_mod = gray / (np.mean(gray) + 1e-5)
        lum_mod = np.clip(lum_mod, 0.75, 1.25)
        paver_surface = paver_tex * lum_mod[:, :, np.newaxis]

        alpha = mask_3d * 0.65
        return img * (1.0 - alpha) + paver_surface * alpha

    def _apply_cool_roof(
        self, img: np.ndarray, polys: List[List[Tuple[float, float]]], w: int, h: int, cov: float
    ) -> np.ndarray:
        """High-albedo reflective titanium cool roof coating."""
        if not polys:
            return img
        mask = self._create_polygon_mask(polys, w, h, blur_k=3) * cov
        if np.max(mask) < 0.05:
            return img

        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Lift roof brightness into clean reflective white/off-white (albedo ~0.80)
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * 1.4 + 55.0, 0.0, 245.0)
        lab[:, :, 1] = lab[:, :, 1] * 0.3 + 128.0 * 0.7
        lab[:, :, 2] = lab[:, :, 2] * 0.3 + 128.0 * 0.7

        brightened = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        alpha = mask_3d * 0.75
        return img * (1.0 - alpha) + brightened * alpha

    def _apply_green_roof(
        self, img: np.ndarray, polys: List[List[Tuple[float, float]]], w: int, h: int, cov: float
    ) -> np.ndarray:
        """Extensive vegetative sedum succulent green roof."""
        if not polys:
            return img
        mask = self._create_polygon_mask(polys, w, h, blur_k=5) * cov
        if np.max(mask) < 0.05:
            return img

        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Micro-vegetation sedum mat
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 18, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (5, 5), 0)

        sedum_base = np.array([52.0, 122.0, 58.0], dtype=np.float32)
        sedum_layer = np.clip(sedum_base + noise[:, :, np.newaxis], 0, 255)

        alpha = mask_3d * 0.70
        return img * (1.0 - alpha) + sedum_layer * alpha

    def _apply_tree_canopy(
        self, img: np.ndarray, polys: List[List[Tuple[float, float]]], w: int, h: int, cov: float
    ) -> np.ndarray:
        """Urban tree canopy expansion: renders realistic leafy canopy crowns along sidewalks

        and roads, combined with deep dappled ground cast shadows and tree planter basins.
        """
        if not polys:
            return img

        ground_mask = self._create_polygon_mask(polys, w, h, blur_k=5)
        if np.max(ground_mask) < 0.05:
            return img

        out = img.copy()

        # 1. Cast Dappled Leaf Shadows onto the Ground
        # Multi-frequency organic shadow pattern simulating canopy filtering
        rng = np.random.default_rng(2026)
        noise_large = cv2.resize(rng.uniform(0.0, 1.0, (max(4, h // 24), max(4, w // 24))).astype(np.float32), (w, h))
        noise_small = cv2.resize(rng.uniform(0.0, 1.0, (max(8, h // 10), max(8, w // 10))).astype(np.float32), (w, h))
        dappled = (noise_large * 0.65 + noise_small * 0.35)
        dappled = cv2.GaussianBlur(dappled, (17, 17), 0)

        # Shadow clusters with sharp sunspeckle falloff
        shadow_intensity = np.clip((dappled - 0.30) * 2.8, 0.0, 1.0)
        # Ground shadow darkens pavement by 35%
        ground_shadow = ground_mask * shadow_intensity * (cov * 0.42)
        shadow_3d = np.repeat(ground_shadow[:, :, np.newaxis], 3, axis=2)
        out = out * (1.0 - shadow_3d)

        # 2. Place Real Photographic Trees along the Sidewalk / Corridor
        if self.tree_sprites:
            y_indices, x_indices = np.where(ground_mask > 0.25)
            if len(x_indices) > 50:
                # Number of trees proportional to coverage (2 to 4 trees)
                num_trees = max(2, min(5, int(round(3.5 * cov))))

                min_x = np.percentile(x_indices, 8)
                max_x = np.percentile(x_indices, 92)
                x_targets = np.linspace(min_x, max_x, num_trees + 2)[1:-1]

                tree_sites = []
                for xt in x_targets:
                    col_mask = (np.abs(x_indices - xt) < (w * 0.12))
                    if np.any(col_mask):
                        col_y = y_indices[col_mask]
                        plant_y = int(np.percentile(col_y, 45 + rng.integers(-8, 12)))
                        plant_x = int(xt + rng.integers(-10, 10))
                        tree_sites.append((plant_x, plant_y))

                # Render background trees first (lower Y), foreground trees in front (higher Y)
                tree_sites.sort(key=lambda pt: pt[1])

                for idx, (cx, cy) in enumerate(tree_sites):
                    sprite = self.tree_sprites[idx % len(self.tree_sprites)]
                    sh, sw = sprite.shape[:2]

                    # Perspective scaling based on planting depth in frame
                    y_ratio = np.clip(cy / float(h), 0.35, 0.95)
                    tree_h = int(h * (0.35 + 0.40 * y_ratio) * np.clip(cov + 0.2, 0.7, 1.25))
                    tree_w = int(tree_h * (sw / float(sh)))

                    resized = cv2.resize(sprite, (tree_w, tree_h), interpolation=cv2.INTER_AREA)

                    # Soft realistic ground contact shadow beneath tree trunk
                    shadow_rw = max(12, tree_w // 3)
                    shadow_rh = max(4, tree_h // 14)
                    shadow_mask = np.zeros((h, w), dtype=np.float32)
                    cv2.ellipse(
                        shadow_mask,
                        (cx, min(h - 1, cy + shadow_rh // 2)),
                        (shadow_rw, shadow_rh),
                        0, 0, 360,
                        0.55 * cov,
                        -1,
                    )
                    shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
                    out = out * (1.0 - shadow_mask[:, :, np.newaxis])

                    # Tree trunk base sits at (cx, cy)
                    x0 = cx - tree_w // 2
                    y0 = cy - tree_h + int(tree_h * 0.04)
                    x1 = x0 + tree_w
                    y1 = y0 + tree_h

                    # Clip to canvas bounds
                    src_x0 = max(0, -x0)
                    src_y0 = max(0, -y0)
                    src_x1 = tree_w - max(0, x1 - w)
                    src_y1 = tree_h - max(0, y1 - h)

                    dst_x0 = max(0, x0)
                    dst_y0 = max(0, y0)
                    dst_x1 = min(w, x1)
                    dst_y1 = min(h, y1)

                    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
                        crop_sprite = resized[src_y0:src_y1, src_x0:src_x1]
                        # BGRA -> RGB
                        sprite_rgb = crop_sprite[:, :, :3][:, :, [2, 1, 0]].astype(np.float32)
                        sprite_alpha = (crop_sprite[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

                        target_roi = out[dst_y0:dst_y1, dst_x0:dst_x1]
                        blended_roi = target_roi * (1.0 - sprite_alpha) + sprite_rgb * sprite_alpha
                        out[dst_y0:dst_y1, dst_x0:dst_x1] = blended_roi

        return out

    def _apply_shade_structures(
        self, img: np.ndarray, polys: List[List[Tuple[float, float]]], w: int, h: int, cov: float
    ) -> np.ndarray:
        """Modern tensile fabric shade structure casting clean directional shade on ground."""
        if not polys:
            return img
        ground_mask = self._create_polygon_mask(polys, w, h, blur_k=3)
        if np.max(ground_mask) < 0.05:
            return img

        out = img.copy()

        # Angled geometric shade pattern cast on ground
        x_coords = np.tile(np.linspace(0, 1, w), (h, 1)).astype(np.float32)
        y_coords = np.tile(np.linspace(0, 1, h)[:, np.newaxis], (1, w)).astype(np.float32)
        shade_slats = np.sin((x_coords * 22.0 + y_coords * 12.0) * math.pi) * 0.5 + 0.5
        shade_slats = cv2.GaussianBlur(shade_slats, (5, 5), 0)

        ground_shade = ground_mask * shade_slats * (cov * 0.40)
        shade_3d = np.repeat(ground_shade[:, :, np.newaxis], 3, axis=2)
        out = out * (1.0 - shade_3d)

        # Overhead tensile fabric canopy sails
        sail_layer = np.zeros((h, w, 4), dtype=np.float32)
        y_indices, x_indices = np.where(ground_mask > 0.4)
        if len(x_indices) > 50:
            center_x = int(np.mean(x_indices))
            center_y = int(np.mean(y_indices) * 0.75)
            span_x = int(w * 0.22 * cov)
            span_y = int(h * 0.12 * cov)

            # Triangular tensile fabric canopy
            pts = np.array([
                [center_x - span_x, center_y],
                [center_x + span_x, center_y - span_y // 2],
                [center_x, center_y - span_y],
            ], dtype=np.int32)
            cv2.fillPoly(sail_layer, [pts], (235.0, 240.0, 245.0, 0.85))

        sail_alpha = sail_layer[:, :, 3]
        if np.max(sail_alpha) > 0.05:
            sail_alpha_3d = np.repeat(sail_alpha[:, :, np.newaxis], 3, axis=2)
            out = out * (1.0 - sail_alpha_3d) + sail_layer[:, :, :3] * sail_alpha_3d

        return out
