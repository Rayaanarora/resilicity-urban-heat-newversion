"""Stage 4: Resilient Urban Inpainting Engine.

Generates realistic photo-level before/after cooling transformations strictly bounded
by segmentation masks and urban resilience interventions:
- Tree Canopy Expansion: lush organic foliage crowns along curbs/sidewalks + dappled ground cast shadows + planter basins.
- Cool Pavement: high-albedo solar-reflective light-gray road sealcoat preserving lane markings and surface texture.
- Permeable Pavement: interlocking modular stone/concrete paver blocks with joint lines.
- Cool Roof: high-reflectance titanium-white / off-white cooling roof coating.
- Green Roof: vegetative sedum succulent mat with multi-hue organic micro-foliage.
- Shade Structures: modern tensile fabric canopies with crisp angled ground shadows.
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

        # 2. Place Leafy Canopy Clusters along Ground / Verge Boundaries
        # Find suitable planting anchor points along upper/edge bounds of ground
        ground_binary = (ground_mask > 0.3).astype(np.uint8) * 255
        contours, _ = cv2.findContours(ground_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Layer for foliage painting
        foliage_layer = np.zeros((h, w, 4), dtype=np.float32)

        # Tree crown radius scaled to image dimensions
        crown_rx = max(35, int(w * 0.12 * min(1.2, cov + 0.3)))
        crown_ry = max(30, int(h * 0.14 * min(1.2, cov + 0.3)))

        # Determine tree planting sites along sidewalk / road margins
        num_trees = max(2, int(round(4 * cov)))
        y_indices, x_indices = np.where(ground_mask > 0.4)
        
        if len(x_indices) > 50:
            # Sort by Y ascending to place trees along mid-ground / roadside edge
            min_y = np.percentile(y_indices, 15)
            max_y = np.percentile(y_indices, 70)
            eligible = (y_indices >= min_y) & (y_indices <= max_y)
            if np.sum(eligible) > 20:
                e_x = x_indices[eligible]
                e_y = y_indices[eligible]
                step = max(1, len(e_x) // num_trees)
                
                for t_idx in range(num_trees):
                    sel = min(len(e_x) - 1, t_idx * step + step // 2)
                    cx, cy = int(e_x[sel]), int(e_y[sel])

                    # Draw organic multi-layered tree crown
                    # Sub-crown 1: Deep shadow foliage interior
                    cv2.ellipse(
                        foliage_layer,
                        (cx, max(crown_ry, cy - crown_ry // 3)),
                        (crown_rx, crown_ry),
                        0, 0, 360,
                        (28.0, 72.0, 32.0, 0.90),
                        -1,
                    )
                    # Sub-crown 2: Mid-tone lush green leaf volume
                    cv2.ellipse(
                        foliage_layer,
                        (cx - crown_rx // 6, max(crown_ry, cy - crown_ry // 2)),
                        (int(crown_rx * 0.85), int(crown_ry * 0.85)),
                        -10, 0, 360,
                        (46.0, 118.0, 48.0, 0.92),
                        -1,
                    )
                    # Sub-crown 3: Sunlit crown highlight
                    cv2.ellipse(
                        foliage_layer,
                        (cx + crown_rx // 8, max(crown_ry, cy - int(crown_ry * 0.65))),
                        (int(crown_rx * 0.65), int(crown_ry * 0.65)),
                        15, 0, 360,
                        (82.0, 162.0, 72.0, 0.85),
                        -1,
                    )
                    # Tree trunk basin on ground
                    cv2.ellipse(
                        foliage_layer,
                        (cx, cy),
                        (max(4, crown_rx // 6), max(3, crown_ry // 10)),
                        0, 0, 360,
                        (45.0, 38.0, 32.0, 0.85),
                        -1,
                    )

        # Add organic leaf cluster fractal noise to foliage edges
        f_alpha = foliage_layer[:, :, 3]
        if np.max(f_alpha) > 0.05:
            f_alpha = cv2.GaussianBlur(f_alpha, (9, 9), 0)
            leaf_noise = cv2.resize(rng.uniform(0.75, 1.25, (h // 6, w // 6)).astype(np.float32), (w, h))
            leaf_noise = cv2.GaussianBlur(leaf_noise, (7, 7), 0)
            
            f_rgb = foliage_layer[:, :, :3] * leaf_noise[:, :, np.newaxis]
            f_alpha_3d = np.repeat(np.clip(f_alpha * 0.88, 0.0, 1.0)[:, :, np.newaxis], 3, axis=2)

            out = out * (1.0 - f_alpha_3d) + np.clip(f_rgb, 0, 255) * f_alpha_3d

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
