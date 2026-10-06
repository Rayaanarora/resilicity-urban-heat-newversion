"""Spatial Inpainting Mask Builder for Autonomous Urban Redesign.

Converts SegFormer semantic perception masks and planner decisions into
geometrically grounded inpainting regions (binary PIL Image, mode 'L').
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger("resilicity.mask_builder")


def _draw_polygon(draw: ImageDraw.ImageDraw, polygon_pts: List[List[float]], width: int, height: int, fill: int = 255):
    """Draw a polygon given percentage coordinates [[x%, y%], ...]."""
    if len(polygon_pts) < 3:
        return
    pixel_pts = [
        (int(round(pt[0] / 100.0 * width)), int(round(pt[1] / 100.0 * height)))
        for pt in polygon_pts
    ]
    draw.polygon(pixel_pts, fill=fill)


def build_inpainting_mask(
    image: Image.Image,
    seg_result: Dict[str, Any],
    interventions: List[Any],
    save_debug: bool = False,
    debug_dir: Optional[Path] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Build an architecturally constrained inpainting mask for generative redesign.

    Args:
        image: Original input photograph.
        seg_result: Structured segmentation result from SegFormerEngine.
        interventions: List of planned interventions (SpatialInterventionSpec or dicts).
        save_debug: If True, saves debug images for development inspection.
        debug_dir: Directory where debug images will be saved.

    Returns:
        Tuple of (mask_image, mask_metadata) where mask_image is PIL Image 'L' (255=inpaint, 0=preserve).
    """
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    # Extract masks by class name from SegFormer result
    masks_by_class: Dict[str, List[List[List[float]]]] = {}
    area_pct_by_class: Dict[str, float] = {}

    for m in seg_result.get("masks", []):
        cls_name = m.get("className", "")
        area_pct_by_class[cls_name] = m.get("areaPercentage", 0.0)
        polys = m.get("polygons", [])
        if not polys and m.get("polygonPoints"):
            polys = [m["polygonPoints"]]
        if cls_name and polys:
            masks_by_class.setdefault(cls_name, []).extend(polys)

    # Map intervention types to target surface classes
    active_types = set()
    for iv in interventions:
        itype = getattr(iv, "type", None) or (iv.get("type") if isinstance(iv, dict) else str(iv))
        if itype:
            active_types.add(itype)

    has_pavement = "pavement" in masks_by_class and area_pct_by_class.get("pavement", 0) > 2.0
    has_road = "road" in masks_by_class and area_pct_by_class.get("road", 0) > 5.0
    has_roof = "roof" in masks_by_class and area_pct_by_class.get("roof", 0) > 1.5

    applied_regions = []

    # 1. Permeable Pavers / Sidewalk treatment
    if "permeable_pave" in active_types and has_pavement:
        for poly in masks_by_class["pavement"]:
            _draw_polygon(draw, poly, w, h, fill=255)
        applied_regions.append("pavement_permeable_pavers")

    # 2. Cool Pavement on Roadway
    if "cool_pavement" in active_types and has_road:
        for poly in masks_by_class["road"]:
            _draw_polygon(draw, poly, w, h, fill=255)
        applied_regions.append("road_cool_pavement")

    # 3. Cool Roof or Green Roof on visible rooftops
    if ("cool_roof" in active_types or "green_roof" in active_types) and has_roof:
        for poly in masks_by_class["roof"]:
            _draw_polygon(draw, poly, w, h, fill=255)
        applied_regions.append("rooftop_resilience")

    # 4. Tree Canopy Expansion
    # Mature street trees require planting zones (sidewalk / roadside verges) and vertical canopy space
    if "tree_canopy" in active_types or not applied_regions:
        if has_pavement:
            # Mask pavement areas and extend vertical canopy envelope upward along pedestrian edges
            for poly in masks_by_class["pavement"]:
                _draw_polygon(draw, poly, w, h, fill=255)

            # Upward canopy expansion over pedestrian margins
            for poly in masks_by_class["pavement"]:
                xs = [pt[0] / 100.0 * w for pt in poly]
                ys = [pt[1] / 100.0 * h for pt in poly]
                if xs and ys:
                    min_x, max_x = max(0, int(min(xs))), min(w, int(max(xs)))
                    min_y = max(0, int(min(ys) - h * 0.40))  # Extend canopy up
                    base_y = min(h, int(max(ys)))
                    # Draw canopy corridor above sidewalk
                    draw.rectangle([min_x, min_y, max_x, base_y], fill=255)
            applied_regions.append("sidewalk_tree_canopy")
        elif has_road:
            # If no distinct pavement, plant along the outer flanks of the roadway (curbside corridors)
            # Left curb verge & canopy envelope
            draw.rectangle([0, int(h * 0.30), int(w * 0.30), int(h * 0.95)], fill=255)
            # Right curb verge & canopy envelope
            draw.rectangle([int(w * 0.70), int(h * 0.30), w, int(h * 0.95)], fill=255)
            applied_regions.append("roadside_verge_trees")
        else:
            # General street verge envelope (bottom 50% outer flanks)
            draw.rectangle([0, int(h * 0.40), int(w * 0.35), h], fill=255)
            draw.rectangle([int(w * 0.65), int(h * 0.40), w, h], fill=255)
            applied_regions.append("fallback_curb_planting")

    # 5. Tensile Shade Structure / Pergola
    if "shade_structure" in active_types:
        if has_pavement:
            for poly in masks_by_class["pavement"]:
                xs = [pt[0] / 100.0 * w for pt in poly]
                ys = [pt[1] / 100.0 * h for pt in poly]
                if xs and ys:
                    min_x, max_x = max(0, int(min(xs))), min(w, int(max(xs)))
                    min_y = max(0, int(min(ys) - h * 0.25))
                    max_y = min(h, int(max(ys)))
                    draw.rectangle([min_x, min_y, max_x, max_y], fill=255)
        else:
            draw.rectangle([int(w * 0.05), int(h * 0.45), int(w * 0.40), int(h * 0.85)], fill=255)
        applied_regions.append("shade_structure")

    # Ensure mask is not empty
    mask_np = np.array(mask)
    covered_px = int(np.count_nonzero(mask_np))
    total_px = w * h
    coverage_pct = round((covered_px / total_px) * 100.0, 1)

    if coverage_pct < 3.0:
        # Guarantee a meaningful inpainting region (outer roadside corridor)
        draw.rectangle([0, int(h * 0.40), int(w * 0.35), int(h * 0.95)], fill=255)
        draw.rectangle([int(w * 0.65), int(h * 0.40), w, int(h * 0.95)], fill=255)
        mask_np = np.array(mask)
        covered_px = int(np.count_nonzero(mask_np))
        coverage_pct = round((covered_px / total_px) * 100.0, 1)
        applied_regions.append("minimum_corridor_guarantee")

    # Smooth mask boundaries slightly so SDXL inpainting blends seamlessly into unmasked buildings/sky
    smoothed_mask = mask.filter(ImageFilter.GaussianBlur(radius=4))

    metadata = {
        "width": w,
        "height": h,
        "covered_pixels": covered_px,
        "coverage_percentage": coverage_pct,
        "applied_regions": applied_regions,
    }

    # Debug artifact saving (for development inspection only, internal)
    if save_debug:
        out_dir = debug_dir or Path(__file__).resolve().parent.parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            image.save(out_dir / "debug_original.png")
            smoothed_mask.save(out_dir / "debug_generation_mask.png")
            logger.info("Saved internal debug mask to: %s", out_dir / "debug_generation_mask.png")
        except Exception as e:
            logger.warning("Could not write debug mask images: %s", e)

    return smoothed_mask, metadata
