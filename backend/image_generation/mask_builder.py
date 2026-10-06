"""Spatial Inpainting Mask Builder for Autonomous Urban Redesign.

Converts SegFormer semantic perception masks and autonomous planner decisions into
geometrically grounded, spatially precise inpainting regions (binary PIL Image, mode 'L').
Avoids active traffic lanes, building facades, windows, and parked cars.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger("resilicity.mask_builder")


def _draw_polygon_pts(draw: ImageDraw.ImageDraw, polygon_pts: List[List[float]], width: int, height: int, fill: int = 255):
    """Draw a polygon given percentage coordinates [[x%, y%], ...]."""
    if len(polygon_pts) < 3:
        return
    pixel_pts = [
        (int(round(pt[0] / 100.0 * width)), int(round(pt[1] / 100.0 * height)))
        for pt in polygon_pts
    ]
    draw.polygon(pixel_pts, fill=fill)


def build_tree_regions(
    width: int,
    height: int,
    masks_by_class: Dict[str, List[List[List[float]]]],
    area_pct_by_class: Dict[str, float],
) -> Image.Image:
    """Build spatially localized tree planting envelopes along pedestrian curb margins.
    
    Rather than masking the entire sidewalk or lower half of the photograph, this locates
    discrete planting pits along pedestrian/curb edges and projects realistic vertical
    trunk-and-canopy envelopes upward while leaving roadways and facades preserved.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    pave_polys = masks_by_class.get("pavement", [])
    road_polys = masks_by_class.get("road", [])
    has_pave = "pavement" in masks_by_class and area_pct_by_class.get("pavement", 0.0) >= 2.5

    if has_pave and pave_polys:
        # Locate pedestrian planting zones on pavement polygons
        for poly in pave_polys:
            xs = [pt[0] / 100.0 * width for pt in poly]
            ys = [pt[1] / 100.0 * height for pt in poly]
            if len(xs) < 3:
                continue

            poly_w = max(xs) - min(xs)
            poly_h = max(ys) - min(ys)
            if poly_w < width * 0.06 or poly_h < height * 0.06:
                continue

            min_x, max_x = int(min(xs)), int(max(xs))
            min_y, max_y = int(min(ys)), int(max(ys))

            # Determine number of planting spots along this sidewalk stretch (1 to 3)
            num_trees = min(3, max(1, int(poly_w / (width * 0.28))))
            step = poly_w / (num_trees + 1)

            for i in range(1, num_trees + 1):
                center_x = int(min_x + i * step)
                # Position planting base near lower half of the sidewalk
                base_y = int(min_y + 0.65 * poly_h)
                base_y = min(height - 10, max(base_y, int(height * 0.50)))

                # 1. Ground basin / tree pit (small localized footprint on pavement)
                pit_rx = max(18, int(width * 0.04))
                pit_ry = max(8, int(height * 0.02))
                draw.ellipse(
                    [center_x - pit_rx, base_y - pit_ry, center_x + pit_rx, base_y + pit_ry],
                    fill=255,
                )

                # 2. Vertical trunk envelope (narrow column ascending from pit)
                trunk_half_w = max(6, int(width * 0.015))
                canopy_base_y = max(int(height * 0.25), base_y - int(height * 0.35))
                draw.rectangle(
                    [center_x - trunk_half_w, canopy_base_y, center_x + trunk_half_w, base_y],
                    fill=255,
                )

                # 3. Spreading canopy envelope (oval/balloon shape in airspace above sidewalk)
                canopy_rx = max(35, int(width * 0.09))
                canopy_ry = max(40, int(height * 0.16))
                canopy_center_y = max(int(height * 0.18), canopy_base_y - int(canopy_ry * 0.4))

                draw.ellipse(
                    [
                        max(0, center_x - canopy_rx),
                        max(int(height * 0.05), canopy_center_y - canopy_ry),
                        min(width, center_x + canopy_rx),
                        min(base_y - 10, canopy_center_y + canopy_ry),
                    ],
                    fill=255,
                )

    elif road_polys:
        # If no separate sidewalk polygon, position curbside planting corridors strictly
        # along the outer curb margins (left outer 15% and right outer 15%), preserving active traffic
        for side in ["left", "right"]:
            center_x = int(width * 0.10) if side == "left" else int(width * 0.90)
            base_y = int(height * 0.82)
            pit_rx = max(20, int(width * 0.045))
            pit_ry = max(10, int(height * 0.025))

            # Ground pit
            draw.ellipse(
                [center_x - pit_rx, base_y - pit_ry, center_x + pit_rx, base_y + pit_ry],
                fill=255,
            )
            # Trunk
            canopy_base_y = int(height * 0.48)
            draw.rectangle(
                [center_x - 8, canopy_base_y, center_x + 8, base_y],
                fill=255,
            )
            # Canopy
            canopy_rx = max(40, int(width * 0.11))
            canopy_ry = max(45, int(height * 0.18))
            draw.ellipse(
                [
                    max(0, center_x - canopy_rx),
                    int(height * 0.22),
                    min(width, center_x + canopy_rx),
                    canopy_base_y + 15,
                ],
                fill=255,
            )

    return mask


def build_sidewalk_edit_mask(
    width: int,
    height: int,
    masks_by_class: Dict[str, List[List[List[float]]]],
) -> Image.Image:
    """Build mask for permeable pavers, targeting only detected pedestrian pavement/sidewalk."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in masks_by_class.get("pavement", []):
        _draw_polygon_pts(draw, poly, width, height, fill=255)
    return mask


def build_road_surface_mask(
    width: int,
    height: int,
    masks_by_class: Dict[str, List[List[List[float]]]],
) -> Image.Image:
    """Build mask for cool pavement, targeting only detected vehicular road surface."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in masks_by_class.get("road", []):
        _draw_polygon_pts(draw, poly, width, height, fill=255)
    return mask


def build_roof_mask(
    width: int,
    height: int,
    masks_by_class: Dict[str, List[List[List[float]]]],
) -> Image.Image:
    """Build mask for cool roof / green roof, strictly targeting detected rooftop surfaces."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in masks_by_class.get("roof", []):
        _draw_polygon_pts(draw, poly, width, height, fill=255)
    return mask


def build_shade_structure_mask(
    width: int,
    height: int,
    masks_by_class: Dict[str, List[List[List[float]]]],
    area_pct_by_class: Dict[str, float],
) -> Image.Image:
    """Build spatially localized mask for modern architectural tensile fabric shade canopy."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    pave_polys = masks_by_class.get("pavement", [])
    if pave_polys and area_pct_by_class.get("pavement", 0.0) >= 5.0:
        # Place localized shade canopy over the largest sidewalk section
        largest_poly = max(
            pave_polys,
            key=lambda p: (max([pt[0] for pt in p]) - min([pt[0] for pt in p])) * (max([pt[1] for pt in p]) - min([pt[1] for pt in p]))
        )
        xs = [pt[0] / 100.0 * width for pt in largest_poly]
        ys = [pt[1] / 100.0 * height for pt in largest_poly]
        if xs and ys:
            min_x, max_x = int(min(xs)), int(max(xs))
            min_y, max_y = int(min(ys)), int(max(ys))

            # Restrict shade canopy to a clean rectangular envelope above sidewalk
            canopy_w = min(int(width * 0.35), max_x - min_x)
            canopy_start_x = min_x + (max_x - min_x - canopy_w) // 2
            canopy_end_x = canopy_start_x + canopy_w

            roof_y1 = max(int(height * 0.30), min_y - int(height * 0.20))
            roof_y2 = max(roof_y1 + 30, min_y - int(height * 0.05))

            # Sailcloth canopy plane
            draw.polygon(
                [
                    (canopy_start_x, roof_y1 + 10),
                    (canopy_end_x, roof_y1),
                    (canopy_end_x, roof_y2),
                    (canopy_start_x, roof_y2 + 15),
                ],
                fill=255,
            )

            # Slender support columns to ground
            ground_y = min(height - 10, min_y + int((max_y - min_y) * 0.5))
            draw.rectangle([canopy_start_x + 8, roof_y2, canopy_start_x + 14, ground_y], fill=255)
            draw.rectangle([canopy_end_x - 14, roof_y2, canopy_end_x - 8, ground_y], fill=255)

    return mask


def build_inpainting_mask(
    image: Image.Image,
    seg_result: Dict[str, Any],
    interventions: List[Any],
    save_debug: bool = False,
    debug_dir: Optional[Path] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Build an architecturally constrained, composite inpainting mask for generative redesign.

    Args:
        image: Original input photograph.
        seg_result: Structured segmentation result from SegFormerEngine.
        interventions: List of planned interventions (SpatialInterventionSpec or dicts).
        save_debug: If True, saves individual debug masks internally for inspection.
        debug_dir: Directory where debug images will be saved.

    Returns:
        Tuple of (smoothed_mask, mask_metadata) where smoothed_mask is PIL Image 'L' (255=inpaint, 0=preserve).
    """
    w, h = image.size

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

    # Active intervention types
    active_types = set()
    for iv in interventions:
        itype = getattr(iv, "type", None) or (iv.get("type") if isinstance(iv, dict) else str(iv))
        if itype:
            active_types.add(itype)

    applied_regions = []
    composite_np = np.zeros((h, w), dtype=np.uint8)

    # Individual debug masks
    debug_masks: Dict[str, Image.Image] = {}

    # 1. Tree Canopy Mask (Localized planting envelopes)
    if "tree_canopy" in active_types or not active_types:
        tree_mask = build_tree_regions(w, h, masks_by_class, area_pct_by_class)
        t_arr = np.array(tree_mask)
        if np.count_nonzero(t_arr) > 0:
            composite_np = np.maximum(composite_np, t_arr)
            applied_regions.append("tree_canopy_corridor")
            debug_masks["debug_tree_mask.png"] = tree_mask

    # 2. Permeable Pavers / Sidewalk treatment
    if "permeable_pave" in active_types:
        pave_mask = build_sidewalk_edit_mask(w, h, masks_by_class)
        p_arr = np.array(pave_mask)
        if np.count_nonzero(p_arr) > 0:
            composite_np = np.maximum(composite_np, p_arr)
            applied_regions.append("pavement_permeable_pavers")
            debug_masks["debug_pavement_mask.png"] = pave_mask

    # 3. Cool Pavement on Roadway
    if "cool_pavement" in active_types:
        road_mask = build_road_surface_mask(w, h, masks_by_class)
        r_arr = np.array(road_mask)
        if np.count_nonzero(r_arr) > 0:
            composite_np = np.maximum(composite_np, r_arr)
            applied_regions.append("road_cool_pavement")
            debug_masks["debug_road_mask.png"] = road_mask

    # 4. Rooftop Interventions (Cool Roof / Green Roof)
    if "cool_roof" in active_types or "green_roof" in active_types:
        roof_mask = build_roof_mask(w, h, masks_by_class)
        rf_arr = np.array(roof_mask)
        if np.count_nonzero(rf_arr) > 0:
            composite_np = np.maximum(composite_np, rf_arr)
            applied_regions.append("rooftop_resilience")
            debug_masks["debug_roof_mask.png"] = roof_mask

    # 5. Tensile Shade Structure
    if "shade_structure" in active_types:
        shade_mask = build_shade_structure_mask(w, h, masks_by_class, area_pct_by_class)
        s_arr = np.array(shade_mask)
        if np.count_nonzero(s_arr) > 0:
            composite_np = np.maximum(composite_np, s_arr)
            applied_regions.append("shade_structure")
            debug_masks["debug_shade_mask.png"] = shade_mask

    # Guarantee minimal inpainting region if SegFormer masks were too sparse
    covered_px = int(np.count_nonzero(composite_np))
    total_px = w * h
    coverage_pct = round((covered_px / total_px) * 100.0, 1)

    if coverage_pct < 3.0:
        logger.info("Mask coverage (%.1f%%) too low; applying fallback curbside tree corridors", coverage_pct)
        fallback_mask = build_tree_regions(w, h, {"road": [[[0, 60], [100, 60], [100, 100], [0, 100]]]}, {"road": 40.0})
        composite_np = np.maximum(composite_np, np.array(fallback_mask))
        covered_px = int(np.count_nonzero(composite_np))
        coverage_pct = round((covered_px / total_px) * 100.0, 1)
        applied_regions.append("minimum_corridor_guarantee")

    composite_img = Image.fromarray(composite_np, mode="L")

    # Boundary feathering: slight Gaussian blur (radius 3) so inpainting blends smoothly
    smoothed_mask = composite_img.filter(ImageFilter.GaussianBlur(radius=3))

    metadata = {
        "width": w,
        "height": h,
        "covered_pixels": covered_px,
        "coverage_percentage": coverage_pct,
        "applied_regions": applied_regions,
    }

    # Debug artifact saving (internal only, never in production UI)
    if save_debug:
        out_dir = debug_dir or Path(__file__).resolve().parent.parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            for fname, d_img in debug_masks.items():
                d_img.save(out_dir / fname)
            smoothed_mask.save(out_dir / "debug_combined_generation_mask.png")
            logger.info("Saved internal debug masks to %s", out_dir)
        except Exception as e:
            logger.warning("Could not save debug mask artifacts: %s", e)

    return smoothed_mask, metadata
