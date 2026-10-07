"""Spatial Inpainting Mask Builder for Autonomous Urban Redesign.

Converts SegFormer semantic perception masks and autonomous planner decisions into
geometrically grounded, spatially precise inpainting regions (binary PIL Image, mode 'L').
Avoids masking the entire image: strictly localized to intervention zones (tree planting
corridors, sidewalk pavers, road pavement, shade structures, rooftops).

Key design principles:
- 5 separate localized masks: tree_mask, sidewalk_mask, road_mask, shade_mask, roof_mask.
- Derived directly from SegFormer segmentation polygons wherever detected.
- Tree masks provide grounded planting corridors along sidewalk/curb verges plus realistic
  canopy airspace above for SDXL to paint full natural tree trunks and foliage.
- Strictly binary masks (0 = preserve original, 255 = inpaint/redesign) to ensure sharp diffusion targeting.
- Dilation support (expansion_level: 0, 1, 2) for retry escalation when greater change is needed.
- Full debug artifact export for visual inspection.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger("resilicity.mask_builder")


# ---------------------------------------------------------------------------
# Polygon Helpers
# ---------------------------------------------------------------------------

def _draw_polygon_pts(
    draw: ImageDraw.ImageDraw,
    polygon_pts: List[List[float]],
    width: int,
    height: int,
    fill: int = 255,
):
    """Draw a polygon given percentage coordinates [[x%, y%], ...]."""
    if len(polygon_pts) < 3:
        return
    pixel_pts = [
        (int(round(pt[0] / 100.0 * width)), int(round(pt[1] / 100.0 * height)))
        for pt in polygon_pts
    ]
    draw.polygon(pixel_pts, fill=fill)


def _extract_polygons(seg_result: Dict[str, Any]) -> Tuple[
    Dict[str, List[List[List[float]]]],
    Dict[str, float],
]:
    """Extract masks_by_class and area_pct_by_class from SegFormer output."""
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
    return masks_by_class, area_pct_by_class


def _dilate_mask(mask: Image.Image, radius: int = 12) -> Image.Image:
    """Expand white regions of an 'L' mask by radius pixels using MaxFilter."""
    arr = np.array(mask)
    if np.count_nonzero(arr) == 0:
        return mask
    expanded = mask.copy()
    passes = max(1, radius // 4)
    for _ in range(passes):
        expanded = expanded.filter(ImageFilter.MaxFilter(size=5))
    return expanded


# ---------------------------------------------------------------------------
# Individual Mask Builders (Task 2)
# ---------------------------------------------------------------------------

def build_tree_regions(
    width: int,
    height: int,
    masks_by_class: Dict[str, List[List[List[float]]]],
    area_pct_by_class: Dict[str, float],
    expansion_level: int = 0,
) -> Image.Image:
    """Build localized tree planting corridors with vertical trunk + canopy envelope.

    Grounded in SegFormer pavement polygons or road curb margins, providing sufficient
    surrounding spatial context for SDXL to paint full natural tree trunks and foliage.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    pave_polys = masks_by_class.get("pavement", [])
    road_polys = masks_by_class.get("road", [])
    has_pave = area_pct_by_class.get("pavement", 0.0) >= 2.0

    if has_pave and pave_polys:
        # Ground tree basins in the sidewalk polygons
        for poly in pave_polys:
            if len(poly) < 3:
                continue
            _draw_polygon_pts(draw, poly, width, height, fill=255)

            xs = [pt[0] / 100.0 * width for pt in poly]
            ys = [pt[1] / 100.0 * height for pt in poly]
            if not xs:
                continue

            poly_top = int(min(ys))
            min_x, max_x = int(min(xs)), int(max(xs))

            # Canopy airspace: upward from sidewalk top into sky/facade margin
            canopy_height = int(height * (0.35 + 0.08 * expansion_level))
            canopy_top = max(int(height * 0.04), poly_top - canopy_height)
            canopy_bottom = poly_top + 5

            if canopy_bottom > canopy_top and (max_x - min_x) > width * 0.03:
                draw.rectangle([min_x, canopy_top, max_x, canopy_bottom], fill=255)

    elif road_polys:
        # No distinct sidewalk: curbside planting corridors (left and right 18% margins)
        curb_width = 0.18 + 0.04 * expansion_level
        for side_frac in [0.0, max(0.70, 1.0 - curb_width)]:
            x_start = int(width * side_frac)
            x_end = int(width * (side_frac + curb_width))
            draw.rectangle([x_start, int(height * 0.45), x_end, height], fill=255)
            draw.rectangle([x_start, int(height * 0.06), x_end, int(height * 0.50)], fill=255)

    else:
        # Fallback curbside margins
        strip_w = max(int(width * (0.16 + 0.03 * expansion_level)), 60)
        for x_start in [0, width - strip_w]:
            draw.rectangle([x_start, int(height * 0.08), x_start + strip_w, height], fill=255)

    if expansion_level > 0:
        mask = _dilate_mask(mask, radius=12 * expansion_level)
    return mask


def build_sidewalk_edit_mask(
    width: int,
    height: int,
    masks_by_class: Dict[str, List[List[List[float]]]],
    expansion_level: int = 0,
) -> Image.Image:
    """Mask detected pedestrian pavement/sidewalk polygons for permeable pavers."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in masks_by_class.get("pavement", []):
        _draw_polygon_pts(draw, poly, width, height, fill=255)
    if expansion_level > 0:
        mask = _dilate_mask(mask, radius=8 * expansion_level)
    return mask


def build_road_surface_mask(
    width: int,
    height: int,
    masks_by_class: Dict[str, List[List[List[float]]]],
    expansion_level: int = 0,
) -> Image.Image:
    """Mask detected vehicular road surface polygons for cool pavement treatment."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in masks_by_class.get("road", []):
        _draw_polygon_pts(draw, poly, width, height, fill=255)
    if expansion_level > 0:
        mask = _dilate_mask(mask, radius=8 * expansion_level)
    return mask


def build_roof_mask(
    width: int,
    height: int,
    masks_by_class: Dict[str, List[List[List[float]]]],
    expansion_level: int = 0,
) -> Image.Image:
    """Mask detected rooftop surfaces for cool-roof or green-roof treatment."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in masks_by_class.get("roof", []):
        _draw_polygon_pts(draw, poly, width, height, fill=255)
    if expansion_level > 0:
        mask = _dilate_mask(mask, radius=8 * expansion_level)
    return mask


def build_shade_structure_mask(
    width: int,
    height: int,
    masks_by_class: Dict[str, List[List[List[float]]]],
    area_pct_by_class: Dict[str, float],
    expansion_level: int = 0,
) -> Image.Image:
    """Mask zone for tensile shade canopy over sidewalk/pedestrian areas."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    pave_polys = masks_by_class.get("pavement", [])
    if pave_polys and area_pct_by_class.get("pavement", 0.0) >= 3.0:
        largest_poly = max(
            pave_polys,
            key=lambda p: (
                (max(pt[0] for pt in p) - min(pt[0] for pt in p))
                * (max(pt[1] for pt in p) - min(pt[1] for pt in p))
            ) if len(p) >= 3 else 0,
        )
        xs = [pt[0] / 100.0 * width for pt in largest_poly]
        ys = [pt[1] / 100.0 * height for pt in largest_poly]
        if xs and ys:
            min_x, max_x = int(min(xs)), int(max(xs))
            min_y = int(min(ys))

            canopy_top = max(int(height * 0.12), min_y - int(height * (0.28 + 0.05 * expansion_level)))
            canopy_bottom = min_y + int(height * 0.06)
            draw.rectangle([min_x, canopy_top, max_x, canopy_bottom], fill=255)

            # Columns
            col_w = max(6, int(width * 0.015))
            ground_y = min(height - 5, min_y + int((max(ys) - min_y) * 0.5))
            draw.rectangle([min_x + 5, canopy_bottom, min_x + 5 + col_w, ground_y], fill=255)
            draw.rectangle([max_x - 5 - col_w, canopy_bottom, max_x - 5, ground_y], fill=255)
    else:
        # Central-verge shade canopy
        x1, x2 = int(width * 0.22), int(width * 0.78)
        y1, y2 = int(height * 0.18), int(height * 0.44)
        draw.rectangle([x1, y1, x2, y2], fill=255)

    if expansion_level > 0:
        mask = _dilate_mask(mask, radius=8 * expansion_level)
    return mask


# ---------------------------------------------------------------------------
# Composite Mask Builder (Main Entry Point)
# ---------------------------------------------------------------------------

MIN_COVERAGE_PCT = 12.0  # Minimum coverage to allow visible physical redesign


def build_inpainting_mask(
    image: Image.Image,
    seg_result: Dict[str, Any],
    interventions: List[Any],
    save_debug: bool = False,
    debug_dir: Optional[Path] = None,
    expansion_level: int = 0,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Build an architecturally constrained, composite inpainting mask.

    Generates all 5 individual localized masks (tree, sidewalk, road, shade, roof)
    and combines them based on the active planned interventions.

    Args:
        image: Original input photograph.
        seg_result: Structured segmentation result from SegFormerEngine.
        interventions: List of planned interventions (SpatialInterventionSpec or dicts).
        save_debug: If True, saves individual debug masks and original image.
        debug_dir: Directory where debug images will be saved.
        expansion_level: 0=normal, 1=expanded (+dilation), 2=aggressive expansion.

    Returns:
        Tuple of (combined_binary_mask, mask_metadata).
    """
    w, h = image.size
    total_px = w * h

    masks_by_class, area_pct_by_class = _extract_polygons(seg_result)

    # Active intervention types
    active_types = set()
    for iv in interventions:
        itype = getattr(iv, "type", None) or (iv.get("type") if isinstance(iv, dict) else str(iv))
        if itype:
            active_types.add(itype)

    # 1. Build all 5 individual localized masks (Task 2)
    tree_mask = build_tree_regions(w, h, masks_by_class, area_pct_by_class, expansion_level=expansion_level)
    sidewalk_mask = build_sidewalk_edit_mask(w, h, masks_by_class, expansion_level=expansion_level)
    road_mask = build_road_surface_mask(w, h, masks_by_class, expansion_level=expansion_level)
    shade_mask = build_shade_structure_mask(w, h, masks_by_class, area_pct_by_class, expansion_level=expansion_level)
    roof_mask = build_roof_mask(w, h, masks_by_class, expansion_level=expansion_level)

    individual_masks = {
        "tree_mask": tree_mask,
        "sidewalk_mask": sidewalk_mask,
        "road_mask": road_mask,
        "shade_mask": shade_mask,
        "roof_mask": roof_mask,
    }

    individual_coverages: Dict[str, float] = {}
    for name, m_img in individual_masks.items():
        arr = np.array(m_img)
        nz = int(np.count_nonzero(arr > 30))
        pct = round((nz / total_px) * 100.0, 2)
        individual_coverages[name] = pct
        logger.info("Individual mask %-15s: %5.2f%% (%d px)", name, pct, nz)

    # 2. Combine masks based on active planned interventions
    composite_np = np.zeros((h, w), dtype=np.uint8)
    applied_regions = []

    if "tree_canopy" in active_types or not active_types:
        composite_np = np.maximum(composite_np, np.array(tree_mask))
        applied_regions.append("tree_canopy")

    if "cool_pavement" in active_types:
        composite_np = np.maximum(composite_np, np.array(road_mask))
        applied_regions.append("cool_pavement")

    if "permeable_pave" in active_types:
        composite_np = np.maximum(composite_np, np.array(sidewalk_mask))
        applied_regions.append("permeable_pave")

    if "shade_structure" in active_types:
        composite_np = np.maximum(composite_np, np.array(shade_mask))
        applied_regions.append("shade_structure")

    if "cool_roof" in active_types or "green_roof" in active_types:
        composite_np = np.maximum(composite_np, np.array(roof_mask))
        applied_regions.append("rooftop")

    # 3. Minimum coverage safeguard: ensure at least MIN_COVERAGE_PCT
    covered_px = int(np.count_nonzero(composite_np > 30))
    coverage_pct = round((covered_px / total_px) * 100.0, 1)

    if coverage_pct < MIN_COVERAGE_PCT:
        logger.warning("Mask coverage %.1f%% < %.1f%% minimum. Adding supplementary regions.", coverage_pct, MIN_COVERAGE_PCT)
        if "road_cool_pavement" not in applied_regions and masks_by_class.get("road"):
            composite_np = np.maximum(composite_np, np.array(road_mask))
            applied_regions.append("road_supplement")
        if "pavement_permeable_pavers" not in applied_regions and masks_by_class.get("pavement"):
            composite_np = np.maximum(composite_np, np.array(sidewalk_mask))
            applied_regions.append("pavement_supplement")
        covered_px = int(np.count_nonzero(composite_np > 30))
        coverage_pct = round((covered_px / total_px) * 100.0, 1)

    if coverage_pct < MIN_COVERAGE_PCT:
        # Fallback curbside corridors
        logger.warning("Coverage still %.1f%%. Adding curbside corridor fallback.", coverage_pct)
        fallback = Image.new("L", (w, h), 0)
        fd = ImageDraw.Draw(fallback)
        strip_w = max(int(w * 0.20), 80)
        fd.rectangle([0, int(h * 0.08), strip_w, h], fill=255)
        fd.rectangle([w - strip_w, int(h * 0.08), w, h], fill=255)
        fd.rectangle([int(w * 0.20), int(h * 0.55), int(w * 0.80), h], fill=255)
        composite_np = np.maximum(composite_np, np.array(fallback))
        applied_regions.append("curbside_fallback")
        covered_px = int(np.count_nonzero(composite_np > 30))
        coverage_pct = round((covered_px / total_px) * 100.0, 1)

    # 4. Strictly binary output (0 or 255) with minimal edge blur
    composite_np = np.where(composite_np > 30, 255, 0).astype(np.uint8)
    composite_img = Image.fromarray(composite_np, mode="L")
    smoothed = composite_img.filter(ImageFilter.GaussianBlur(radius=2))
    smoothed_arr = np.where(np.array(smoothed) > 30, 255, 0).astype(np.uint8)
    final_mask = Image.fromarray(smoothed_arr, mode="L")

    final_covered_px = int(np.count_nonzero(smoothed_arr))
    final_coverage_pct = round((final_covered_px / total_px) * 100.0, 1)

    metadata = {
        "width": w,
        "height": h,
        "covered_pixels": final_covered_px,
        "coverage_percentage": final_coverage_pct,
        "applied_regions": applied_regions,
        "individual_coverages": individual_coverages,
        "individual_masks": individual_masks,
        "expansion_level": expansion_level,
    }

    logger.info(
        "Combined inpainting mask: %.1f%% coverage (%d px), regions=%s, expansion=%d",
        final_coverage_pct,
        final_covered_px,
        applied_regions,
        expansion_level,
    )

    # 5. Debug artifact saving (Task 6)
    if save_debug:
        out_dir = debug_dir or (Path(__file__).resolve().parent.parent / "data")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            image.save(out_dir / "debug_original_image.png")
            final_mask.save(out_dir / "debug_combined_mask.png")
            for m_name, m_img in individual_masks.items():
                m_img.save(out_dir / f"debug_{m_name}.png")
            logger.info("Saved debug masks to %s", out_dir)
        except Exception as e:
            logger.warning("Could not save debug mask artifacts: %s", e)

    return final_mask, metadata
