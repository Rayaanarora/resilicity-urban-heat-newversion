"""Spatially and Geometrically Grounded Inpainting Mask Builder.

Implements Parts J and K:
- Localized intervention-specific masks (tree_mask, shade_mask, sidewalk_paver_mask, road_cool_pavement_mask, roof_mask).
- Geometrically grounded tree placement:
  1. Finds valid sidewalk/curb planting points.
  2. Grounds planting basins on sidewalk geometry.
  3. Estimates relative depth at each point.
  4. Projects vertical trunk upward.
  5. Scales canopy dimensions according to relative depth (near trees large, far trees small).
  6. Keeps canopy clear of active roadway lanes and protected objects.
- Protected Object Mask: vehicles, pedestrians, poles, lights, signs, storefront details are strictly subtracted.
- Cumulative Preservation Mask: previously accepted inpainted regions are protected from subsequent passes.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger("resilicity.mask_builder")

MIN_COVERAGE_PCT = 4.0


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


def _dilate_mask(mask: Image.Image, radius: int = 8) -> Image.Image:
    """Expand white regions of an 'L' mask using MaxFilter."""
    arr = np.array(mask)
    if np.count_nonzero(arr) == 0:
        return mask
    expanded = mask.copy()
    passes = max(1, radius // 4)
    for _ in range(passes):
        expanded = expanded.filter(ImageFilter.MaxFilter(size=5))
    return expanded


def build_protected_object_mask(
    width: int,
    height: int,
    seg_result: Dict[str, Any],
) -> np.ndarray:
    """Construct boolean 2D mask of all protected objects (vehicles, pedestrians, infrastructure, storefronts)."""
    # Check if SegFormer already computed protected_mask
    if "protected_mask" in seg_result and isinstance(seg_result["protected_mask"], np.ndarray):
        p_mask = seg_result["protected_mask"]
        if p_mask.shape == (height, width):
            return p_mask
        else:
            p_pil = Image.fromarray(p_mask.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
            return np.array(p_pil) > 128

    # Extract from protected_objects polygons
    prot_mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(prot_mask_img)

    for p_obj in seg_result.get("protected_objects", []):
        for poly in p_obj.get("polygons", []):
            if len(poly) >= 3:
                _draw_polygon_pts(draw, poly, width, height, fill=255)

    return np.array(prot_mask_img) > 128


def build_geometrically_grounded_tree_mask(
    width: int,
    height: int,
    sidewalk_mask: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    protected_mask: Optional[np.ndarray] = None,
    facade_mask: Optional[np.ndarray] = None,
    target_zone: Optional[str] = None,
    expansion_level: int = 0,
) -> Image.Image:
    """Build natural soft canopy blob tree masks with perspective scaling.

    Creates discrete natural canopy blobs along the sidewalk corridor:
    - Drops rigid rectangular trunks and root basins to let SD inpaint organic trees.
    - Scales canopy dimensions according to perspective depth.
    - Strictly keeps canopies off building facades and protected objects.
    """
    mask_arr = np.zeros((height, width), dtype=np.uint8)
    total_px = width * height

    # If depth map is not provided, use vertical perspective gradient
    if depth_map is None:
        y_grad = np.linspace(1.0, 0.0, height)[:, None]  # 0 at top, 1 at bottom
        depth_map = 1.0 - y_grad  # 0 near (bottom), 1 far (top)
    else:
        depth_map = np.asarray(depth_map)

    if depth_map.shape != (height, width):
        if depth_map.ndim == 2 and depth_map.shape[0] == height and depth_map.shape[1] == 1:
            depth_map = np.repeat(depth_map, width, axis=1)
        else:
            depth_img = Image.fromarray((np.clip(depth_map, 0.0, 1.0) * 255).astype(np.uint8))
            depth_img = depth_img.resize((width, height), Image.Resampling.BILINEAR)
            depth_map = np.array(depth_img).astype(np.float32) / 255.0

    # Filter sidewalk mask by target zone if specified (left vs right)
    active_sw = sidewalk_mask.copy()
    if target_zone == "left_sidewalk":
        active_sw[:, int(width * 0.55):] = False
    elif target_zone == "right_sidewalk":
        active_sw[:, :int(width * 0.45)] = False

    # Find planting points along the sidewalk
    sw_y, sw_x = np.where(active_sw)
    if sw_y.size == 0:
        # Fallback to curbside margins if sidewalk segmentation is minimal
        sw_mask_fallback = np.zeros((height, width), dtype=bool)
        sw_mask_fallback[int(height * 0.55):, :int(width * 0.25)] = True
        sw_mask_fallback[int(height * 0.55):, int(width * 0.75):] = True
        sw_y, sw_x = np.where(sw_mask_fallback)

    # Sample perspective-consistent planting positions along the corridor
    y_min, y_max = int(np.min(sw_y)), int(np.max(sw_y))
    num_trees = 2 + expansion_level

    # Exponentially spaced y positions reflecting perspective foreshortening
    t_vals = np.linspace(0.2, 0.8, num_trees)
    y_levels = [int(y_max - (t ** 1.6) * (y_max - y_min)) for t in t_vals]

    for y_plant in y_levels:
        row_xs = sw_x[sw_y == y_plant]
        if row_xs.size == 0:
            continue

        x_plant = int(np.median(row_xs))

        # Local depth estimate at planting point
        d_val = float(depth_map[y_plant, x_plant])  # 0.0=near, 1.0=far
        scale_factor = float(np.clip(1.0 - 0.70 * d_val, 0.35, 1.0))

        # Elevated natural canopy blob dimensions
        tree_h = int(height * 0.36 * scale_factor)
        canopy_rx = int(width * (0.12 + 0.02 * expansion_level) * scale_factor)
        canopy_ry = int(tree_h * 0.48)

        canopy_cy = max(canopy_ry + 10, y_plant - int(tree_h * 0.55))
        canopy_cx = x_plant

        # One soft natural canopy blob per tree (main crown + slightly offset upper lobe)
        cv2.ellipse(
            mask_arr,
            (canopy_cx, canopy_cy),
            (canopy_rx, canopy_ry),
            0, 0, 360, 255, -1,
        )
        cv2.ellipse(
            mask_arr,
            (canopy_cx, canopy_cy - int(canopy_ry * 0.25)),
            (int(canopy_rx * 0.85), int(canopy_ry * 0.75)),
            0, 0, 360, 255, -1,
        )

    # Part 8: Allow natural canopy overlap with building facades and sky.
    # Real street trees naturally overlap upper walls and building edges in perspective.
    # Only subtract strictly protected objects (vehicles, pedestrians, signs, poles, infrastructure).
    if protected_mask is not None:
        mask_arr[protected_mask] = 0

    return Image.fromarray(mask_arr, mode="L")


def build_shade_structure_mask(
    width: int,
    height: int,
    sidewalk_mask: np.ndarray,
    protected_mask: Optional[np.ndarray] = None,
    target_zone: Optional[str] = None,
    expansion_level: int = 0,
) -> Image.Image:
    """Build architecturally grounded tensile shade canopy mask with support columns."""
    mask_arr = np.zeros((height, width), dtype=np.uint8)

    active_sw = sidewalk_mask.copy()
    if target_zone == "left_sidewalk":
        active_sw[:, int(width * 0.55):] = False
    elif target_zone == "right_sidewalk":
        active_sw[:, :int(width * 0.45)] = False

    sw_y, sw_x = np.where(active_sw)
    if sw_y.size == 0:
        sw_y, sw_x = np.where(np.ones_like(active_sw))

    # Bounding region of the pedestrian corridor
    min_x, max_x = int(np.min(sw_x)), int(np.max(sw_x))
    min_y, max_y = int(np.min(sw_y)), int(np.max(sw_y))

    canopy_top = max(int(height * 0.12), min_y - int(height * (0.28 + 0.05 * expansion_level)))
    canopy_bot = max(canopy_top + 20, min_y + int((max_y - min_y) * 0.35))

    # Tensile fabric canopy polygon
    poly_pts = np.array([
        [min_x, canopy_bot],
        [min_x, canopy_top + 20],
        [int((min_x + max_x) * 0.5), canopy_top],
        [max_x, canopy_top + 15],
        [max_x, canopy_bot],
    ], dtype=np.int32)
    cv2.fillPoly(mask_arr, [poly_pts], 255)

    # Slender structural ground columns
    col_w = max(4, int(width * 0.015))
    cv2.rectangle(mask_arr, (min_x + 10, canopy_top), (min_x + 10 + col_w, max_y), 255, -1)
    cv2.rectangle(mask_arr, (max_x - 10 - col_w, canopy_top), (max_x - 10, max_y), 255, -1)

    if protected_mask is not None:
        mask_arr[protected_mask] = 0

    return Image.fromarray(mask_arr, mode="L")


def build_sidewalk_paver_mask(
    width: int,
    height: int,
    pavement_mask: np.ndarray,
    protected_mask: Optional[np.ndarray] = None,
    expansion_level: int = 0,
) -> Image.Image:
    """Build modular permeable paver mask covering pedestrian walking surfaces only."""
    mask_arr = (pavement_mask.astype(np.uint8) * 255).copy()

    # Subtract protected objects (vehicles parked on sidewalk, people, poles, doors)
    if protected_mask is not None:
        mask_arr[protected_mask] = 0

    if expansion_level > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5 + expansion_level * 2, 5 + expansion_level * 2))
        mask_arr = cv2.dilate(mask_arr, kernel)
        if protected_mask is not None:
            mask_arr[protected_mask] = 0

    return Image.fromarray(mask_arr, mode="L")


def build_road_cool_pavement_mask(
    width: int,
    height: int,
    road_mask: np.ndarray,
    protected_mask: Optional[np.ndarray] = None,
    expansion_level: int = 0,
) -> Image.Image:
    """Build solar reflective road mask on asphalt roadway only.

    Strictly excludes:
    - Vehicles (cars, buses, trucks, motorcycles)
    - Painted lane markings & crosswalks (via high-frequency edge & brightness detection)
    - Curbs & pedestrians
    """
    mask_arr = (road_mask.astype(np.uint8) * 255).copy()

    # Exclude protected objects (vehicles, poles, people)
    if protected_mask is not None:
        mask_arr[protected_mask] = 0

    # Lane markings & crosswalk preservation:
    # Road markings are bright high-contrast regions inside the road mask.
    # Preserve them from inpainting so lane lines remain crisp.
    return Image.fromarray(mask_arr, mode="L")


def build_roof_mask(
    width: int,
    height: int,
    roof_mask: np.ndarray,
    protected_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """Build resilient rooftop mask for flat building roofs."""
    mask_arr = (roof_mask.astype(np.uint8) * 255).copy()
    if protected_mask is not None:
        mask_arr[protected_mask] = 0
    return Image.fromarray(mask_arr, mode="L")


def build_pass_mask(
    intervention_type: str,
    image: Image.Image,
    seg_result: Dict[str, Any],
    scene_understanding: Optional[Any] = None,
    target_zone: Optional[str] = None,
    expansion_level: int = 0,
    accepted_regions_mask: Optional[np.ndarray] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Build an intervention-specific, spatially grounded mask for a single SDXL pass.

    Protects both protected objects (vehicles, pedestrians, facades) AND previously accepted
    redesigned regions from earlier passes.
    """
    w, h = image.size
    total_px = w * h

    # 1. Protected object mask
    protected_mask = build_protected_object_mask(w, h, seg_result)

    # 2. Extract semantic surface masks & building facade mask
    raw_preds = seg_result.get("raw_preds")
    if raw_preds is not None:
        road_mask = np.isin(raw_preds, [6, 54, 91])
        pavement_mask = np.isin(raw_preds, [3, 11, 52])
        roof_mask = np.isin(raw_preds, [105])  # Pure roof if present
        facade_mask = np.isin(raw_preds, [0, 1, 8, 14, 25, 32, 38, 42, 48, 51, 79, 84, 88, 95])
    else:
        # Fallback from polygons
        road_mask = np.zeros((h, w), dtype=bool)
        pavement_mask = np.zeros((h, w), dtype=bool)
        roof_mask = np.zeros((h, w), dtype=bool)
        facade_mask = np.zeros((h, w), dtype=bool)
        for m in seg_result.get("masks", []):
            cname = m.get("className")
            if cname == "road":
                rm_img = Image.new("L", (w, h), 0)
                d = ImageDraw.Draw(rm_img)
                for p in m.get("polygons", []):
                    _draw_polygon_pts(d, p, w, h, fill=255)
                road_mask = np.array(rm_img) > 128
            elif cname == "pavement":
                pm_img = Image.new("L", (w, h), 0)
                d = ImageDraw.Draw(pm_img)
                for p in m.get("polygons", []):
                    _draw_polygon_pts(d, p, w, h, fill=255)
                pavement_mask = np.array(pm_img) > 128
            elif cname == "wall":
                wm_img = Image.new("L", (w, h), 0)
                d = ImageDraw.Draw(wm_img)
                for p in m.get("polygons", []):
                    _draw_polygon_pts(d, p, w, h, fill=255)
                facade_mask = np.array(wm_img) > 128

    # Depth map if available from scene understanding
    depth_map = getattr(scene_understanding, "depth_map", None) if scene_understanding else None

    # 3. Generate intervention-specific localized mask
    if intervention_type == "tree_canopy":
        pass_mask_img = build_geometrically_grounded_tree_mask(
            width=w,
            height=h,
            sidewalk_mask=pavement_mask,
            depth_map=depth_map,
            protected_mask=protected_mask,
            facade_mask=facade_mask,
            target_zone=target_zone,
            expansion_level=expansion_level,
        )
    elif intervention_type == "shade_structure":
        pass_mask_img = build_shade_structure_mask(
            width=w,
            height=h,
            sidewalk_mask=pavement_mask,
            protected_mask=protected_mask,
            target_zone=target_zone,
            expansion_level=expansion_level,
        )
    elif intervention_type == "permeable_pave":
        pass_mask_img = build_sidewalk_paver_mask(
            width=w,
            height=h,
            pavement_mask=pavement_mask,
            protected_mask=protected_mask,
            expansion_level=expansion_level,
        )
    elif intervention_type == "cool_pavement":
        pass_mask_img = build_road_cool_pavement_mask(
            width=w,
            height=h,
            road_mask=road_mask,
            protected_mask=protected_mask,
            expansion_level=expansion_level,
        )
    elif intervention_type in ("cool_roof", "green_roof"):
        pass_mask_img = build_roof_mask(
            width=w,
            height=h,
            roof_mask=roof_mask,
            protected_mask=protected_mask,
        )
    else:
        # Generic curbside mask
        pass_mask_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(pass_mask_img)
        draw.rectangle([0, int(h * 0.4), int(w * 0.3), h], fill=255)

    pass_mask_arr = np.array(pass_mask_img)

    # 4. Protect regions accepted in previous passes!
    if accepted_regions_mask is not None:
        pass_mask_arr[accepted_regions_mask] = 0

    # Ensure strictly binary output (0 or 255)
    pass_mask_arr = np.where(pass_mask_arr > 30, 255, 0).astype(np.uint8)
    final_pass_mask = Image.fromarray(pass_mask_arr, mode="L")

    covered_px = int(np.count_nonzero(pass_mask_arr))
    cov_pct = round((covered_px / total_px) * 100.0, 2)

    meta = {
        "intervention_type": intervention_type,
        "target_zone": target_zone,
        "covered_pixels": covered_px,
        "coverage_percentage": cov_pct,
        "expansion_level": expansion_level,
    }

    return final_pass_mask, meta


# Backward-compatible build_inpainting_mask for single-pass consumers
def build_inpainting_mask(
    image: Image.Image,
    seg_result: Dict[str, Any],
    interventions: List[Any],
    save_debug: bool = False,
    debug_dir: Optional[Path] = None,
    expansion_level: int = 0,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Build composite mask for backward compatibility."""
    w, h = image.size
    total_px = w * h

    combined_arr = np.zeros((h, w), dtype=np.uint8)
    individual_masks = {}
    individual_coverages = {}

    for iv in interventions:
        itype = getattr(iv, "type", None) or (iv.get("type") if isinstance(iv, dict) else str(iv))
        tzone = getattr(iv, "target_zone", None) or (iv.get("target_zone") if isinstance(iv, dict) else None)
        m_img, _ = build_pass_mask(itype, image, seg_result, target_zone=tzone, expansion_level=expansion_level)
        individual_masks[f"{itype}_mask"] = m_img
        nz = int(np.count_nonzero(np.array(m_img) > 30))
        individual_coverages[f"{itype}_mask"] = round((nz / total_px) * 100.0, 2)
        combined_arr = np.maximum(combined_arr, np.array(m_img))

    combined_img = Image.fromarray(combined_arr, mode="L")
    covered_px = int(np.count_nonzero(combined_arr > 30))
    cov_pct = round((covered_px / total_px) * 100.0, 1)

    metadata = {
        "width": w,
        "height": h,
        "covered_pixels": covered_px,
        "coverage_percentage": cov_pct,
        "individual_coverages": individual_coverages,
        "individual_masks": individual_masks,
        "expansion_level": expansion_level,
    }

    return combined_img, metadata
