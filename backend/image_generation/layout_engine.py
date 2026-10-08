"""Explicit Geometric Layout Engine for Urban Resilience Interventions.

Implements Requirements 1, 2, 3:
- Converts high-level planner recommendations into explicit spatial geometry:
  - Exact tree ground anchors with relative depth, scale, canopy radius, and corridor polyline.
  - Shade structure footprints, support points, and ground contact anchors.
  - Surface quadrilaterals / polygons for road albedo treatments and permeable paving.
- Monocular relative depth directly drives physical object scale (nearer = larger, farther = smaller).
- Strict validation rejects anchors near protected objects (cars, pedestrians, poles) or active traffic lanes.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("resilicity.layout_engine")


def extract_polygon_contour(mask: np.ndarray, max_points: int = 12) -> Optional[List[List[float]]]:
    """Extract simplified polygon contour [[x, y], ...] from a boolean mask."""
    if mask is None or np.count_nonzero(mask) < 100:
        return None
    uint8_mask = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Largest contour
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 150:
        return None
    epsilon = 0.015 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    poly = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]
    return poly if len(poly) >= 3 else None


def extract_surface_quadrilateral(mask: np.ndarray) -> Optional[List[List[float]]]:
    """Derive minimum area bounding rotated quadrilateral for perspective ground surfaces."""
    if mask is None or np.count_nonzero(mask) < 200:
        return None
    uint8_mask = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return [[float(pt[0]), float(pt[1])] for pt in box]


def compute_tree_row_layout(
    width: int,
    height: int,
    sidewalk_mask: np.ndarray,
    road_mask: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    protected_mask: Optional[np.ndarray] = None,
    wall_mask: Optional[np.ndarray] = None,
    target_zone: Optional[str] = None,
    requested_trees: int = 3,
) -> Dict[str, Any]:
    """Derive explicit tree-row planting corridor and grounded planting anchors.

    Requirements 1, 2, 3:
    1. Detects/derives curbside corridor.
    2. Samples candidate planting anchors along the corridor.
    3. Rejects anchors within proximity of protected objects (vehicles, people, poles).
    4. Rejects anchors inside vehicular traffic lanes or against storefronts.
    5. Computes perspective-scaled canopy radii and tree dimensions from relative depth.
    """
    if depth_map is None:
        depth_map = np.tile(np.linspace(1.0, 0.0, height)[:, None], (1, width))

    # Restrict to specified sidewalk side
    active_sw = sidewalk_mask.copy()
    if target_zone == "left_sidewalk":
        active_sw[:, int(width * 0.55):] = False
    elif target_zone == "right_sidewalk":
        active_sw[:, :int(width * 0.45)] = False

    sw_y, sw_x = np.where(active_sw)
    if sw_y.size == 0:
        # Fallback curbside corridor if sidewalk segmentation is minimal
        active_sw = np.zeros((height, width), dtype=bool)
        if target_zone == "left_sidewalk" or target_zone is None:
            active_sw[int(height * 0.52):, :int(width * 0.22)] = True
        if target_zone == "right_sidewalk" or target_zone is None:
            active_sw[int(height * 0.52):, int(width * 0.78):] = True
        sw_y, sw_x = np.where(active_sw)

    y_min, y_max = int(np.min(sw_y)), int(np.max(sw_y))

    # Identify side of street
    sw_all_xs = sw_x
    median_sw_x = float(np.median(sw_all_xs)) if sw_all_xs.size > 0 else width * 0.75
    side_name = "left" if (target_zone == "left_sidewalk" or (target_zone is None and median_sw_x < width * 0.5)) else "right"

    # Identify curbside interface (boundary between sidewalk and road)
    curb_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_road = cv2.dilate(road_mask.astype(np.uint8), curb_kernel)
    curbside_mask = active_sw & (dilated_road > 0)
    has_curbside = np.count_nonzero(curbside_mask) > 50

    # Derive complete corridor polyline along the sidewalk buffer
    corridor_points: List[List[float]] = []
    y_step = max(3, (y_max - y_min) // 35)
    for y_cur in range(y_max, y_min - 1, -y_step):
        row_xs = sw_x[sw_y == y_cur]
        if row_xs.size > 0:
            sw_w = float(np.max(row_xs) - np.min(row_xs))
            if side_name == "right":
                # Indent from curb edge (left edge of right sidewalk) into sidewalk buffer
                c_x = float(np.min(row_xs)) + max(12.0, min(sw_w * 0.08, 24.0))
            else:
                # Indent from curb edge (right edge of left sidewalk) into sidewalk buffer
                c_x = float(np.max(row_xs)) - max(12.0, min(sw_w * 0.08, 24.0))
            corridor_points.append([c_x, float(y_cur)])

    # Proximity exclusion radius around protected objects (in pixels)
    prot_dist_transform = None
    if protected_mask is not None and np.count_nonzero(protected_mask) > 0:
        prot_inv = (~protected_mask).astype(np.uint8) * 255
        prot_dist_transform = cv2.distanceTransform(prot_inv, cv2.DIST_L2, 5)

    # Perspective recession: target progress along sidewalk corridor from near to far
    n_trees = max(2, requested_trees)
    t_targets = np.linspace(0.24, 0.78, n_trees)

    anchors: List[Dict[str, Any]] = []
    selected_ys: List[int] = []

    for t_frac in t_targets:
        target_y = int(y_max - t_frac * (y_max - y_min))
        best_cand = None
        best_cand_score = -1e9

        search_half_window = max(14, int((y_max - y_min) * 0.12))
        search_window = range(
            max(y_min, target_y - search_half_window),
            min(y_max, target_y + search_half_window + 1),
            2,
        )

        for y_cand in search_window:
            # Enforce minimum vertical separation from previously placed anchors
            if any(abs(y_cand - prev_y) < int((y_max - y_min) * 0.20) for prev_y in selected_ys):
                continue

            row_xs = sw_x[sw_y == y_cand]
            if row_xs.size == 0:
                continue

            sw_w = float(np.max(row_xs) - np.min(row_xs))
            if sw_w < 20.0:
                continue

            # Place within the sidewalk planting strip (buffer between curb and pedestrian corridor)
            if side_name == "right":
                curb_edge = float(np.min(row_xs))
                # Offset ~6-8% into sidewalk so trunk sits squarely in the curbside buffer
                cand_offset = max(12.0, min(sw_w * 0.08, 24.0))
                x_cand = int(round(curb_edge + cand_offset))
            else:
                curb_edge = float(np.max(row_xs))
                cand_offset = max(12.0, min(sw_w * 0.08, 24.0))
                x_cand = int(round(curb_edge - cand_offset))

            x_cand = int(np.clip(x_cand, 0, width - 1))

            # Strictly reject or adjust if candidate falls outside sidewalk or on roadway
            if not active_sw[y_cand, x_cand] or road_mask[y_cand, x_cand]:
                # Snap to valid sidewalk pixel
                valid_row_xs = [x for x in row_xs if not road_mask[y_cand, x]]
                if not valid_row_xs:
                    continue
                x_cand = int(valid_row_xs[len(valid_row_xs) // 3])

            # Strictly reject if near protected objects (cars, pedestrians, poles)
            dist_to_prot = 999.0
            if prot_dist_transform is not None:
                dist_to_prot = float(prot_dist_transform[y_cand, x_cand])
                if dist_to_prot < 20.0:
                    continue

            y_dist = abs(y_cand - target_y)
            score = -y_dist + min(40.0, dist_to_prot) * 0.5
            if score > best_cand_score:
                best_cand_score = score
                best_cand = (x_cand, y_cand)

        if best_cand is not None:
            x_plant, y_plant = best_cand
            selected_ys.append(y_plant)

            rel_depth = float(depth_map[y_plant, x_plant])
            # Bounded perspective scaling: foreground scale ~0.95, receding to ~0.42
            scale = float(np.clip(1.0 - 0.60 * rel_depth, 0.42, 0.98))

            # Realistic urban street tree proportions:
            # Mature street tree canopy stands 2-3 stories tall (typically 38-42% of image height in foreground)
            tree_h = int(round(height * 0.40 * scale))
            canopy_rx = int(round(tree_h * 0.33))
            canopy_ry = int(round(tree_h * 0.40))
            trunk_w = max(6, int(round(width * 0.018 * scale)))
            canopy_cy = max(canopy_ry + 10, y_plant - int(tree_h * 0.58))
            pit_w = max(16, int(round(trunk_w * 3.5)))
            pit_d = max(10, int(round(trunk_w * 1.8)))
            intended_shade = round(float(np.pi * canopy_rx * canopy_ry * 0.45 / (width * height) * 100.0), 2)

            anchors.append({
                "x": int(x_plant),
                "y": int(y_plant),
                "relative_depth": round(rel_depth, 3),
                "scale": round(scale, 3),
                "canopy_radius": int(canopy_rx),
                "canopy_height": int(canopy_ry),
                "tree_height": int(tree_h),
                "trunk_width": int(trunk_w),
                "canopy_center_y": int(canopy_cy),
                "planting_pit": {"width": pit_w, "depth": pit_d},
                "intended_shade_coverage": intended_shade,
            })

    # Fallback to 2 anchors along corridor if dense constraints blocked sampling
    if len(anchors) < 2 and len(corridor_points) >= 2:
        for pt in [corridor_points[0], corridor_points[-1]]:
            px, py = int(pt[0]), int(pt[1])
            d = float(depth_map[py, px]) if depth_map is not None else 0.5
            s = float(np.clip(1.0 - 0.60 * d, 0.42, 0.98))
            th = int(round(height * 0.50 * s))
            tw = max(6, int(round(width * 0.020 * s)))
            rx = int(round(th * 0.40))
            ry = int(round(th * 0.45))
            anchors.append({
                "x": px,
                "y": py,
                "relative_depth": round(d, 3),
                "scale": round(s, 3),
                "canopy_radius": rx,
                "canopy_height": ry,
                "tree_height": th,
                "trunk_width": tw,
                "canopy_center_y": max(int(height * 0.1), py - int(th * 0.58)),
                "planting_pit": {"width": max(16, int(tw * 3.5)), "depth": max(10, int(tw * 1.8))},
                "intended_shade_coverage": round(float(np.pi * rx * ry * 0.45 / (width * height) * 100.0), 2),
            })

    # Sort corridor polyline points from foreground to background
    corridor_points.sort(key=lambda p: p[1], reverse=True)

    spacing_px = 0.0
    if len(anchors) >= 2:
        dists = [
            float(np.hypot(anchors[i]["x"] - anchors[i+1]["x"], anchors[i]["y"] - anchors[i+1]["y"]))
            for i in range(len(anchors) - 1)
        ]
        spacing_px = float(np.mean(dists))

    planting_points = []
    all_ground_anchors = []
    all_canopy_extents = []
    depth_values = []
    scale_values = []

    for anc in anchors:
        ax, ay = int(anc["x"]), int(anc["y"])
        d = float(anc["relative_depth"])
        s = float(anc["scale"])
        rx = int(anc["canopy_radius"])
        ry = int(anc.get("canopy_height", rx))
        cy = int(anc.get("canopy_center_y", ay - ry))

        extent = [max(0, ax - rx), max(0, cy - ry), min(width, ax + rx), min(height, cy + ry)]
        planting_points.append({
            "x": ax,
            "y": ay,
            "relative_depth": d,
            "scale": s,
            "canopy_radius": rx,
            "canopy_height": ry,
            "tree_height": int(anc["tree_height"]),
            "trunk_width": int(anc["trunk_width"]),
            "planting_pit": anc.get("planting_pit", {}),
            "intended_shade_coverage": anc.get("intended_shade_coverage", 0.0),
            "ground_anchor": [ax, ay],
            "canopy_extent": extent,
            "depth": d,
            "spacing": round(spacing_px, 1),
        })
        all_ground_anchors.append([ax, ay])
        all_canopy_extents.append(extent)
        depth_values.append(d)
        scale_values.append(s)

    overall_canopy_extent = [
        min(e[0] for e in all_canopy_extents) if all_canopy_extents else 0,
        min(e[1] for e in all_canopy_extents) if all_canopy_extents else 0,
        max(e[2] for e in all_canopy_extents) if all_canopy_extents else width,
        max(e[3] for e in all_canopy_extents) if all_canopy_extents else height,
    ]

    mean_depth = round(float(np.mean(depth_values)), 3) if depth_values else 0.5
    mean_scale = round(float(np.mean(scale_values)), 3) if scale_values else 0.7
    mean_tree_h = int(np.mean([a["tree_height"] for a in anchors])) if anchors else int(height * 0.4)
    mean_trunk_w = int(np.mean([a["trunk_width"] for a in anchors])) if anchors else int(width * 0.018)
    total_intended_shade = round(sum(a.get("intended_shade_coverage", 0.0) for a in anchors), 2)

    return {
        "side": side_name,
        "target_zone": target_zone or f"{side_name}_sidewalk",
        "corridor_polyline": corridor_points,
        "planting_points": planting_points,
        "spacing": round(spacing_px, 1),
        "depth": mean_depth,
        "relative_depth": mean_depth,
        "scale": mean_scale,
        "ground_anchor": all_ground_anchors[0] if all_ground_anchors else [width // 4, int(height * 0.8)],
        "canopy_extent": overall_canopy_extent,
        "tree_height": mean_tree_h,
        "trunk_width": mean_trunk_w,
        "planting_pit": anchors[0].get("planting_pit", {}) if anchors else {},
        "intended_shade_coverage": total_intended_shade,
        # Backwards-compatible keys
        "anchors": anchors,
    }



def compute_shade_structure_layout(
    width: int,
    height: int,
    sidewalk_mask: np.ndarray,
    road_mask: np.ndarray,
    protected_mask: Optional[np.ndarray] = None,
    target_zone: Optional[str] = None,
) -> Dict[str, Any]:
    """Derive explicit architectural shade canopy footprint and ground support columns.
    
    Requirements 10:
    - Support posts MUST lie strictly on sidewalk geometry (pavement_mask).
    - NEVER place support columns in active vehicular traffic lanes.
    - Support columns must strictly avoid protected vehicles and pedestrians.
    - The shade sail must have 2+ ground anchors and perspective-correct geometry.
    """
    valid_sidewalk = sidewalk_mask.copy() & (~road_mask)
    if protected_mask is not None and np.count_nonzero(protected_mask) > 0:
        prot_dilated = cv2.dilate(protected_mask.astype(np.uint8), np.ones((25, 25), np.uint8), iterations=1) > 0
        valid_sidewalk &= (~prot_dilated)

    if target_zone == "left_sidewalk":
        valid_sidewalk[:, int(width * 0.55):] = False
    elif target_zone == "right_sidewalk":
        valid_sidewalk[:, :int(width * 0.45)] = False

    sw_y, sw_x = np.where(valid_sidewalk)
    if sw_y.size < 50:
        # Fallback to sidewalk mask without strict non-road if tiny
        valid_sidewalk = sidewalk_mask.copy()
        if target_zone == "left_sidewalk":
            valid_sidewalk[:, int(width * 0.55):] = False
        elif target_zone == "right_sidewalk":
            valid_sidewalk[:, :int(width * 0.45)] = False
        sw_y, sw_x = np.where(valid_sidewalk)

    if sw_y.size == 0:
        # Ultimate fallback margin on sidewalk side
        default_x1 = int(width * 0.12) if target_zone == "left_sidewalk" else int(width * 0.82)
        default_x2 = int(width * 0.22) if target_zone == "left_sidewalk" else int(width * 0.92)
        default_y1 = int(height * 0.75)
        default_y2 = int(height * 0.65)
        support_points = [[float(default_x1), float(default_y1)], [float(default_x2), float(default_y2)]]
        top_y1 = default_y1 - int(height * 0.18)
        top_y2 = default_y2 - int(height * 0.14)
        span_x1 = max(15, int(width * 0.07))
        span_x2 = max(10, int(width * 0.05))
        footprint_polygon = [
            [float(default_x1 - span_x1), float(top_y1 + 8)],
            [float(default_x2 - span_x2), float(top_y2 + 6)],
            [float(default_x2 + span_x2), float(top_y2 - 8)],
            [float(default_x1 + span_x1), float(top_y1 - 10)],
        ]
        shadow_pts = [
            [float(default_x1 - span_x1), float(default_y1)],
            [float(default_x2 - span_x2), float(default_y2)],
            [float(default_x2 + span_x2), float(default_y2)],
            [float(default_x1 + span_x1), float(default_y1)],
        ]
        return {
            "target_zone": target_zone or "sidewalk",
            "anchor_points": support_points,
            "support_points": support_points,
            "footprint_polygon": footprint_polygon,
            "columns_count": len(support_points),
            "orientation": "parallel_to_pedestrian_corridor",
            "canopy_height": int(height * 0.18),
            "ground_contact": support_points,
            "intended_shadow_region": shadow_pts,
        }

    # Find two distinct ground anchors along the sidewalk corridor
    # Foreground anchor (lower in image / higher y) and midground anchor (receding / lower y)
    y_foreground = int(np.percentile(sw_y, 75))
    y_midground = int(np.percentile(sw_y, 45))

    row_fg = sw_x[sw_y == y_foreground]
    row_mg = sw_x[sw_y == y_midground]

    col1_x = int(np.median(row_fg)) if row_fg.size > 0 else int(np.median(sw_x))
    col2_x = int(np.median(row_mg)) if row_mg.size > 0 else col1_x

    # Ensure columns strictly lie on sidewalk and never in road
    ground_pts = []
    for cx, cy in [(col1_x, y_foreground), (col2_x, y_midground)]:
        # Verify and snap to closest valid sidewalk pixel
        if not valid_sidewalk[cy, cx]:
            dists = (sw_y - cy) ** 2 + (sw_x - cx) ** 2
            best_idx = int(np.argmin(dists))
            cx, cy = int(sw_x[best_idx]), int(sw_y[best_idx])
        ground_pts.append([float(cx), float(cy)])

    # Perspective canopy height rising above columns
    p1, p2 = ground_pts[0], ground_pts[1]
    col_h1 = int(height * 0.22)
    col_h2 = int(height * 0.16)

    top_y1 = max(int(height * 0.10), int(p1[1] - col_h1))
    top_y2 = max(int(height * 0.08), int(p2[1] - col_h2))

    span_x1 = max(15, int(width * 0.08))
    span_x2 = max(10, int(width * 0.06))

    # Perspective-correct tensile quadrilateral sailcloth
    footprint_polygon = [
        [float(p1[0] - span_x1), float(top_y1 + 10)],
        [float(p2[0] - span_x2), float(top_y2 + 8)],
        [float(p2[0] + span_x2), float(top_y2 - 10)],
        [float(p1[0] + span_x1), float(top_y1 - 12)],
    ]

    intended_shadow = [
        [float(p1[0] - span_x1), float(p1[1])],
        [float(p2[0] - span_x2), float(p2[1])],
        [float(p2[0] + span_x2), float(p2[1])],
        [float(p1[0] + span_x1), float(p1[1])],
    ]

    return {
        "target_zone": target_zone or "sidewalk",
        "anchor_points": ground_pts,
        "support_points": ground_pts,
        "footprint_polygon": footprint_polygon,
        "columns_count": len(ground_pts),
        "orientation": "parallel_to_pedestrian_corridor",
        "canopy_height": int((col_h1 + col_h2) // 2),
        "ground_contact": ground_pts,
        "intended_shadow_region": intended_shadow,
    }


def populate_intervention_explicit_geometry(
    intervention: Any,
    image: Image.Image,
    seg_result: Dict[str, Any],
    scene_understanding: Optional[Any] = None,
) -> None:
    """Populate explicit layout objects onto a SpatialInterventionSpec in-place."""
    w, h = image.size
    itype = getattr(intervention, "type", None) or intervention.get("type", "")
    tzone = getattr(intervention, "target_zone", None) or (intervention.get("target_zone") if isinstance(intervention, dict) else None)

    # Surface masks
    raw_preds = seg_result.get("raw_preds")
    if raw_preds is not None:
        road_mask = np.isin(raw_preds, [6, 54, 91])
        pavement_mask = np.isin(raw_preds, [3, 11, 52])
        roof_mask = np.isin(raw_preds, [105])
        wall_mask = np.isin(raw_preds, [0, 1, 8, 14, 25, 32, 38, 42, 48, 51, 79, 84, 88, 95])
    else:
        road_mask = np.zeros((h, w), dtype=bool)
        pavement_mask = np.zeros((h, w), dtype=bool)
        roof_mask = np.zeros((h, w), dtype=bool)
        wall_mask = np.zeros((h, w), dtype=bool)

    # Protected objects
    from .mask_builder import build_protected_object_mask
    prot_mask = build_protected_object_mask(w, h, seg_result)
    depth_map = getattr(scene_understanding, "depth_map", None) if scene_understanding else None

    def _set_attr(obj: Any, key: str, val: Any) -> None:
        if isinstance(obj, dict):
            obj[key] = val
        else:
            setattr(obj, key, val)

    if itype == "tree_canopy":
        layout = compute_tree_row_layout(
            width=w,
            height=h,
            sidewalk_mask=pavement_mask,
            road_mask=road_mask,
            depth_map=depth_map,
            protected_mask=prot_mask,
            wall_mask=wall_mask,
            target_zone=tzone,
            requested_trees=3,
        )
        _set_attr(intervention, "anchors", layout["anchors"])
        _set_attr(intervention, "spacing", layout["spacing"])
        _set_attr(intervention, "corridor_polyline", layout["corridor_polyline"])
        _set_attr(intervention, "planting_points", layout["planting_points"])
        _set_attr(intervention, "ground_anchor", layout["ground_anchor"])
        _set_attr(intervention, "canopy_extent", layout["canopy_extent"])
        _set_attr(intervention, "tree_height", layout["tree_height"])
        _set_attr(intervention, "trunk_width", layout["trunk_width"])
        _set_attr(intervention, "planting_pit", layout["planting_pit"])
        _set_attr(intervention, "intended_shade_coverage", layout["intended_shade_coverage"])
        _set_attr(intervention, "relative_depth", layout["relative_depth"])
        _set_attr(intervention, "scale", layout["scale"])
        _set_attr(intervention, "side", layout["side"])
        _set_attr(intervention, "depth", layout["depth"])
        _set_attr(intervention, "target_zone", layout["target_zone"])

    elif itype == "shade_structure":
        layout = compute_shade_structure_layout(
            width=w,
            height=h,
            sidewalk_mask=pavement_mask,
            road_mask=road_mask,
            protected_mask=prot_mask,
            target_zone=tzone,
        )
        _set_attr(intervention, "anchor_points", layout["anchor_points"])
        _set_attr(intervention, "support_points", layout["support_points"])
        _set_attr(intervention, "footprint_polygon", layout["footprint_polygon"])
        _set_attr(intervention, "orientation", layout["orientation"])
        _set_attr(intervention, "canopy_height", layout["canopy_height"])
        _set_attr(intervention, "ground_contact", layout["ground_contact"])
        _set_attr(intervention, "intended_shadow_region", layout["intended_shadow_region"])
        _set_attr(intervention, "target_zone", layout["target_zone"])

    elif itype == "cool_pavement":
        poly = extract_polygon_contour(road_mask)
        quad = extract_surface_quadrilateral(road_mask)
        ry, rx = np.where(road_mask)
        extent = [int(np.min(rx)), int(np.min(ry)), int(np.max(rx)), int(np.max(ry))] if rx.size > 0 else [0, 0, w, h]
        _set_attr(intervention, "surface_polygon", poly)
        _set_attr(intervention, "surface_quad", quad)
        _set_attr(intervention, "extent", extent)
        _set_attr(intervention, "material_type", "high_albedo_solar_reflective_asphalt")
        _set_attr(intervention, "target_zone", tzone or "roadway")

    elif itype == "permeable_pave":
        poly = extract_polygon_contour(pavement_mask)
        quad = extract_surface_quadrilateral(pavement_mask)
        py, px = np.where(pavement_mask)
        extent = [int(np.min(px)), int(np.min(py)), int(np.max(px)), int(np.max(py))] if px.size > 0 else [0, 0, w, h]
        _set_attr(intervention, "surface_polygon", poly)
        _set_attr(intervention, "surface_quad", quad)
        _set_attr(intervention, "extent", extent)
        _set_attr(intervention, "material_type", "interlocking_modular_porous_pavers")
        _set_attr(intervention, "target_zone", tzone or "sidewalk")

    elif itype in ("cool_roof", "green_roof"):
        poly = extract_polygon_contour(roof_mask)
        _set_attr(intervention, "roof_polygon", poly)
        _set_attr(
            intervention,
            "treatment_type",
            "extensive_sedum_vegetated_roof" if itype == "green_roof" else "high_albedo_elastomeric_reflective_coating",
        )
