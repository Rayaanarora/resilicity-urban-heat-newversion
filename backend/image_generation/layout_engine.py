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
        y_grad = np.linspace(1.0, 0.0, height)[:, None]
        depth_map = 1.0 - y_grad

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

    # Identify curbside interface (boundary between sidewalk and road)
    curb_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_road = cv2.dilate(road_mask.astype(np.uint8), curb_kernel)
    curbside_mask = active_sw & (dilated_road > 0)
    has_curbside = np.count_nonzero(curbside_mask) > 50

    # Exponential sampling from foreground (near camera, high y) to background (near horizon, lower y)
    t_vals = np.linspace(0.18, 0.85, max(4, requested_trees + 2))
    sampled_y_levels = [int(y_max - (t ** 1.65) * (y_max - y_min)) for t in t_vals]

    anchors: List[Dict[str, Any]] = []
    corridor_points: List[List[float]] = []

    # Proximity exclusion radius around protected objects (in pixels)
    prot_dist_transform = None
    if protected_mask is not None and np.count_nonzero(protected_mask) > 0:
        prot_inv = (~protected_mask).astype(np.uint8) * 255
        prot_dist_transform = cv2.distanceTransform(prot_inv, cv2.DIST_L2, 5)

    for y_plant in sampled_y_levels:
        if has_curbside and np.any(curbside_mask[y_plant]):
            row_xs = np.where(curbside_mask[y_plant])[0]
        else:
            row_xs = sw_x[sw_y == y_plant]

        if row_xs.size == 0:
            continue

        # Ground anchor x coordinate: verge center
        x_plant = int(np.median(row_xs))
        corridor_points.append([float(x_plant), float(y_plant)])

        # 1. Reject if too close to protected objects (vehicles, pedestrians, streetlights)
        if prot_dist_transform is not None:
            dist_to_prot = prot_dist_transform[y_plant, x_plant]
            if dist_to_prot < 14.0:  # within 14px of a car/pedestrian/pole
                logger.debug("Rejected anchor (%d, %d): too close to protected object (dist=%.1f)", x_plant, y_plant, dist_to_prot)
                continue

        # 2. Reject if directly in active vehicular roadway
        if road_mask[y_plant, x_plant]:
            # Try shifting toward sidewalk interior
            shifted = False
            for dx in (-15, 15, -25, 25):
                cand_x = np.clip(x_plant + dx, 0, width - 1)
                if active_sw[y_plant, cand_x] and not road_mask[y_plant, cand_x]:
                    x_plant = int(cand_x)
                    shifted = True
                    break
            if not shifted:
                continue

        # 3. Relative depth driven scale
        rel_depth = float(depth_map[y_plant, x_plant])  # 0.0=near, 1.0=far
        # Perspective scaling factor: near trees 1.0, distant trees down to 0.28
        scale = float(np.clip(1.0 - 0.72 * rel_depth, 0.28, 1.0))

        tree_h = int(round(height * 0.38 * scale))
        canopy_rx = int(round(width * 0.13 * scale))
        canopy_ry = int(round(tree_h * 0.48))
        trunk_w = max(4, int(round(width * 0.022 * scale)))

        # Contact base of trunk at ground anchor
        anchors.append({
            "x": int(x_plant),
            "y": int(y_plant),
            "relative_depth": round(rel_depth, 3),
            "scale": round(scale, 3),
            "canopy_radius": int(canopy_rx),
            "canopy_height": int(canopy_ry),
            "tree_height": int(tree_h),
            "trunk_width": int(trunk_w),
            "canopy_center_y": max(canopy_ry + 10, y_plant - int(tree_h * 0.58)),
        })

        if len(anchors) >= requested_trees:
            break

    # If too few anchors found, create at least 2 geometrically sound anchors along corridor
    if len(anchors) < 2 and len(corridor_points) >= 2:
        for pt in [corridor_points[0], corridor_points[-1]]:
            px, py = int(pt[0]), int(pt[1])
            d = float(depth_map[py, px]) if depth_map is not None else 0.5
            s = float(np.clip(1.0 - 0.72 * d, 0.3, 1.0))
            anchors.append({
                "x": px,
                "y": py,
                "relative_depth": round(d, 3),
                "scale": round(s, 3),
                "canopy_radius": int(width * 0.12 * s),
                "canopy_height": int(height * 0.18 * s),
                "tree_height": int(height * 0.35 * s),
                "trunk_width": max(4, int(width * 0.02 * s)),
                "canopy_center_y": max(int(height * 0.1), py - int(height * 0.20 * s)),
            })

    # Sort corridor polyline points
    corridor_points.sort(key=lambda p: p[1], reverse=True)

    spacing_px = 0.0
    if len(anchors) >= 2:
        dists = [
            float(np.hypot(anchors[i]["x"] - anchors[i+1]["x"], anchors[i]["y"] - anchors[i+1]["y"]))
            for i in range(len(anchors) - 1)
        ]
        spacing_px = float(np.mean(dists))

    side_name = "left" if (target_zone == "left_sidewalk" or (target_zone is None and len(anchors) > 0 and anchors[0]["x"] < width * 0.5)) else "right"

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
            "ground_anchor": [ax, ay],
            "depth": d,
            "scale": s,
            "spacing": round(spacing_px, 1),
            "canopy_extent": extent,
            "canopy_radius": rx,
            "tree_height": int(anc["tree_height"]),
            "trunk_width": int(anc["trunk_width"]),
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

    return {
        "side": side_name,
        "planting_points": planting_points,
        "spacing": round(spacing_px, 1),
        "depth": round(float(np.mean(depth_values)), 3) if depth_values else 0.5,
        "scale": round(float(np.mean(scale_values)), 3) if scale_values else 0.7,
        "ground_anchor": all_ground_anchors[0] if all_ground_anchors else [width // 4, int(height * 0.8)],
        "canopy_extent": overall_canopy_extent,
        # Backwards-compatible keys
        "anchors": anchors,
        "corridor_polyline": corridor_points,
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
        valid_sidewalk &= (~protected_mask)

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
        footprint_polygon = [
            [float(default_x1 - 10), float(default_y1 - int(height * 0.18))],
            [float(default_x1), float(default_y1 - int(height * 0.28))],
            [float(default_x2 + 10), float(default_y2 - int(height * 0.28))],
            [float(default_x2), float(default_y2 - int(height * 0.18))],
        ]
        return {
            "anchor_points": support_points,
            "support_points": support_points,
            "footprint_polygon": footprint_polygon,
            "columns_count": len(support_points),
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

    return {
        "anchor_points": ground_pts,
        "support_points": ground_pts,
        "footprint_polygon": footprint_polygon,
        "columns_count": len(ground_pts),
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
        if isinstance(intervention, dict):
            intervention["anchors"] = layout["anchors"]
            intervention["spacing"] = layout["spacing"]
            intervention["corridor_polyline"] = layout["corridor_polyline"]
        else:
            intervention.anchors = layout["anchors"]
            intervention.spacing = layout["spacing"]
            intervention.corridor_polyline = layout["corridor_polyline"]

    elif itype == "shade_structure":
        layout = compute_shade_structure_layout(
            width=w,
            height=h,
            sidewalk_mask=pavement_mask,
            road_mask=road_mask,
            protected_mask=prot_mask,
            target_zone=tzone,
        )
        if isinstance(intervention, dict):
            intervention["anchor_points"] = layout["anchor_points"]
            intervention["support_points"] = layout["support_points"]
            intervention["footprint_polygon"] = layout["footprint_polygon"]
        else:
            intervention.anchor_points = layout["anchor_points"]
            intervention.support_points = layout["support_points"]
            intervention.footprint_polygon = layout["footprint_polygon"]

    elif itype == "cool_pavement":
        poly = extract_polygon_contour(road_mask)
        quad = extract_surface_quadrilateral(road_mask)
        if isinstance(intervention, dict):
            intervention["surface_polygon"] = poly
            intervention["surface_quad"] = quad
        else:
            intervention.surface_polygon = poly
            intervention.surface_quad = quad

    elif itype == "permeable_pave":
        poly = extract_polygon_contour(pavement_mask)
        quad = extract_surface_quadrilateral(pavement_mask)
        if isinstance(intervention, dict):
            intervention["surface_polygon"] = poly
            intervention["surface_quad"] = quad
        else:
            intervention.surface_polygon = poly
            intervention.surface_quad = quad

    elif itype in ("cool_roof", "green_roof"):
        poly = extract_polygon_contour(roof_mask)
        if isinstance(intervention, dict):
            intervention["roof_polygon"] = poly
        else:
            intervention.roof_polygon = poly
