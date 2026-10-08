"""Classical Computer Vision and Geometry Analysis for Streetscapes.

Derives perspective convergence, vanishing point, horizon estimation,
road corridor boundaries, curb lines, left vs right sidewalk separation,
street canyon H/W ratio proxy, and Sky View Factor (SVF) proxy.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("resilicity.geometry")


def estimate_vanishing_point(
    image: Image.Image,
    road_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Estimate vanishing point (vx, vy) in normalized coordinates [0, 1] and horizon line y.
    
    Uses Canny edge detection and probabilistic Hough transform, filtering for
    converging perspective lines in the street corridor.
    """
    w, h = image.size
    img_gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    
    # Gaussian blur and Canny edge
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Focus primarily on the lower 70% of the image for street perspective lines
    roi_mask = np.zeros_like(edges)
    roi_mask[int(h * 0.25):, :] = 255
    edges_roi = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=int(min(w, h) * 0.15),
        maxLineGap=20,
    )

    if lines is None or len(lines) < 2:
        # Default typical perspective: center x, 45% y
        return 0.5, 0.45, 0.45

    left_slopes = []
    right_slopes = []
    left_lines = []
    right_lines = []

    for line in lines:
        coords = np.array(line).flatten()
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        if abs(x2 - x1) < 1e-3:
            continue
        slope = (y2 - y1) / (x2 - x1)
        angle = np.degrees(np.arctan(slope))

        # Perspective lines typically angle between 20° and 75°
        if -75 < angle < -20:  # Slanted upwards-right (from left side of street)
            left_slopes.append(slope)
            left_lines.append((x1, y1, x2, y2))
        elif 20 < angle < 75:  # Slanted upwards-left (from right side of street)
            right_slopes.append(slope)
            right_lines.append((x1, y1, x2, y2))

    if left_lines and right_lines:
        # Compute intersections between left and right perspective lines
        intersections = []
        for lx1, ly1, lx2, ly2 in left_lines[:8]:
            for rx1, ry1, rx2, ry2 in right_lines[:8]:
                # Line equations: a1*x + b1*y = c1
                denom = (lx1 - lx2) * (ry1 - ry2) - (ly1 - ly2) * (rx1 - rx2)
                if abs(denom) > 1e-4:
                    ix = ((lx1 * ly2 - ly1 * lx2) * (rx1 - rx2) - (lx1 - lx2) * (rx1 * ry2 - ry1 * rx2)) / denom
                    iy = ((lx1 * ly2 - ly1 * lx2) * (ry1 - ry2) - (ly1 - ly2) * (rx1 * ry2 - ry1 * rx2)) / denom
                    # Only accept intersections in plausible upper-middle area
                    if 0.1 * w <= ix <= 0.9 * w and 0.15 * h <= iy <= 0.70 * h:
                        intersections.append((ix, iy))

        if intersections:
            med_x = float(np.median([p[0] for p in intersections])) / w
            med_y = float(np.median([p[1] for p in intersections])) / h
            return float(np.clip(med_x, 0.2, 0.8)), float(np.clip(med_y, 0.25, 0.65)), float(np.clip(med_y, 0.25, 0.65))

    # Fallback if specific intersections weren't found
    return 0.5, 0.45, 0.45


def analyze_street_corridor(
    width: int,
    height: int,
    road_mask: np.ndarray,
    pavement_mask: np.ndarray,
    wall_mask: np.ndarray,
    sky_mask: np.ndarray,
    vanishing_point: Tuple[float, float, float],
) -> Dict[str, Any]:
    """Segment street into left sidewalk, right sidewalk, and roadway corridors,

    and derive relative width proxies, street canyon strength, and SVF proxy.
    """
    vx_norm, vy_norm, horizon_y_norm = vanishing_point
    center_x = int(vx_norm * width)
    horizon_y = int(horizon_y_norm * height)

    # 1. Roadway Corridor
    road_pixels = np.count_nonzero(road_mask)
    road_pct = float(road_pixels / (width * height))
    # Road width proxy in lower third (near camera)
    lower_slice = road_mask[int(height * 0.75):, :]
    if lower_slice.size > 0:
        road_width_proxy = float(np.mean(np.sum(lower_slice, axis=1)) / width)
    else:
        road_width_proxy = min(1.0, road_pct * 2.5)

    # 2. Left vs Right Sidewalk Partitioning
    # Split the pavement mask by the street center/vanishing axis
    x_indices = np.arange(width)
    left_half_mask = np.tile(x_indices < center_x, (height, 1))
    right_half_mask = ~left_half_mask

    left_pave_mask = pavement_mask & left_half_mask
    right_pave_mask = pavement_mask & right_half_mask

    left_pave_px = int(np.count_nonzero(left_pave_mask))
    right_pave_px = int(np.count_nonzero(right_pave_mask))
    total_px = width * height

    left_pave_pct = float(left_pave_px / total_px)
    right_pave_pct = float(right_pave_px / total_px)

    # Sidewalk width proxies (average horizontal span in lower half)
    pave_lower_left = left_pave_mask[int(height * 0.55):, :]
    pave_lower_right = right_pave_mask[int(height * 0.55):, :]

    left_width_proxy = float(np.mean(np.sum(pave_lower_left, axis=1)) / (width * 0.5)) if pave_lower_left.size > 0 else left_pave_pct * 3.0
    right_width_proxy = float(np.mean(np.sum(pave_lower_right, axis=1)) / (width * 0.5)) if pave_lower_right.size > 0 else right_pave_pct * 3.0

    left_width_proxy = float(np.clip(left_width_proxy, 0.0, 1.0))
    right_width_proxy = float(np.clip(right_width_proxy, 0.0, 1.0))

    # 3. Building Density & Facades
    wall_pct = float(np.count_nonzero(wall_mask) / total_px)
    sky_pct = float(np.count_nonzero(sky_mask) / total_px)

    # 4. Street Canyon Strength & Height-to-Width (H/W) Proxy
    # Enclosure is proportional to vertical wall presence flanking the corridor divided by open street width
    street_width_proxy = max(0.15, road_width_proxy + 0.5 * (left_width_proxy + right_width_proxy))
    # Wall height proxy: average vertical extent of wall pixels
    wall_cols = np.sum(wall_mask, axis=0) / height
    avg_wall_height_proxy = float(np.mean(wall_cols[wall_cols > 0.05])) if np.any(wall_cols > 0.05) else (wall_pct * 1.5)
    
    hw_ratio_proxy = float(np.clip(avg_wall_height_proxy / street_width_proxy, 0.1, 4.0))

    # Canyon strength in [0.0, 1.0]: high H/W and low sky visibility
    canyon_strength = float(np.clip(
        0.5 * (avg_wall_height_proxy / (avg_wall_height_proxy + street_width_proxy)) +
        0.5 * (1.0 - min(1.0, sky_pct * 2.5)),
        0.0, 1.0,
    ))

    # 5. Sky View Factor (SVF) Proxy
    # Approximates the fraction of overhead sky visible from street level
    svf_proxy = float(np.clip(sky_pct * 1.8 + (1.0 - canyon_strength) * 0.2, 0.05, 0.95))

    return {
        "vanishing_point": [vx_norm, vy_norm],
        "horizon_y": horizon_y_norm,
        "road_width_proxy": round(road_width_proxy, 3),
        "left_sidewalk_width_proxy": round(left_width_proxy, 3),
        "right_sidewalk_width_proxy": round(right_width_proxy, 3),
        "left_sidewalk_pct": round(left_pave_pct * 100.0, 2),
        "right_sidewalk_pct": round(right_pave_pct * 100.0, 2),
        "building_wall_density": round(wall_pct, 3),
        "street_canyon_strength": round(canyon_strength, 3),
        "height_to_width_proxy": round(hw_ratio_proxy, 2),
        "sky_visibility_proxy": round(svf_proxy, 3),
        "left_sidewalk_mask": left_pave_mask,
        "right_sidewalk_mask": right_pave_mask,
    }


def detect_roadway_material(
    image: Image.Image,
    road_mask: np.ndarray,
) -> Dict[str, Any]:
    """Analyze road surface texture and color to determine if it is historic cobblestone/stone setts

    or standard asphalt roadway.

    Cobblestones exhibit:
    - High local Laplacian texture variance across road pixels
    - Regular high-frequency gradient edges
    - Warmer/earthen gray tone rather than uniform dark bitumen
    """
    if not np.any(road_mask):
        return {
            "material": "standard_asphalt",
            "is_heritage": False,
            "texture_energy": 0.0,
            "confidence": 0.70,
            "reason": "No roadway pixels detected.",
        }

    w, h = image.size
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Focus on central/lower portion of road where texture is clear
    lower_road = road_mask.copy()
    lower_road[:int(h * 0.45), :] = False

    if not np.any(lower_road):
        lower_road = road_mask

    # Calculate Laplacian variance as texture energy measure
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    road_lap = laplacian[lower_road]
    tex_var = float(np.var(road_lap)) if road_lap.size > 0 else 0.0

    # Sobel gradient energy
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    road_grad = grad_mag[lower_road]
    grad_mean = float(np.mean(road_grad)) if road_grad.size > 0 else 0.0

    # Check color warmth/tone of road
    road_rgb = img_np[lower_road]
    mean_r = float(np.mean(road_rgb[:, 0])) if road_rgb.size > 0 else 100.0
    mean_g = float(np.mean(road_rgb[:, 1])) if road_rgb.size > 0 else 100.0
    mean_b = float(np.mean(road_rgb[:, 2])) if road_rgb.size > 0 else 100.0

    # Threshold: historic cobblestone or stone sett pavement has high texture variance
    is_heritage = bool(tex_var > 950.0 or grad_mean > 32.0)
    mat_name = "historic_cobblestone" if is_heritage else "standard_asphalt"
    reason = (
        "High spatial frequency texture and inter-block relief detected on roadway surface, indicative of historic cobblestone or stone setts."
        if is_heritage
        else "Relatively uniform surface texture characteristic of standard asphalt bitumen roadway."
    )

    return {
        "material": mat_name,
        "is_heritage": is_heritage,
        "texture_energy": round(tex_var, 1),
        "gradient_mean": round(grad_mean, 1),
        "confidence": 0.88,
        "reason": reason,
    }


def analyze_vegetation_spatial_distribution(
    vegetation_mask: np.ndarray,
    left_sidewalk_mask: np.ndarray,
    right_sidewalk_mask: np.ndarray,
    road_mask: np.ndarray,
    vanishing_point: Tuple[float, float, float],
) -> Dict[str, Any]:
    """Analyze spatial continuity and distribution of existing trees/canopy.

    Distinguishes:
    - clustered vs continuous distribution
    - pedestrian shade coverage
    - presence of shade gaps along walking corridors
    """
    h, w = vegetation_mask.shape
    total_veg_px = int(np.count_nonzero(vegetation_mask))

    if total_veg_px < 50:
        return {
            "canopy_continuity": 0.0,
            "canopy_distribution": "absent",
            "pedestrian_shade_coverage": 0.0,
            "shade_gaps_pct": 100.0,
            "canopy_continuity_score": 0.0,
            "description": "Vegetative canopy is virtually absent along street corridors.",
        }

    # Evaluate horizontal distribution across street
    y_coords, x_coords = np.nonzero(vegetation_mask)
    x_std_norm = float(np.std(x_coords) / w)
    y_mean_norm = float(np.mean(y_coords) / h)

    # If y_mean_norm < 0.4, vegetation is in background / far end
    # If x_std_norm is small (< 0.12), clustered in one spot
    if x_std_norm < 0.12:
        dist_type = "clustered_isolated"
        continuity = 0.20
    elif y_mean_norm < 0.40:
        dist_type = "clustered_far_end"
        continuity = 0.35
    elif x_std_norm > 0.25:
        dist_type = "distributed"
        continuity = 0.70
    else:
        dist_type = "fragmented"
        continuity = 0.45

    # Check sidewalk overlap (canopy directly shading pedestrians)
    sw_mask = left_sidewalk_mask | right_sidewalk_mask
    sw_px = np.count_nonzero(sw_mask)
    if sw_px > 0:
        # Dilate vegetation slightly to approximate cast shade footprint
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        veg_shade = cv2.dilate(vegetation_mask.astype(np.uint8), kernel)
        sw_shaded_px = np.count_nonzero(veg_shade & sw_mask)
        ped_shade_cov = float(sw_shaded_px / sw_px * 100.0)
    else:
        ped_shade_cov = 0.0

    shade_gaps = max(0.0, 100.0 - ped_shade_cov)

    return {
        "canopy_continuity": round(continuity, 2),
        "canopy_distribution": dist_type,
        "pedestrian_shade_coverage": round(ped_shade_cov, 1),
        "shade_gaps_pct": round(shade_gaps, 1),
        "canopy_continuity_score": round(continuity, 2),
        "description": f"Existing canopy is {dist_type} with {ped_shade_cov:.1f}% pedestrian sidewalk shade coverage.",
    }
