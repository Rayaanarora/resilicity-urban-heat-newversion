"""Contour and polygon extraction from semantic segmentation masks."""

from typing import List, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def extract_polygons_for_mask(
    mask: np.ndarray,
    min_area_ratio: float = 0.0015,
    max_polygons: int = 8,
    epsilon_factor: float = 0.008,
) -> List[List[List[float]]]:
    """Extract simplified polygon regions for a binary mask.
    
    Args:
        mask: 2D boolean or uint8 binary mask of shape (H, W).
        min_area_ratio: Minimum contour area relative to total image area to filter noise.
        max_polygons: Maximum number of largest disconnected polygons to return.
        epsilon_factor: Approximation accuracy for cv2.approxPolyDP relative to arc length.
        
    Returns:
        List of polygons, where each polygon is a list of [x%, y%] coordinates.
    """
    if cv2 is None or not np.any(mask):
        return []

    h, w = mask.shape
    total_area = h * w
    min_area = total_area * min_area_ratio

    # Downsample slightly if image is very large for fast contour finding
    scale = 1.0
    if max(h, w) > 800:
        scale = 800.0 / max(h, w)
        small_w = max(1, int(w * scale))
        small_h = max(1, int(h * scale))
        binary = cv2.resize(mask.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        cur_w, cur_h = small_w, small_h
    else:
        binary = mask.astype(np.uint8)
        cur_w, cur_h = w, h

    cur_min_area = (cur_w * cur_h) * min_area_ratio

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Sort contours by area descending and take top N
    significant_contours = [
        c for c in contours if cv2.contourArea(c) >= cur_min_area
    ]
    significant_contours.sort(key=cv2.contourArea, reverse=True)
    top_contours = significant_contours[:max_polygons]

    polygons: List[List[List[float]]] = []
    for c in top_contours:
        peri = cv2.arcLength(c, True)
        epsilon = max(1.5, epsilon_factor * peri)
        approx = cv2.approxPolyDP(c, epsilon, True)
        
        # We need at least a triangle
        if len(approx) < 3:
            continue
            
        pts = approx[:, 0, :]
        # Convert to percentage relative to original image [0, 100]
        polygon_pts = [
            [round(float(pt[0]) / cur_w * 100.0, 2), round(float(pt[1]) / cur_h * 100.0, 2)]
            for pt in pts
        ]
        polygons.append(polygon_pts)

    return polygons
