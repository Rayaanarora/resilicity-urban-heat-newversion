"""Visual Solar Exposure and Shade Understanding for Urban Streetscapes.

Estimates direct sunlight vs shaded regions, visual solar exposure proxies,
existing shade density, and exposed sidewalk/roadway segments using image
luminance (L channel in LAB) and semantic surface masks.
All scores are normalized proxies in [0.0, 1.0].
"""

import logging
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("resilicity.solar")


def estimate_solar_and_shade(
    image: Image.Image,
    road_mask: np.ndarray,
    left_sidewalk_mask: np.ndarray,
    right_sidewalk_mask: np.ndarray,
    wall_mask: np.ndarray,
) -> Dict[str, Any]:
    """Estimate visual solar exposure and existing shade distribution across spatial zones."""
    img_rgb = np.array(image.convert("RGB"))
    # Convert to LAB color space to isolate perceived luminance (L channel: 0-255)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel = img_lab[:, :, 0].astype(np.float32)

    # Ground surface mask (road + left sidewalk + right sidewalk)
    ground_mask = road_mask | left_sidewalk_mask | right_sidewalk_mask
    ground_pixels = l_channel[ground_mask]

    if ground_pixels.size > 0:
        # Ground luminance threshold to separate direct sunlight from cast shadow
        # Use 40th percentile of ground luminance as shade threshold
        shade_thresh = float(np.percentile(ground_pixels, 40))
        # Direct sunlight threshold at 65th percentile
        sun_thresh = float(np.percentile(ground_pixels, 65))
    else:
        shade_thresh = 110.0
        sun_thresh = 150.0

    shade_map = (l_channel <= shade_thresh).astype(np.float32)
    sun_map = (l_channel >= sun_thresh).astype(np.float32)

    # Zone-level exposure proxies
    def _zone_stats(zone_mask: np.ndarray) -> Tuple[float, float]:
        px_count = np.count_nonzero(zone_mask)
        if px_count == 0:
            return 0.5, 0.2
        sun_ratio = float(np.sum(sun_map * zone_mask) / px_count)
        shade_ratio = float(np.sum(shade_map * zone_mask) / px_count)
        # Solar exposure proxy [0, 1]
        solar_exposure = float(np.clip(sun_ratio * 1.5 + (1.0 - shade_ratio) * 0.3, 0.05, 0.98))
        existing_shade = float(np.clip(shade_ratio, 0.02, 0.95))
        return round(solar_exposure, 3), round(existing_shade, 3)

    road_solar, road_shade = _zone_stats(road_mask)
    left_solar, left_shade = _zone_stats(left_sidewalk_mask)
    right_solar, right_shade = _zone_stats(right_sidewalk_mask)

    # Facade shading asymmetry to infer approximate sun angle (left vs right)
    h, w = l_channel.shape
    mid_x = w // 2
    left_wall_mask = wall_mask[:, :mid_x]
    right_wall_mask = wall_mask[:, mid_x:]

    left_wall_l = float(np.mean(l_channel[:, :mid_x][left_wall_mask])) if np.any(left_wall_mask) else 128.0
    right_wall_l = float(np.mean(l_channel[:, mid_x:][right_wall_mask])) if np.any(right_wall_mask) else 128.0

    if left_wall_l > right_wall_l + 15:
        sun_direction = "illuminating_from_right"
        shade_side = "right_facade_shaded"
    elif right_wall_l > left_wall_l + 15:
        sun_direction = "illuminating_from_left"
        shade_side = "left_facade_shaded"
    else:
        sun_direction = "overhead_solar_insolation"
        shade_side = "balanced_solar_exposure"

    # Overall visual solar exposure proxy
    overall_solar = float(np.clip(0.5 * road_solar + 0.25 * left_solar + 0.25 * right_solar, 0.1, 0.95))
    overall_shade = float(np.clip(0.5 * road_shade + 0.25 * left_shade + 0.25 * right_shade, 0.05, 0.95))

    return {
        "solar_exposure_proxy": round(overall_solar, 3),
        "existing_shade_proxy": round(overall_shade, 3),
        "sun_direction_proxy": sun_direction,
        "shade_distribution": shade_side,
        "roadway": {
            "solar_exposure": road_solar,
            "existing_shade": road_shade,
        },
        "left_sidewalk": {
            "solar_exposure": left_solar,
            "existing_shade": left_shade,
        },
        "right_sidewalk": {
            "solar_exposure": right_solar,
            "existing_shade": right_shade,
        },
        "shade_mask": shade_map,
        "sun_mask": sun_map,
    }
