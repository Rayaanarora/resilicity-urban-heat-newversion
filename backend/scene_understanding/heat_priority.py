"""Spatial Heat Priority Mapping for Targeted Microclimate Interventions.

Computes a continuous 2D spatial HeatPriority(x, y) map in [0.0, 1.0] by combining:
- Imperviousness (impervious asphalt / concrete surfaces)
- Solar exposure (direct solar radiation intensity)
- Pedestrian exposure (sidewalk / walking zones requiring thermal comfort)
- Canyon enclosure (heat-trapping street canyon strength)
- Lack of vegetation (absence of cooling tree canopy / greenery)
- Lack of existing shade (unshaded ground zones)

This is a spatial PRIORITIZATION index that identifies where resilience interventions
matter most, distinctly separate from the quantitative Landsat ML surface temperature model.
"""

import logging
from typing import Any, Dict, Tuple
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("resilicity.heat_priority")


def compute_spatial_heat_priority_map(
    width: int,
    height: int,
    road_mask: np.ndarray,
    pavement_mask: np.ndarray,
    vegetation_mask: np.ndarray,
    wall_mask: np.ndarray,
    sun_mask: np.ndarray,
    shade_mask: np.ndarray,
    street_canyon_strength: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute the 2D spatial HeatPriority(x,y) map and summary metrics.
    
    Returns:
        np.ndarray: float32 2D array of shape (height, width) with values in [0.0, 1.0].
        Dict[str, Any]: summary heat priority statistics across pedestrian and road zones.
    """
    # 1. Imperviousness layer (road, pavement, concrete walls = 1.0, veg/sky = 0.0)
    impervious = (road_mask | pavement_mask | wall_mask).astype(np.float32)

    # 2. Solar exposure layer (sunlit regions = 1.0, shaded regions = 0.0)
    # Smooth slightly to create realistic spatial heat gradients
    solar_smooth = cv2.GaussianBlur(sun_mask.astype(np.float32), (15, 15), 0)

    # 3. Pedestrian exposure layer (sidewalk and walking zones have highest human thermal urgency)
    pedestrian = pavement_mask.astype(np.float32)

    # 4. Canyon enclosure layer (constant street canyon trapping factor across built surfaces)
    canyon = np.where(road_mask | pavement_mask | wall_mask, float(street_canyon_strength), 0.0).astype(np.float32)

    # 5. Lack of vegetation layer (1.0 where no vegetation exists)
    veg_smooth = cv2.GaussianBlur(vegetation_mask.astype(np.float32), (15, 15), 0)
    lack_of_veg = (1.0 - veg_smooth).clip(0.0, 1.0)

    # 6. Lack of existing shade layer (1.0 where direct sun strikes unshaded ground)
    shade_smooth = cv2.GaussianBlur(shade_mask.astype(np.float32), (15, 15), 0)
    lack_of_shade = (1.0 - shade_smooth).clip(0.0, 1.0)

    # Weights: w1..w6 summing to 1.0
    # Imperviousness: 0.20, SolarExposure: 0.25, PedestrianExposure: 0.20,
    # CanyonEnclosure: 0.10, LackOfVeg: 0.15, LackOfShade: 0.10
    w_imp = 0.20
    w_sol = 0.25
    w_ped = 0.20
    w_cyn = 0.10
    w_veg = 0.15
    w_shd = 0.10

    raw_priority = (
        w_imp * impervious +
        w_sol * solar_smooth +
        w_ped * pedestrian +
        w_cyn * canyon +
        w_veg * lack_of_veg +
        w_shd * lack_of_shade
    )

    # Sky and water regions are zeroed out (not ground heat intervention zones)
    non_target_mask = ~(road_mask | pavement_mask | wall_mask | vegetation_mask)
    raw_priority[non_target_mask] = 0.0

    # Normalize to [0.0, 1.0]
    p_max = float(np.max(raw_priority))
    p_min = float(np.min(raw_priority))
    if p_max > p_min:
        norm_priority = ((raw_priority - p_min) / (p_max - p_min)).astype(np.float32)
    else:
        norm_priority = raw_priority.astype(np.float32)

    # Compute zone-specific priority summaries
    ped_priority_val = float(np.mean(norm_priority[pavement_mask])) if np.any(pavement_mask) else 0.5
    road_priority_val = float(np.mean(norm_priority[road_mask])) if np.any(road_mask) else 0.5
    overall_val = float(np.mean(norm_priority[road_mask | pavement_mask])) if np.any(road_mask | pavement_mask) else 0.5

    summary = {
        "overall_heat_priority_score": round(overall_val, 3),
        "pedestrian_heat_priority_score": round(ped_priority_val, 3),
        "roadway_heat_priority_score": round(road_priority_val, 3),
        "high_priority_zone": "pedestrian_sidewalk" if ped_priority_val >= road_priority_val else "vehicular_roadway",
    }

    return norm_priority, summary


def render_heat_priority_colormap(priority_map: np.ndarray) -> Image.Image:
    """Render a visual heat-priority heatmap image (Turbo/Jet colormap) for debugging."""
    scaled = (priority_map * 255.0).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    # Convert BGR to RGB
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored_rgb)
