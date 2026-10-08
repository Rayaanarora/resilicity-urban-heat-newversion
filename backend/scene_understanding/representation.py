"""Comprehensive Spatial Scene Understanding Representation.

Integrates semantic segmentation, monocular depth estimation, classical CV geometry,
solar & shade reasoning, and protected object separation into a unified,
spatially intelligent SceneUnderstanding model.
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from .depth import estimate_relative_depth
from .geometry import analyze_street_corridor, estimate_vanishing_point
from .heat_priority import compute_spatial_heat_priority_map, render_heat_priority_colormap
from .solar import estimate_solar_and_shade

logger = logging.getLogger("resilicity.scene_understanding")


@dataclass
class SidewalkZone:
    zone_name: str
    available_width_proxy: float
    relative_area_pct: float
    solar_exposure: float
    existing_shade: float
    existing_tree_density: float
    pedestrian_heat_priority: float
    tree_feasibility: float
    shade_structure_feasibility: float
    permeable_paver_feasibility: float
    conflict_objects_count: int
    confidence: float
    depth_mean: float


@dataclass
class RoadwayZone:
    road_width_proxy: float
    relative_area_pct: float
    solar_exposure: float
    existing_shade: float
    cool_pavement_feasibility: float
    traffic_conflict_density: float
    confidence: float
    depth_mean: float


@dataclass
class StreetGeometry:
    street_canyon_strength: float
    sky_visibility_proxy: float
    road_width_proxy: float
    building_wall_density: float
    height_to_width_proxy: float
    vanishing_point: List[float]
    horizon_y: float


@dataclass
class SceneUnderstanding:
    street_geometry: StreetGeometry
    left_sidewalk: SidewalkZone
    right_sidewalk: SidewalkZone
    roadway: RoadwayZone
    solar_exposure_proxy: float
    existing_shade_proxy: float
    heat_priority_summary: Dict[str, Any]
    protected_objects_summary: Dict[str, Any]
    design_constraints: List[str]
    confidence: float
    # Auxiliary runtime data (not serialized to basic JSON)
    depth_map: Optional[np.ndarray] = None
    heat_priority_map: Optional[np.ndarray] = None
    left_sidewalk_mask: Optional[np.ndarray] = None
    right_sidewalk_mask: Optional[np.ndarray] = None
    road_mask: Optional[np.ndarray] = None
    protected_mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean JSON serializable dictionary."""
        return {
            "street_geometry": asdict(self.street_geometry),
            "left_sidewalk": asdict(self.left_sidewalk),
            "right_sidewalk": asdict(self.right_sidewalk),
            "roadway": asdict(self.roadway),
            "solar_exposure_proxy": self.solar_exposure_proxy,
            "existing_shade_proxy": self.existing_shade_proxy,
            "heat_priority_summary": self.heat_priority_summary,
            "protected_objects_summary": self.protected_objects_summary,
            "design_constraints": self.design_constraints,
            "confidence": self.confidence,
        }


def analyze_scene(image: Image.Image, seg_result: Dict[str, Any]) -> SceneUnderstanding:
    """Construct deep SceneUnderstanding from image and SegFormer perception results."""
    w, h = image.size
    total_px = w * h

    # 1. Extract raw masks from SegFormer
    raw_preds = seg_result.get("raw_preds")
    if raw_preds is None:
        raw_preds = np.zeros((h, w), dtype=int)

    # Surface masks
    road_mask = np.isin(raw_preds, [6, 54, 91])
    # Pedestrian sidewalk pavement (excluding bare earth/sand/stairs)
    pavement_mask = np.isin(raw_preds, [3, 11, 52])
    vegetation_mask = np.isin(raw_preds, [4, 9, 17, 29, 66, 72])
    wall_mask = np.isin(raw_preds, [0, 1, 25, 48, 79, 84, 88])
    sky_mask = (raw_preds == 2)

    # Protected objects mask
    protected_mask = seg_result.get("protected_mask")
    if protected_mask is None:
        protected_mask = np.isin(raw_preds, [20, 80, 83, 102, 116, 127, 12, 43, 87, 93, 136, 69, 138, 8, 14, 38, 42])

    # 2. Monocular Depth Estimation
    depth_map = estimate_relative_depth(image)

    # 3. Geometry Analysis
    vanishing_point = estimate_vanishing_point(image, road_mask=road_mask)
    corridor_geom = analyze_street_corridor(
        width=w,
        height=h,
        road_mask=road_mask,
        pavement_mask=pavement_mask,
        wall_mask=wall_mask,
        sky_mask=sky_mask,
        vanishing_point=vanishing_point,
    )

    left_pave_mask = corridor_geom["left_sidewalk_mask"]
    right_pave_mask = corridor_geom["right_sidewalk_mask"]

    # 4. Solar & Shade Understanding
    solar_info = estimate_solar_and_shade(
        image=image,
        road_mask=road_mask,
        left_sidewalk_mask=left_pave_mask,
        right_sidewalk_mask=right_pave_mask,
        wall_mask=wall_mask,
    )

    # 5. Spatial Heat Priority Map
    heat_priority_map, heat_summary = compute_spatial_heat_priority_map(
        width=w,
        height=h,
        road_mask=road_mask,
        pavement_mask=pavement_mask,
        vegetation_mask=vegetation_mask,
        wall_mask=wall_mask,
        sun_mask=solar_info["sun_mask"],
        shade_mask=solar_info["shade_mask"],
        street_canyon_strength=corridor_geom["street_canyon_strength"],
    )

    # 6. Detailed Sidewalk & Roadway Characterization
    def _depth_stat(m: np.ndarray) -> float:
        return float(np.mean(depth_map[m])) if np.any(m) else 0.5

    # Conflict objects (vehicles or poles on or directly adjacent to sidewalks)
    left_conflicts = int(np.count_nonzero(protected_mask & left_pave_mask))
    right_conflicts = int(np.count_nonzero(protected_mask & right_pave_mask))

    # Existing tree density near each sidewalk
    # Dilate sidewalk slightly and check intersection with vegetation
    left_veg_density = float(np.mean(vegetation_mask[left_pave_mask])) if np.any(left_pave_mask) else 0.0
    right_veg_density = float(np.mean(vegetation_mask[right_pave_mask])) if np.any(right_pave_mask) else 0.0

    # Feasibility scores:
    # Tree canopy feasibility requires:
    # - sufficient available sidewalk width (width proxy >= 0.25)
    # - high solar exposure / need for shade
    # - low existing tree density
    # - low conflict with dense vehicles
    left_w = corridor_geom["left_sidewalk_width_proxy"]
    right_w = corridor_geom["right_sidewalk_width_proxy"]
    left_sol = solar_info["left_sidewalk"]["solar_exposure"]
    right_sol = solar_info["right_sidewalk"]["solar_exposure"]

    # For dense urban corridors, tree planting feasibility considers available verge/curbside space
    margin_w_left = max(left_w, 0.25 if (corridor_geom["building_wall_density"] > 0.20 and corridor_geom["road_width_proxy"] > 0.20) else 0.0)
    margin_w_right = max(right_w, 0.25 if (corridor_geom["building_wall_density"] > 0.20 and corridor_geom["road_width_proxy"] > 0.20) else 0.0)

    left_tree_feas = float(np.clip(
        0.35 * margin_w_left + 0.40 * left_sol + 0.25 * (1.0 - left_veg_density),
        0.15, 0.95,
    )) if (left_w >= 0.10 or margin_w_left >= 0.20) else 0.20

    right_tree_feas = float(np.clip(
        0.35 * margin_w_right + 0.40 * right_sol + 0.25 * (1.0 - right_veg_density),
        0.15, 0.95,
    )) if (right_w >= 0.10 or margin_w_right >= 0.20) else 0.20

    # Shade structure feasibility:
    # Favored if sidewalk exists but is narrower or dense street canyon creates overhead attachment opportunity
    left_shade_feas = float(np.clip(
        0.35 * left_sol + 0.35 * corridor_geom["street_canyon_strength"] + 0.30 * (0.8 if left_w < 0.35 else 0.4),
        0.0, 1.0,
    )) if left_w >= 0.08 else 0.10

    right_shade_feas = float(np.clip(
        0.35 * right_sol + 0.35 * corridor_geom["street_canyon_strength"] + 0.30 * (0.8 if right_w < 0.35 else 0.4),
        0.0, 1.0,
    )) if right_w >= 0.08 else 0.10

    # Permeable paver feasibility
    left_paver_feas = float(np.clip(left_w * 1.5, 0.0, 0.95))
    right_paver_feas = float(np.clip(right_w * 1.5, 0.0, 0.95))

    left_sidewalk = SidewalkZone(
        zone_name="left_sidewalk",
        available_width_proxy=left_w,
        relative_area_pct=corridor_geom["left_sidewalk_pct"],
        solar_exposure=left_sol,
        existing_shade=solar_info["left_sidewalk"]["existing_shade"],
        existing_tree_density=round(left_veg_density, 3),
        pedestrian_heat_priority=round(float(np.mean(heat_priority_map[left_pave_mask])) if np.any(left_pave_mask) else 0.5, 3),
        tree_feasibility=round(left_tree_feas, 3),
        shade_structure_feasibility=round(left_shade_feas, 3),
        permeable_paver_feasibility=round(left_paver_feas, 3),
        conflict_objects_count=left_conflicts,
        confidence=0.91 if left_w > 0.1 else 0.75,
        depth_mean=round(_depth_stat(left_pave_mask), 3),
    )

    right_sidewalk = SidewalkZone(
        zone_name="right_sidewalk",
        available_width_proxy=right_w,
        relative_area_pct=corridor_geom["right_sidewalk_pct"],
        solar_exposure=right_sol,
        existing_shade=solar_info["right_sidewalk"]["existing_shade"],
        existing_tree_density=round(right_veg_density, 3),
        pedestrian_heat_priority=round(float(np.mean(heat_priority_map[right_pave_mask])) if np.any(right_pave_mask) else 0.5, 3),
        tree_feasibility=round(right_tree_feas, 3),
        shade_structure_feasibility=round(right_shade_feas, 3),
        permeable_paver_feasibility=round(right_paver_feas, 3),
        conflict_objects_count=right_conflicts,
        confidence=0.91 if right_w > 0.1 else 0.75,
        depth_mean=round(_depth_stat(right_pave_mask), 3),
    )

    road_w = corridor_geom["road_width_proxy"]
    road_sol = solar_info["roadway"]["solar_exposure"]
    roadway = RoadwayZone(
        road_width_proxy=road_w,
        relative_area_pct=round(float(np.count_nonzero(road_mask) / total_px * 100.0), 2),
        solar_exposure=road_sol,
        existing_shade=solar_info["roadway"]["existing_shade"],
        cool_pavement_feasibility=round(float(np.clip(road_w * 1.2 * road_sol, 0.2, 0.96)), 3),
        traffic_conflict_density=round(float(np.count_nonzero(protected_mask & road_mask) / max(1, np.count_nonzero(road_mask))), 3),
        confidence=0.94 if road_w > 0.2 else 0.80,
        depth_mean=round(_depth_stat(road_mask), 3),
    )

    # 7. Design Constraints
    constraints: List[str] = []
    if corridor_geom["street_canyon_strength"] > 0.65:
        constraints.append("High street canyon enclosure: prioritize pedestrian verges and high-reflectance surfaces to avoid heat trapping.")
    if road_w > 0.50:
        constraints.append("Broad asphalt road corridor: maintain active traffic lane boundaries while treating central road plane.")
    if left_w < 0.15 and right_w < 0.15:
        constraints.append("Narrow pedestrian sidewalk verges: prefer compact planting pits or tensile shade canopies over expansive tree basins.")
    if np.count_nonzero(protected_mask) > total_px * 0.05:
        constraints.append("Detected active vehicles/pedestrians: strictly preserve all vehicle bodies, pedestrians, and traffic fixtures.")

    street_geom = StreetGeometry(
        street_canyon_strength=corridor_geom["street_canyon_strength"],
        sky_visibility_proxy=corridor_geom["sky_visibility_proxy"],
        road_width_proxy=corridor_geom["road_width_proxy"],
        building_wall_density=corridor_geom["building_wall_density"],
        height_to_width_proxy=corridor_geom["height_to_width_proxy"],
        vanishing_point=corridor_geom["vanishing_point"],
        horizon_y=corridor_geom["horizon_y"],
    )

    # Protected objects breakdown
    prot_summary = {
        item.get("id", "item"): item.get("percentage", 0.0)
        for item in seg_result.get("protected_objects", [])
    }

    return SceneUnderstanding(
        street_geometry=street_geom,
        left_sidewalk=left_sidewalk,
        right_sidewalk=right_sidewalk,
        roadway=roadway,
        solar_exposure_proxy=solar_info["solar_exposure_proxy"],
        existing_shade_proxy=solar_info["existing_shade_proxy"],
        heat_priority_summary=heat_summary,
        protected_objects_summary=prot_summary,
        design_constraints=constraints,
        confidence=0.92,
        depth_map=depth_map,
        heat_priority_map=heat_priority_map,
        left_sidewalk_mask=left_pave_mask,
        right_sidewalk_mask=right_pave_mask,
        road_mask=road_mask,
        protected_mask=protected_mask,
    )
