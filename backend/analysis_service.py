"""Photo -> segmentation -> scene understanding -> gated plan -> thermal impact, without image generation.

Shared by /api/v1/analyze-and-redesign and project creation.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from PIL import Image

from image_generation import SceneAnalysis, SpatialDesignPlan
from legacy_heat import compute_thermal_impact
from planner import analyze_scene_heuristics, generate_spatial_plan
from scene_understanding import SceneUnderstanding, analyze_scene
from segmentation import SegFormerEngine

logger = logging.getLogger("resilicity.analysis")

# Used only when segmentation fails, and flagged as assumed in the result.
FALLBACK_SURFACES = {"road": 35.0, "wall": 25.0, "pavement": 15.0, "vegetation": 5.0, "roof": 0.0}


@dataclass
class SiteAnalysisResult:
    surfaces: Dict[str, float]
    segmentation_ok: bool
    seg_result: Dict[str, Any]
    scene: SceneUnderstanding
    plan: SpatialDesignPlan
    analysis: SceneAnalysis
    thermal: Dict[str, Any]


def run_site_analysis(
    image: Image.Image,
    design_profile: Optional[str] = None,
    road_width_m: Optional[float] = None,
    climate_zone: Optional[str] = None,
    budget_lakh: Optional[float] = None,
) -> SiteAnalysisResult:
    seg_engine = SegFormerEngine.get_instance()
    surfaces: Dict[str, float] = {}
    seg_res: Dict[str, Any] = {}
    segmentation_ok = True
    try:
        seg_res = seg_engine.segment_image(image)
        seg_engine.offload_to_cpu()
        for m in seg_res.get("masks", []):
            surfaces[m["className"]] = m["areaPercentage"]
    except Exception as e:  # keep the legacy endpoint's behaviour, but say so
        logger.warning("Segmentation failed, using assumed surfaces: %s", e)
        surfaces, segmentation_ok = dict(FALLBACK_SURFACES), False

    scene = analyze_scene(image, seg_res)
    plan = generate_spatial_plan(
        image,
        surfaces,
        seg_result=seg_res,
        design_profile=design_profile,  # type: ignore[arg-type]
        road_width_m=road_width_m,
        climate_zone=climate_zone,
        budget_lakh=budget_lakh,
    )
    analysis = analyze_scene_heuristics(image, surfaces)
    thermal = compute_thermal_impact(surfaces, [i.dict() for i in plan.interventions])
    return SiteAnalysisResult(surfaces, segmentation_ok, seg_res, scene, plan, analysis, thermal)
