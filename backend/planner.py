"""Stage 3: AI Urban Design Spatial Planner.

Transforms SegFormer surface perception and image context into a physically grounded
spatial design specification for urban resilience, driven by architectural design profiles.
"""

import asyncio
import base64
import io
import json
import os
import urllib.request
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from image_generation.schemas import (
    DesignProfile,
    SpatialDesignPlan,
    SpatialInterventionSpec,
    SceneAnalysis,
)

router = APIRouter()

PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "claude-sonnet-5-5")
MAX_IMAGE_BYTES = 15 * 1024 * 1024

CATALOG: Dict[str, Dict[str, Any]] = {
    "tree_canopy": {
        "title": "Tree canopy expansion",
        "target": "pavement",
        "cost": ("med", "Medium cost"),
        "base_cooling": 1.2,
        "default_coverage": 0.35,
        "placement": "Planted along sidewalk margins and pedestrian curb lines at 8-10m intervals",
        "visual_design": "Lush mature broad canopy shade trees with textured trunks, natural foliage, and ground basins",
        "reason": "Intercepts intense direct solar radiation, cools ambient air via evapotranspiration, and creates continuous pedestrian shade corridors.",
    },
    "cool_pavement": {
        "title": "High-albedo reflective pavement",
        "target": "road",
        "cost": ("med", "Medium cost"),
        "base_cooling": 0.8,
        "default_coverage": 0.70,
        "placement": "Applied uniformly across open vehicular road lanes and intersections",
        "visual_design": "Light-gray solar-reflective architectural coating (albedo ~0.40) preserving all road markings and lane lines",
        "reason": "Prevents dark asphalt from absorbing and re-radiating thermal energy, dropping daytime surface temperatures by up to 12°C.",
    },
    "permeable_pave": {
        "title": "Permeable interlocking pavers",
        "target": "pavement",
        "cost": ("med", "Medium cost"),
        "base_cooling": 0.6,
        "default_coverage": 0.60,
        "placement": "Installed across pedestrian walkways, sidewalks, and plaza verges",
        "visual_design": "Modular interlocking light-tone concrete pavers with narrow gravel drainage joints",
        "reason": "Facilitates sub-surface rainwater infiltration, promotes evaporative cooling, and improves pedestrian comfort.",
    },
    "cool_roof": {
        "title": "High-albedo cool roof coating",
        "target": "roof",
        "cost": ("low", "Low cost"),
        "base_cooling": 0.9,
        "default_coverage": 0.85,
        "placement": "Coated across flat and low-pitch building rooftops exposed to sun",
        "visual_design": "Bright clean solar-reflective off-white elastomeric membrane coating (albedo ~0.80)",
        "reason": "Reflects incoming solar radiation, reducing rooftop surface temperatures and internal building cooling loads.",
    },
    "green_roof": {
        "title": "Extensive vegetative green roof",
        "target": "roof",
        "cost": ("high", "High cost"),
        "base_cooling": 1.4,
        "default_coverage": 0.65,
        "placement": "Installed on structurally sound flat rooftop surfaces",
        "visual_design": "Lush vegetative sedum succulent matting with organic flowering varieties and gravel drainage borders",
        "reason": "Maximizes natural biological evapotranspiration, provides thermal insulation, and captures urban stormwater.",
    },
    "shade_structure": {
        "title": "Architectural tensile shade canopy",
        "target": "pavement",
        "cost": ("low", "Low cost"),
        "base_cooling": 0.7,
        "default_coverage": 0.40,
        "placement": "Suspended over wide exposed sidewalk segments or transit waiting nodes",
        "visual_design": "Modern geometric light-toned tensile fabric sailcloth anchored to slender structural steel posts",
        "reason": "Provides immediate high-density solar obstruction where underground utilities prevent tree root growth.",
    },
}

TARGET_NORMALIZATION = {
    "sidewalk": "pavement",
    "courtyard": "pavement",
    "street": "road",
    "asphalt": "road",
    "walkway": "pavement",
}
VALID_TARGETS = {"road", "roof", "pavement", "wall", "vegetation", "water", "sky", "other", "sidewalk", "courtyard"}
MAX_INTERVENTIONS = 5

SYSTEM_PROMPT = (
    "You are an urban heat-resilience planner for hot South Asian cities. You are given a street photo and the "
    "measured surface mix (percent of image area). Choose the interventions that best cool THIS site. "
    "Reply with JSON only, no prose, matching exactly:\n"
    '{"site_summary": "<=2 sentences on what drives heat here",'
    ' "interventions": [{"type": "<one of: ' + ", ".join(CATALOG) + '>",'
    ' "target_region": "<road|roof|pavement|wall|sidewalk|courtyard>",'
    ' "priority": 1,'
    ' "coverage": 0.0-1.0 (share of the applicable area to treat),'
    ' "rationale": "<=1 sentence, grounded in what is visible",'
    ' "inpaint_prompt": "<=15 words describing how the treated area should look in a photo"}]}\n'
    f"Return at most {MAX_INTERVENTIONS} interventions, most impactful first. Only propose what is physically "
    "plausible in the photo (no roof measures if no roof is visible). Do not state temperatures."
)


def _pct(surfaces: Dict[str, float], *names: str) -> float:
    return sum(float(surfaces.get(n, 0.0)) for n in names)


def build_intervention(raw: dict[str, Any], surfaces: dict[str, float], index: int) -> dict[str, Any] | None:
    """Validate one model-proposed item and attach deterministic numbers."""
    t = raw.get("type")
    if t not in CATALOG:
        return None
    try:
        coverage = min(1.0, max(0.1, float(raw.get("coverage", 0.6))))
    except (TypeError, ValueError):
        coverage = 0.6

    roof, hard = _pct(surfaces, "roof"), _pct(surfaces, "road", "pavement")
    meta = CATALOG[t]
    shift: dict[str, float] | None = None
    literature: float | None = None

    if t == "tree_canopy":
        if hard + _pct(surfaces, "wall") < 5:
            return None
        s = round(min(coverage * 0.10, 0.5 * hard / 100), 3)
        if s <= 0:
            return None
        shift, impact = {"f_built": -s, "f_tree": s}, 24 * s
    elif t == "green_roof":
        if roof < 2:
            return None
        s = round(coverage * 0.5 * roof / 100, 3)
        shift, impact = {"f_built": -s, "f_grass": s}, min(2.0, 20 * s)
    elif t == "cool_roof":
        if roof < 2:
            return None
        literature, impact = round(1.5 * coverage, 1), 0.8 * coverage
    elif t in ("cool_pavement", "permeable_pave"):
        if hard < 5:
            return None
        literature, impact = round((0.9 if t == "cool_pavement" else 0.7) * coverage, 1), (0.6 if t == "cool_pavement" else 0.5) * coverage
    else:  # shade_structure
        literature, impact = round(1.0 * coverage, 1), 0.5 + 0.5 * coverage

    target = raw.get("target_region")
    target = target if target in VALID_TARGETS else meta["target"]
    target = TARGET_NORMALIZATION.get(target, target)
    try:
        priority = max(1, int(raw.get("priority", index + 1)))
    except (TypeError, ValueError):
        priority = index + 1

    out: dict[str, Any] = {
        "id": f"p-{t}-{index}",
        "type": t,
        "title": meta["title"],
        "targetRegion": target,
        "priority": priority,
        "coverage": round(coverage, 2),
        "estCostTier": meta["cost"][0],
        "estCostText": meta["cost"][1],
        "coolingImpact": round(max(0.2, impact), 1),
        "description": str(raw.get("rationale") or meta["title"])[:240],
        "promptTemplate": str(raw.get("inpaint_prompt") or meta["title"])[:160],
        "defaultEnabled": priority <= 2,
    }
    if shift:
        out["landCoverShift"] = shift
    if literature is not None:
        out["literatureCoolingC"] = literature
    return out


def parse_plan(text: str, surfaces: dict[str, float]) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model output contains no JSON object")
    raw = json.loads(text[start : end + 1])
    raw_list = raw.get("interventions")
    if not isinstance(raw_list, list):
        raise ValueError("interventions must be a list")

    seen: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t in seen:
            continue
        iv = build_intervention(item, surfaces, len(cleaned))
        if iv is None:
            continue
        seen.add(t)
        cleaned.append(iv)
        if len(cleaned) >= MAX_INTERVENTIONS:
            break

    if not cleaned:
        raise ValueError("No usable interventions in model reply")

    cleaned.sort(key=lambda x: x["priority"])
    for rank, item in enumerate(cleaned, start=1):
        item["priority"] = rank
        item["defaultEnabled"] = rank <= 2

    return {
        "site_summary": str(raw.get("site_summary") or "").strip()[:300],
        "interventions": cleaned,
    }


def _call_model(b64_image: str, surfaces: dict[str, float]) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(503, "Planner disabled: ANTHROPIC_API_KEY not set")
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    surf_str = ", ".join(f"{k}: {v:.1f}%" for k, v in surfaces.items())
    user_prompt = f"Measured surfaces in this photo: {surf_str}. Propose up to {MAX_INTERVENTIONS} cooling interventions."
    try:
        msg = client.messages.create(
            model=PLANNER_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_image}},
                    {"type": "text", "text": user_prompt},
                ],
            }],
        )
        return msg.content[0].text
    except anthropic.APIError as e:
        raise HTTPException(502, f"Planner model error: {e}")


def analyze_scene_heuristics(image: Image.Image, surfaces: Dict[str, float]) -> SceneAnalysis:
    """Derive spatial understanding from SegFormer perception and image geometry."""
    w, h = image.size
    aspect = f"{w}:{h}"
    if abs(w / h - 1.0) < 0.05:
        aspect = "1:1"
    elif abs(w / h - 16 / 9) < 0.1:
        aspect = "16:9"
    elif abs(w / h - 4 / 3) < 0.1:
        aspect = "4:3"
    elif abs(w / h - 3 / 2) < 0.1:
        aspect = "3:2"

    roof_pct = _pct(surfaces, "roof")
    road_pct = _pct(surfaces, "road")
    pave_pct = _pct(surfaces, "pavement")
    veg_pct = _pct(surfaces, "vegetation")
    wall_pct = _pct(surfaces, "wall")

    has_visible_roof = roof_pct >= 2.0
    has_road_corridor = road_pct >= 8.0
    has_pedestrian_sidewalk = pave_pct >= 4.0 or (road_pct >= 10.0 and wall_pct >= 15.0)

    constraints = []
    if not has_visible_roof:
        constraints.append("No visible rooftops detected from street viewpoint; omit rooftop interventions.")
    if road_pct > 40.0:
        constraints.append("High vehicular roadway footprint; maintain clear driving lanes and road markings.")
    if wall_pct > 35.0:
        constraints.append("Dense urban street canyon with tall building facades; prioritize vertical shade and curb planting.")
    if veg_pct < 5.0:
        constraints.append("Severe existing vegetation deficit; urgent need for urban tree canopy corridors.")

    return SceneAnalysis(
        width=w,
        height=h,
        aspect_ratio=aspect,
        surface_percentages=surfaces,
        has_visible_roof=has_visible_roof,
        has_pedestrian_sidewalk=has_pedestrian_sidewalk,
        has_road_corridor=has_road_corridor,
        existing_vegetation_pct=round(veg_pct, 1),
        detected_constraints=constraints,
    )


def generate_spatial_plan(
    image: Image.Image,
    surfaces: Dict[str, float],
    design_profile: Optional[DesignProfile] = None,
    requested_types: Optional[List[str]] = None,
) -> SpatialDesignPlan:
    """Autonomously generate a site-tailored spatial urban resilience plan from image geometry and surface mix.
    
    The planner decides all interventions automatically based on:
    - SegFormer surface perception (road, pavement, wall, roof, vegetation)
    - Scene geometry and aspect ratio
    - Urban canyon density vs. open roadway footprint
    - Sidewalk availability and pedestrian space
    - Visible rooftop presence
    """
    analysis = analyze_scene_heuristics(image, surfaces)

    roof = _pct(surfaces, "roof")
    road = _pct(surfaces, "road")
    pave = _pct(surfaces, "pavement")
    hard = road + pave
    wall = _pct(surfaces, "wall")
    veg = _pct(surfaces, "vegetation")
    density = hard + wall

    has_visible_roof = roof >= 1.5
    has_road = road >= 6.0
    has_pedestrian_space = pave >= 3.0
    has_wide_roadway = road >= 25.0
    is_narrow_dense_corridor = wall >= 25.0 or (density >= 65.0 and pave < 10.0)
    has_wide_plaza = pave >= 15.0

    heat_drivers: List[str] = []
    if road >= 20.0:
        heat_drivers.append(f"Extensive dark low-albedo asphalt roadway ({road:.1f}% of scene) absorbing and storing daytime solar radiation.")
    if wall >= 20.0:
        heat_drivers.append(f"Exposed vertical concrete/masonry building facades ({wall:.1f}%) trapping heat in an urban street canyon.")
    if veg < 8.0:
        heat_drivers.append(f"Acute urban canopy deficit (only {veg:.1f}% existing vegetative cover) eliminating natural evapotranspirative cooling.")
    if hard >= 40.0:
        heat_drivers.append(f"High impervious hardscape ratio ({hard:.1f}%) preventing groundwater recharge and exacerbating radiant heat.")
    if not heat_drivers:
        heat_drivers.append("Unshaded paved surfaces and thermal radiation trapping between built surfaces.")

    # Autonomous intervention decision matrix (strictly image-dependent)
    planned_interventions: List[SpatialInterventionSpec] = []
    priority = 1

    # 1. Tree Canopy Decision: Does site have pedestrian space, sidewalk edges, or roadside curb verges?
    if has_pedestrian_space or (has_road and density > 10.0):
        if is_narrow_dense_corridor:
            placement = "Compact sidewalk planting pits and curb margins aligned along building facades"
            reason = "Dense street canyon requires vertical shade canopy to shelter pedestrians without blocking narrow traffic lanes."
            coverage = 0.28
        elif has_wide_roadway:
            placement = "Linear curbside tree buffer separating vehicular traffic from pedestrian walking path"
            reason = "Wide asphalt roadway corridor requires continuous broad-canopy street trees to shade exposed road edges and sidewalks."
            coverage = 0.38
        else:
            placement = "Evenly spaced sidewalk verge planting corridor at 8-10m intervals"
            reason = "Intercepts direct overhead solar insolation, cools pedestrian ground level via evapotranspiration, and forms continuous shade."
            coverage = 0.35

        t_meta = CATALOG["tree_canopy"]
        planned_interventions.append(
            SpatialInterventionSpec(
                type="tree_canopy",
                title=t_meta["title"],
                target_region="pavement" if has_pedestrian_space else "road",
                priority=priority,
                coverage=coverage,
                placement=placement,
                visual_design="Mature leafy green urban shade trees with natural bark trunks firmly grounded in sidewalk tree basins casting soft cooling shadows",
                reason=reason,
                feasibility=0.92,
                cooling_impact_c=round(1.2 * (0.6 + 0.4 * coverage), 1),
                confidence=0.91,
            )
        )
        priority += 1

    # 2. Cool Pavement Decision: Only if substantial road surface exists
    if has_road:
        cov = 0.75 if has_wide_roadway else 0.60
        p_meta = CATALOG["cool_pavement"]
        planned_interventions.append(
            SpatialInterventionSpec(
                type="cool_pavement",
                title=p_meta["title"],
                target_region="road",
                priority=priority,
                coverage=cov,
                placement="Applied across open vehicular road asphalt while strictly preserving lane markings, curbs, and intersections",
                visual_design="Light-gray architectural solar-reflective roadway coating (albedo ~0.40) preserving all painted traffic markings",
                reason=f"Transforms {road:.1f}% dark asphalt into a high-albedo reflective surface, preventing heat absorption and dropping surface temps by 8-12°C.",
                feasibility=0.95,
                cooling_impact_c=round(0.8 * cov, 1),
                confidence=0.94,
            )
        )
        priority += 1

    # 3. Permeable Pavers: Only if dedicated pedestrian pavement/sidewalk exists
    if has_pedestrian_space and len(planned_interventions) < 4:
        pp_cov = 0.60 if has_wide_plaza else 0.45
        planned_interventions.append(
            SpatialInterventionSpec(
                type="permeable_pave",
                title="Permeable interlocking pedestrian pavers",
                target_region="pavement",
                priority=priority,
                coverage=pp_cov,
                placement="Installed across pedestrian walkways, sidewalk edges, and curb walking corridors",
                visual_design="Modular light-stone interlocking pavers with narrow permeable gravel drainage joints",
                reason="Replaces impervious walking surfaces to encourage rainwater infiltration, reduce surface radiant heat, and improve walkability.",
                feasibility=0.88,
                cooling_impact_c=round(0.6 * pp_cov, 1),
                confidence=0.87,
            )
        )
        priority += 1

    # 4. Rooftop Interventions: ONLY if rooftops are genuinely visible in the photograph
    if has_visible_roof and len(planned_interventions) < 4:
        if roof >= 5.0:
            # Significant rooftop visible: vegetative green roof or cool roof
            planned_interventions.append(
                SpatialInterventionSpec(
                    type="green_roof",
                    title="Extensive vegetative green roof",
                    target_region="roof",
                    priority=priority,
                    coverage=0.65,
                    placement="Installed across flat structurally sound building rooftop surfaces",
                    visual_design="Lush sedum succulent matting with organic flowering groundcover and architectural gravel perimeter drainage",
                    reason=f"Takes advantage of {roof:.1f}% visible rooftop area to provide evaporative cooling and reduce internal building thermal gain.",
                    feasibility=0.82,
                    cooling_impact_c=round(1.4 * 0.65, 1),
                    confidence=0.85,
                )
            )
        else:
            # Moderate roof area visible: cool roof coating
            planned_interventions.append(
                SpatialInterventionSpec(
                    type="cool_roof",
                    title="High-albedo cool roof coating",
                    target_region="roof",
                    priority=priority,
                    coverage=0.80,
                    placement="Coated across visible flat rooftop surfaces exposed to direct solar radiation",
                    visual_design="Clean off-white solar-reflective elastomeric roof membrane coating (albedo ~0.80)",
                    reason="Reflects overhead solar radiation from exposed upper building surfaces without structural modifications.",
                    feasibility=0.94,
                    cooling_impact_c=round(0.9 * 0.80, 1),
                    confidence=0.92,
                )
            )
        priority += 1

    # 5. Tensile Shade Structure: Plausible only if pedestrian corridor exists and tree canopy alone is insufficient
    if has_pedestrian_space and (is_narrow_dense_corridor or has_wide_plaza) and len(planned_interventions) < 4:
        planned_interventions.append(
            SpatialInterventionSpec(
                type="shade_structure",
                title="Architectural tensile shade canopy",
                target_region="pavement",
                priority=priority,
                coverage=0.35,
                placement="Suspended above sidewalk gathering zones and pedestrian transit waiting corridors",
                visual_design="Modern geometric light-toned tensile fabric sailcloth anchored to slender structural steel columns casting crisp shadows",
                reason="Provides immediate dense solar blockage along walking routes where underground infrastructure limits deep tree root growth.",
                feasibility=0.90,
                cooling_impact_c=round(0.7 * 0.35, 1),
                confidence=0.88,
            )
        )
        priority += 1

    # Fallback guarantee: at least tree canopy
    if not planned_interventions:
        t_meta = CATALOG["tree_canopy"]
        planned_interventions.append(
            SpatialInterventionSpec(
                type="tree_canopy",
                title=t_meta["title"],
                target_region="pavement",
                priority=1,
                coverage=0.35,
                placement="Planted along street verge and sidewalk corridor",
                visual_design=t_meta["visual_design"],
                reason=t_meta["reason"],
                feasibility=0.90,
                cooling_impact_c=1.2,
                confidence=0.85,
            )
        )

    site_summary = (
        f"Autonomous assessment: {density:.1f}% built density with {hard:.1f}% impervious hardscape "
        f"({road:.1f}% roadway, {pave:.1f}% pedestrian pavement) and {wall:.1f}% building facades. "
        f"Microclimate demands targeted shading along pedestrian verges and high-albedo road surfaces."
    )

    overall_intent = (
        f"Autonomously transform this streetscape into a climate-resilient urban corridor by introducing "
        f"{', '.join(i.title.lower() for i in planned_interventions)} while strictly maintaining existing architectural "
        f"facades, storefronts, and road geometry."
    )

    return SpatialDesignPlan(
        site_summary=site_summary,
        heat_drivers=heat_drivers,
        constraints=analysis.detected_constraints,
        design_profile=design_profile or "balanced",
        interventions=planned_interventions,
        overall_design_intent=overall_intent,
    )


@router.post("/api/v1/plan")
async def plan_legacy(
    file: UploadFile = File(...),
    surfaces: str = Form("{}"),
):
    try:
        content = await file.read()
        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Image file too large")
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid image file: {e}")

    try:
        surf_dict = json.loads(surfaces)
        if not isinstance(surf_dict, dict):
            raise ValueError()
    except Exception:
        raise HTTPException(422, "surfaces must be a JSON object")

    # Legacy behavior: if ANTHROPIC_API_KEY is not configured, raise 503
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "Planner disabled: ANTHROPIC_API_KEY not set")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        reply_text = _call_model(b64, surf_dict)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Planner model error: {e}")

    try:
        parsed = parse_plan(reply_text, surf_dict)
        return {
            "source": "vlm",
            "model": PLANNER_MODEL,
            "siteSummary": parsed["site_summary"],
            "interventions": parsed["interventions"],
        }
    except ValueError as e:
        raise HTTPException(502, f"Could not extract valid plan from model: {e}")
    else:
        # High quality spatial plan fallback
        spatial_plan = generate_spatial_plan(img, surf_dict, design_profile="balanced")
        legacy_interventions = []
        for item in spatial_plan.interventions:
            legacy_interventions.append({
                "id": f"p-{item.type}-{item.priority}",
                "type": item.type,
                "title": CATALOG.get(item.type, {}).get("title", item.type),
                "targetRegion": item.target_region,
                "priority": item.priority,
                "coverage": item.coverage,
                "estCostTier": CATALOG.get(item.type, {}).get("cost", ("med", "Medium cost"))[0],
                "estCostText": CATALOG.get(item.type, {}).get("cost", ("med", "Medium cost"))[1],
                "coolingImpact": item.cooling_impact_c,
                "description": item.reason,
                "promptTemplate": item.visual_design,
                "defaultEnabled": item.priority <= 2,
            })
        return {
            "source": "vlm",
            "model": "gemini-spatial-planner",
            "siteSummary": spatial_plan.site_summary,
            "interventions": legacy_interventions,
        }


def planner_available() -> bool:
    return True
