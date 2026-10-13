"""Spatial prompt synthesis for Generative AI urban redesign."""

from typing import List, Optional

from buildable import get_spec
from buildable.config import RENDERING

from .schemas import SpatialDesignPlan, SpatialInterventionSpec, DesignProfile


PROFILE_GUIDANCE = {
    "balanced": (
        "Strategy: Balanced resilient urban redesign. Mix native shade trees along walking corridors, "
        "high-albedo cool pavement on road surfaces, and architectural shade structures where appropriate."
    ),
    "pedestrian_first": (
        "Strategy: Pedestrian-first climate shelter. Prioritize dense pedestrian walkway shade, wide comfortable "
        "sidewalk tree canopies, permeable stone pavers underfoot, and continuous pedestrian thermal relief."
    ),
    "maximum_cooling": (
        "Strategy: Aggressive maximum heat island mitigation. Maximize high-albedo solar-reflective coatings on all asphalt, "
        "plant mature spreading shade trees, and deploy continuous tensile shade structures over exposed areas."
    ),
    "green_infrastructure": (
        "Strategy: Extensive urban greening and biophilic infrastructure. Maximize native street trees, vegetative planting strips, "
        "and extensive sedum green roofs on visible building rooftops."
    ),
    "low_cost": (
        "Strategy: High-efficiency, low-capital interventions. Prioritize reflective high-albedo pavement sealcoats, "
        "targeted fast-growing street tree planting, and modular tensile fabric shade sails."
    ),
}


INTERVENTION_PROMPT_DESCRIPTIONS = {
    "tree_canopy": (
        "Young nursery-stock street trees about 4 m tall with slim trunks and modest canopies, planted in square "
        "tree pits with metal grates at the kerb edge of the sidewalk, evenly spaced, leaving the walkway clear. "
        "Do not place trees inside buildings, through vehicles, or in active traffic lanes."
    ),
    "cool_pavement": (
        "High-albedo solar-reflective road pavement coating. The weathered dark bitumen asphalt is transformed into "
        "a clean, modern light-gray architectural solar-reflective surface (albedo ~0.40). Crucially, preserve all existing "
        "lane markings, crosswalks, painted road symbols, curb edges, manhole covers, and street perspective."
    ),
    "permeable_pave": (
        "Interlocking modular permeable stone/concrete pavers along sidewalks and pedestrian paths. Distinct clean "
        "architectural paving stones in warm light-stone tones with narrow drainage joints, replacing broken asphalt or cracked walkways."
    ),
    "cool_roof": (
        "High-reflectance titanium cool roof coating on visible flat or low-pitch rooftops. Bright clean solar-reflective "
        "off-white coating (albedo ~0.80) that reflects solar radiation while preserving rooftop structural geometry and fixtures."
    ),
    "green_roof": (
        "Extensive vegetative green roof on visible building rooftops. Lush, varied sedum succulent plant mats and hardy "
        "low-water vegetation covering the roof area with neat architectural gravel perimeter borders."
    ),
    "shade_structure": (
        "Modern architectural tensile fabric shade canopy structures over pedestrian gathering zones or walkways. "
        "Crisp geometric light-colored sailcloth suspended on slender steel support columns, casting angled geometric "
        "cooling shade on the ground."
    ),
}


NEGATIVE_PROMPTS = (
    "cartoon, drawing, anime, 3d render sticker, neon overlay, colored circles, geometric stamps, "
    "floating trees, trees growing out of buildings, trees blocking traffic, deformed vehicles, "
    "altered building facades, shifted horizon, warped perspective, low resolution, blurry, artifacting, "
    "unrealistic jungle, sci-fi fantasy buildings, ivy-covered buildings, overgrown facades, climbing plants on walls, "
    "giant mature trees"
)


def build_redesign_prompt(plan: SpatialDesignPlan) -> str:
    """Build a comprehensive, spatially constrained image editing prompt for Gemini.
    
    The prompt instructs Gemini to redesign the specific photo into a heat-resilient version
    while strictly preserving the architectural and spatial identity of the original location.
    """
    profile_text = PROFILE_GUIDANCE.get(plan.design_profile, PROFILE_GUIDANCE["balanced"])

    # Detail each planned intervention
    intervention_clauses: List[str] = []
    for idx, item in enumerate(plan.interventions, 1):
        spec = get_spec(item.type)
        base_desc = INTERVENTION_PROMPT_DESCRIPTIONS.get(
            item.type, spec.kontext_phrase("the street").capitalize() + "." if spec else item.visual_design
        )
        clause = (
            f"Intervention {idx} [{item.type.replace('_', ' ').title()}]: {base_desc} "
            f"Location: {item.placement}. Visual Details: {item.visual_design} (Coverage: {int(item.coverage * 100)}%)."
        )
        intervention_clauses.append(clause)

    interventions_block = "\n".join(intervention_clauses) if intervention_clauses else (
        "Add native mature urban shade trees along the sidewalk and apply light-gray solar-reflective cool pavement coating."
    )

    constraints_block = ""
    if plan.constraints:
        constraints_block = "PHYSICAL CONSTRAINTS TO RESPECT:\n- " + "\n- ".join(plan.constraints)

    prompt = f"""You are an elite urban architectural designer and landscape architect specializing in urban heat resilience.
Transform this uploaded street photograph into an architecturally realistic, heat-resilient version of THIS EXACT STREET.

CRITICAL IDENTITY PRESERVATION RULES:
1. PRESERVE THE SCENE IDENTITY: Keep the exact same buildings, architecture, storefronts, window patterns, camera viewpoint, horizon, perspective, street layout, and lighting conditions.
2. DO NOT REPLACE THE CITY: This must look like a real photograph of this exact physical location after urban cooling infrastructure has been installed.
3. PHYSICAL REALISM: All added elements must be physically plausible, properly scaled in perspective, and cast natural shadows matching the ambient sunlight in the photo.

DESIGN STRATEGY:
{profile_text}

SPECIFIC INTERVENTIONS TO IMPLEMENT:
{interventions_block}

{constraints_block}

OVERALL REDESIGN INTENT:
{plan.overall_design_intent}

OUTPUT QUALITY REQUIREMENT:
Generate a single photorealistic, high-resolution architectural photograph showing the transformed site with high material fidelity and natural photographic quality."""

    return prompt.strip()


from .schemas import SpatialDesignPlan, SpatialInterventionSpec, DesignProfile, RefinementIntent


def build_refinement_prompt(
    user_instruction: str,
    intent: Optional[RefinementIntent] = None,
    context: Optional[str] = None,
) -> str:
    """Build a precise, spatially grounded refinement prompt for Gemini multimodal image editing.
    
    Transforms the current generated design according to the user's natural-language instruction,
    incorporating structured intent (additions, removals, modifications) while strictly preserving
    the architectural and spatial identity of the original location.
    """
    intent_lines: List[str] = []
    if intent:
        if intent.goal:
            intent_lines.append(f"CORE GOAL: {intent.goal}")
        if intent.add:
            intent_lines.append(f"ELEMENTS TO ADD: {', '.join(intent.add)}")
        if intent.remove:
            intent_lines.append(f"ELEMENTS TO REMOVE / REDUCE: {', '.join(intent.remove)}")
        if intent.modify:
            intent_lines.append(f"MODIFICATIONS: {', '.join(intent.modify)}")
        if intent.spatial_constraints:
            intent_lines.append(f"SPATIAL RESTRAINTS: {'; '.join(intent.spatial_constraints)}")

    intent_block = "\n".join(intent_lines) if intent_lines else f"EDIT REQUEST: {user_instruction}"

    preserve_list = intent.preserve if (intent and intent.preserve) else [
        "exact camera viewpoint, horizon line, and 3D street perspective",
        "existing buildings, architecture, storefronts, and window patterns",
        "road alignment, curb lines, and vehicular traffic corridors",
        "vehicles, utility poles, streetlights, and existing structural elements",
        "consistent ambient sunlight angle, natural shadows, and sky lighting",
    ]
    preserve_block = "\n- ".join(preserve_list)

    prompt = f"""CURRENT SCENE:
This is an AI-generated urban resilience redesign photograph of a real urban location.

USER REQUEST:
"{user_instruction}"

STRUCTURED DESIGN INTENT:
{intent_block}

PRESERVE (CRITICAL ARCHITECTURAL CONSTRAINTS):
- {preserve_block}

EDIT:
- Only make the visual modifications explicitly requested by the user and outlined above.
- If elements are to be removed or reduced (e.g. removing a pergola or reducing trees), cleanly reconstruct the underlying sidewalk, curb, or road surface seamlessly.
- If elements are to be added (e.g. trees, seating, bioswales, light permeable pavers), ensure they are physically plausible, placed in realistic ground positions (along sidewalk verges or pedestrian zones), and do not obstruct active traffic lanes.

REALISM:
- Photorealistic architectural quality
- Physically plausible materials, foliage, and structural supports
- Correct depth, scale, and perspective alignment with the existing street canyon
- Believable cast shadows matching the ambient sunlight direction in the photo
- Consistent pavement and curb connections with no floating artifacts

DO NOT:
- Do NOT regenerate the city or change the location.
- Do NOT move, distort, or replace existing buildings or storefronts unless explicitly asked.
- Do NOT shift the camera viewpoint or perspective.
- Do NOT create floating objects, duplicated trees, or unrealistic fantasy elements.
- Do NOT add text labels, watermarks, colored overlay masks, or geometric stickers.

OUTPUT:
Generate a single photorealistic, high-resolution architectural photograph showing the updated resilient streetscape."""

    return prompt.strip()



# ---------------------------------------------------------------------------
# FLUX.1 Kontext prompts
# ---------------------------------------------------------------------------
# Kontext follows short, concrete edit instructions far better than long briefs, and drifts
# the architecture unless told explicitly what to keep. Keep these terse.

# Detected protected-object categories (segmentation) -> what to name in the keep-list.
# Naming concrete scene content stops Kontext from replacing the whole street, but the list must
# stay short: a long "keep everything unchanged" clause makes Kontext skip the additions too.
PROTECTED_PHRASES = {
    "vehicle": "vehicles",
    "person": "people",
    "infrastructure": "street lamps",
    "facade_detail": "shop signs",
}
MIN_PROTECTED_PCT = 0.5

# How each kind of zone is named in the edit instruction: (left, right, both, unspecified).
ZONE_PHRASES = {
    "sidewalk": ("the left sidewalk", "the right sidewalk", "both sidewalks", "the sidewalk"),
    "kerb": ("the left kerb", "the right kerb", "both kerbs", "the kerb"),
    "facade": ("the left side of the street", "the right side of the street", "both sides of the street",
               "the street frontage"),
    "roadway": ("the road", "the road", "the road", "the road"),
    "roof": ("visible rooftops", "visible rooftops", "visible rooftops", "visible rooftops"),
}

# Kept out unless the plan explicitly includes a green facade: ivy-covered buildings are not buildable.
KONTEXT_FACADE_KEEP = "Keep the building facades free of ivy and climbing plants."


def _where(zone_kind: str, zones: List[str]) -> str:
    left, right, both, unspecified = ZONE_PHRASES.get(zone_kind, ZONE_PHRASES["sidewalk"])
    sides = set(zones)
    if {"left_sidewalk", "right_sidewalk"} <= sides:
        return both
    if "left_sidewalk" in sides:
        return left
    if "right_sidewalk" in sides:
        return right
    return unspecified


def _keep_list(protected_objects: Optional[dict]) -> str:
    items = ["buildings", "shops"]
    for category, pct in sorted((protected_objects or {}).items(), key=lambda kv: -kv[1]):
        if pct >= MIN_PROTECTED_PCT and category in PROTECTED_PHRASES:
            items.append(PROTECTED_PHRASES[category])
    return ", ".join(items)


def illustrated_zones(plan: SpatialDesignPlan) -> dict:
    """Intervention types the image will show (type -> zones), highest priority first.

    Same-type items on both sides count as one; at most RENDERING["max_image_items"] types.
    """
    zones_by_type: dict = {}
    for item in sorted(plan.interventions, key=lambda i: i.priority):
        if get_spec(item.type) is None:
            continue
        if item.type not in zones_by_type and len(zones_by_type) >= RENDERING["max_image_items"]:
            continue
        zones = zones_by_type.setdefault(item.type, [])
        if item.target_zone:
            zones.append(item.target_zone)
    return zones_by_type


def build_kontext_redesign_prompt(plan: SpatialDesignPlan, protected_objects: Optional[dict] = None) -> str:
    """Build a concise FLUX.1 Kontext edit instruction from the spatial design plan.

    Each planned intervention contributes its buildable catalog phrase (dimensions and spacing from
    buildable/config.py), then a short keep-list built from what segmentation detected
    (protected_objects: category -> % of image). Phrasing it as a "redesign" makes Kontext rebuild
    the street as a generic one; an over-long keep-list makes it change almost nothing.
    Only the top RENDERING["max_image_items"] types are drawn (see illustrated_zones).
    Returns "" when the plan has no buildable interventions.
    """
    zones_by_type = illustrated_zones(plan)
    if not zones_by_type:
        return ""

    clauses = []
    for itype, zones in zones_by_type.items():
        spec = get_spec(itype)
        clauses.append(spec.kontext_phrase(_where(spec.zone_kind, zones)))

    edits = "; ".join(clauses)
    edits = edits[0].upper() + edits[1:]
    facade_keep = "" if "green_facade" in zones_by_type else f" {KONTEXT_FACADE_KEEP}"
    return (
        f"{edits}.{facade_keep} "
        f"Keep the same {_keep_list(protected_objects)}, and the same camera angle and perspective. "
        "Photorealistic photograph."
    )


def build_kontext_refinement_prompt(instruction: str) -> str:
    """Wrap a free-form user refinement so Kontext changes only what was asked."""
    instruction = instruction.strip().rstrip(".")
    return (
        f"{instruction}. Keep everything else exactly the same, including the buildings, existing trees "
        "and greenery, paving, camera angle, perspective and lighting. Photorealistic photograph."
    )
