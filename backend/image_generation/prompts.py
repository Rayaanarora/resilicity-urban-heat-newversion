"""Spatial prompt synthesis for Generative AI urban redesign."""

from typing import List, Optional
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
        "Lush mature urban street trees (such as London plane, neem, or oak) planted along the pedestrian sidewalk "
        "or road curbside margins. Each tree must have a natural textured wooden trunk firmly grounded with a mulch "
        "basin, realistic spreading branches, vibrant green leafy canopy, and soft believable cast shadows falling "
        "across the pavement. Do not place trees inside buildings, through vehicles, or in active traffic lanes."
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
    "unrealistic jungle, sci-fi fantasy buildings"
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
        base_desc = INTERVENTION_PROMPT_DESCRIPTIONS.get(item.type, item.visual_design)
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

ZONE_PHRASES = {
    "left_sidewalk": "the left sidewalk",
    "right_sidewalk": "the right sidewalk",
    "roadway": "the road",
    "rooftop": "visible rooftops",
}

KONTEXT_INTERVENTION_PHRASES = {
    "tree_canopy": (
        "add a continuous row of large mature street trees with wide leafy green canopies along {zones}, "
        "planted in tree pits at the curb and casting shade across the pavement"
    ),
    "cool_pavement": "resurface {zones} with light-gray cool reflective paving while keeping lane markings and curbs",
    "permeable_pave": "repave {zones} with light-colored permeable stone pavers",
    "cool_roof": "coat {zones} with bright white cool roofing",
    "green_roof": "cover {zones} with green vegetated roofs",
    "shade_structure": "add light fabric shade canopies on slender poles over {zones}",
}

DEFAULT_ZONES = {
    "tree_canopy": "both sidewalks",
    "cool_pavement": "the road",
    "permeable_pave": "the sidewalks",
    "cool_roof": "visible rooftops",
    "green_roof": "visible rooftops",
    "shade_structure": "the sidewalk",
}

# Architectural cooling applied to every redesign: shading and greening the facades
# without replacing the buildings themselves.
KONTEXT_ARCHITECTURE = (
    "Add green climbing plants on parts of the sun-facing walls and fabric awnings over ground-floor doors."
)


def _join_zones(zones: List[str]) -> str:
    phrases = list(dict.fromkeys(ZONE_PHRASES.get(z, z.replace("_", " ")) for z in zones))
    if {"the left sidewalk", "the right sidewalk"} <= set(phrases):
        phrases = ["both sidewalks"] + [p for p in phrases if p not in ("the left sidewalk", "the right sidewalk")]
    if len(phrases) <= 1:
        return phrases[0] if phrases else ""
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def _keep_list(protected_objects: Optional[dict]) -> str:
    items = ["buildings", "shops"]
    for category, pct in sorted((protected_objects or {}).items(), key=lambda kv: -kv[1]):
        if pct >= MIN_PROTECTED_PCT and category in PROTECTED_PHRASES:
            items.append(PROTECTED_PHRASES[category])
    return ", ".join(items)


def build_kontext_redesign_prompt(plan: SpatialDesignPlan, protected_objects: Optional[dict] = None) -> str:
    """Build a concise FLUX.1 Kontext edit instruction from the spatial design plan.

    The edit leads with concrete, emphatic additions, then a short keep-list built from what
    segmentation detected (protected_objects: category -> % of image). Phrasing it as a
    "redesign" makes Kontext rebuild the street as a generic one; an over-long keep-list makes
    it change almost nothing.
    """
    zones_by_type: dict = {}
    for item in sorted(plan.interventions, key=lambda i: i.priority):
        if item.type not in KONTEXT_INTERVENTION_PHRASES:
            continue
        zones = zones_by_type.setdefault(item.type, [])
        if item.target_zone:
            zones.append(item.target_zone)

    if not zones_by_type:
        zones_by_type = {"tree_canopy": [], "cool_pavement": []}

    clauses = []
    for itype, zones in zones_by_type.items():
        where = _join_zones(zones) or DEFAULT_ZONES[itype]
        clauses.append(KONTEXT_INTERVENTION_PHRASES[itype].format(zones=where))

    edits = "; ".join(clauses)
    edits = edits[0].upper() + edits[1:]
    return (
        f"{edits}. {KONTEXT_ARCHITECTURE} "
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
