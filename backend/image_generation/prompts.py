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


def build_refinement_prompt(base_prompt: str, user_instruction: str) -> str:
    """Build a refinement prompt modifying an existing generated redesign."""
    return f"""Refine this previously generated urban resilience redesign according to the user's specific adjustment:

USER INSTRUCTION: "{user_instruction}"

GUIDELINES:
- Apply the requested refinement precisely while maintaining all other cooling interventions.
- Preserve the underlying architectural identity, perspective, and photographic realism.
- Ensure all added elements continue to have realistic textures, lighting, and ground contact shadows.
- Output a single photorealistic refined photograph."""
