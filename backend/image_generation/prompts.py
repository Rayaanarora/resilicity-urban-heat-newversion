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


SDXL_NEGATIVE_PROMPT = (
    "cartoon, illustration, 3d render, blurry, distorted, low quality, bad architecture, deformed"
)


def build_pass_sdxl_prompt(intervention_type: str, attempt: int = 0) -> str:
    """Build a dedicated, short prompt for localized SD inpainting.
    
    SD 1.5 excels with short, direct prompts describing the localized element itself.
    Strictly kept under 20 tokens to maximize CLIP fidelity.
    """
    if intervention_type == "tree_canopy":
        return "a leafy green street tree, realistic, natural light"

    elif intervention_type == "shade_structure":
        return "modern tensile fabric shade canopy, realistic architectural structure, natural light"

    elif intervention_type == "cool_pavement":
        if attempt == 0:
            return (
                "transform asphalt roadway into realistic light-gray solar-reflective cool pavement surface, "
                "high albedo material, preserving painted lane markings, crosswalks, curbs, vehicles, sharp focus"
            )
        else:
            return (
                "photorealistic street transformation, bright light-gray solar-reflective road coating, "
                "clean smooth high-albedo road surface, intact painted traffic lane lines, authentic streetscape, sharp focus"
            )

    elif intervention_type == "permeable_pave":
        if attempt == 0:
            return (
                "replace pedestrian pavement with realistic light-tone interlocking permeable concrete pavers, "
                "modular paving stones, narrow gravel drainage joints, preserving curbs, buildings, doors, sharp focus"
            )
        else:
            return (
                "architectural street renovation, high-quality light-gray modular permeable interlocking pavers "
                "installed across pedestrian sidewalk, distinct paving units, crisp curb edges, sharp focus"
            )

    elif intervention_type in ("cool_roof", "green_roof"):
        if intervention_type == "green_roof":
            return (
                "extensive vegetative green roof on flat building rooftop, lush sedum succulent plants, "
                "gravel drainage border, preserving rooftop parapets and building geometry, sharp focus"
            )
        else:
            return (
                "solar-reflective bright white elastomeric cool roof coating on flat rooftop, "
                "clean high-albedo membrane, preserving parapets and building architecture, sharp focus"
            )

    return (
        "photorealistic urban streetscape modification, high-albedo resilient urban infrastructure, "
        "natural daylight, preserving architecture and street perspective, sharp focus"
    )


def build_sdxl_inpainting_prompt(plan: SpatialDesignPlan, attempt: int = 0) -> str:
    """Composite SDXL prompt for backward compatibility."""
    if plan.interventions:
        # Use primary intervention prompt
        primary = plan.interventions[0].type
        return build_pass_sdxl_prompt(primary, attempt=attempt)
    return "photorealistic resilient urban streetscape with mature green street trees and light-gray cool pavement"




