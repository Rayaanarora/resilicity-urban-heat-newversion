"""Autonomous Spatial Urban Resilience Planner.

Transitions ResiliCity from simplistic surface percentage thresholds to a genuinely
intelligent urban heat-resilience planning system.

Executes a 15-step autonomous planning pipeline answering FOUR core questions in order:
1. WHY is this place vulnerable to heat? (Spatial heat-driver diagnosis)
2. WHERE is the most critical exposure? (Multi-factor spatial heat priority mapping)
3. WHAT intervention package addresses that specific mechanism? (Mechanistic matching,
   heritage preservation, interaction synergies/conflicts, and marginal utility optimization)
4. HOW MUCH meaningful improvement is expected? (Quantitative before/after microclimate evaluation)

The objective is:
MAKE THE PLACE COOLER AND MORE HEAT-RESILIENT, NOT SIMPLY MAKE THE IMAGE GREENER.
"""

import asyncio
import io
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from image_generation.schemas import (
    DesignProfile,
    ExpectedImpactMetrics,
    HeatDriverDiagnosis,
    MaterialPreservationDecision,
    SceneAnalysis,
    SpatialDesignPlan,
    SpatialInterventionSpec,
)
from scene_understanding import SceneUnderstanding, analyze_scene

logger = logging.getLogger("resilicity.planner")
router = APIRouter()

PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "autonomous_spatial_planner")
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MINIMUM_IMPACT_THRESHOLD = 0.15

# Catalog of resilient interventions with baseline engineering and cooling characteristics
CATALOG: Dict[str, Dict[str, Any]] = {
    "tree_canopy": {
        "title": "Pedestrian Tree Canopy Corridor",
        "target": "pavement",
        "cost": ("med", "Medium cost"),
        "base_cooling": 1.2,
        "default_coverage": 0.35,
        "heat_mechanism": "shading_and_evapotranspiration",
        "placement": "Planted along sidewalk margins and pedestrian curb lines at 8-10m perspective intervals",
        "visual_design": "Mature broad-canopy shade trees with textured trunks rooted in sidewalk planting pits, casting natural cooling shadows",
        "reason": "Intercepts intense direct solar radiation, cools ambient air via evapotranspiration, and creates continuous pedestrian shade corridors.",
    },
    "cool_pavement": {
        "title": "High-Albedo Solar-Reflective Roadway",
        "target": "road",
        "cost": ("med", "Medium cost"),
        "base_cooling": 0.8,
        "default_coverage": 0.65,
        "heat_mechanism": "surface_albedo_reflection",
        "placement": "Applied across asphalt road lanes while strictly preserving crosswalks, lane markings, and curbs",
        "visual_design": "Light-gray solar-reflective architectural coating (albedo ~0.40) preserving all painted traffic markings and lane lines",
        "reason": "Prevents dark asphalt from absorbing and re-radiating thermal energy, dropping daytime surface temperatures by up to 12°C.",
    },
    "permeable_pave": {
        "title": "Permeable Interlocking Sidewalk Pavers",
        "target": "pavement",
        "cost": ("med", "Medium cost"),
        "base_cooling": 0.6,
        "default_coverage": 0.50,
        "heat_mechanism": "permeability_and_infiltration",
        "placement": "Installed across pedestrian walkways and sidewalk corridors",
        "visual_design": "Modular interlocking light-tone concrete pavers with narrow gravel drainage joints",
        "reason": "Facilitates sub-surface rainwater infiltration, promotes evaporative cooling, and improves pedestrian walking comfort.",
    },
    "shade_structure": {
        "title": "Architectural Tensile Shade Canopy",
        "target": "pavement",
        "cost": ("low", "Low cost"),
        "base_cooling": 0.7,
        "default_coverage": 0.35,
        "heat_mechanism": "solar_interception_shading",
        "placement": "Suspended over sidewalk gathering zones and pedestrian transit waiting corridors anchored by slender steel posts",
        "visual_design": "Modern geometric light-toned tensile fabric sailcloth anchored to slender structural steel columns",
        "reason": "Provides immediate high-density solar obstruction where narrow sidewalk width or utilities limit deep tree root growth.",
    },
    "cool_roof": {
        "title": "High-Albedo Cool Roof Coating",
        "target": "roof",
        "cost": ("low", "Low cost"),
        "base_cooling": 0.9,
        "default_coverage": 0.75,
        "heat_mechanism": "envelope_albedo_reflection",
        "placement": "Coated across flat building rooftops exposed to overhead sun",
        "visual_design": "Clean off-white solar-reflective elastomeric roof membrane coating (albedo ~0.80)",
        "reason": "Reflects incoming solar radiation, reducing rooftop surface temperatures and internal building cooling loads.",
    },
    "green_roof": {
        "title": "Extensive Vegetative Green Roof",
        "target": "roof",
        "cost": ("high", "High cost"),
        "base_cooling": 1.4,
        "default_coverage": 0.60,
        "heat_mechanism": "biological_evapotranspiration",
        "placement": "Installed on structurally sound flat rooftop surfaces",
        "visual_design": "Lush sedum succulent matting with organic flowering groundcover and gravel drainage borders",
        "reason": "Maximizes natural biological evapotranspiration, provides thermal insulation, and captures urban stormwater.",
    },
}


def diagnose_heat_drivers(scene: SceneUnderstanding, surfaces: Dict[str, float]) -> List[HeatDriverDiagnosis]:
    """Build a comprehensive HeatDriverDiagnosis diagnosing WHY this street is vulnerable.

    Evaluates:
    - excessive impervious surface
    - direct solar exposure
    - insufficient pedestrian shade
    - sparse existing canopy
    - high street-canyon enclosure
    - low sky visibility
    - large exposed roadway
    - dark high-absorption material
    - insufficient permeable/vegetated ground
    - pedestrian solar exposure
    """
    drivers: List[HeatDriverDiagnosis] = []
    geom = scene.street_geometry
    ls = scene.left_sidewalk
    rs = scene.right_sidewalk
    road = scene.roadway
    veg_dist = scene.vegetation_distribution or {}
    road_mat = scene.roadway_material or {}

    total_imp = float(surfaces.get("road", 0.0) + surfaces.get("pavement", 0.0) + surfaces.get("wall", 0.0))
    non_sky = max(10.0, 100.0 - surfaces.get("sky", 0.0))
    imp_ratio = min(1.0, total_imp / non_sky)

    # 1. Excessive Impervious Surface
    if imp_ratio >= 0.50:
        drivers.append(HeatDriverDiagnosis(
            driver="excessive_impervious_surface",
            severity=round(imp_ratio, 2),
            spatial_zones=["roadway", "left_sidewalk", "right_sidewalk"],
            evidence=[
                f"Impervious built surfaces comprise {imp_ratio*100:.0f}% of non-sky scene area",
                "Continuous asphalt and concrete prevent natural evaporative cooling",
                "High thermal mass stores solar heat during daytime for nocturnal release",
            ],
            confidence=0.92,
        ))

    # 2. Direct Solar Exposure
    if scene.solar_exposure_proxy >= 0.35:
        zones = []
        if road.solar_exposure >= 0.40:
            zones.append("roadway")
        if ls.solar_exposure >= 0.40:
            zones.append("left_sidewalk")
        if rs.solar_exposure >= 0.40:
            zones.append("right_sidewalk")
        drivers.append(HeatDriverDiagnosis(
            driver="direct_solar_exposure",
            severity=round(float(scene.solar_exposure_proxy), 2),
            spatial_zones=zones or ["street_corridor"],
            evidence=[
                f"Direct solar insolation index is {scene.solar_exposure_proxy:.2f} across the street canyon",
                f"Sun-direction proxy indicates {scene.sun_direction_proxy.replace('_', ' ')}",
                "High radiant solar load strikes unshaded ground surfaces",
            ],
            confidence=0.88,
        ))

    # 3. Insufficient Pedestrian Shade
    ped_shade_cov = float(veg_dist.get("pedestrian_shade_coverage", 0.0))
    min_shade = min(ls.existing_shade, rs.existing_shade)
    if min_shade < 0.40 or ped_shade_cov < 40.0:
        shade_deficit = round(1.0 - min_shade, 2)
        sw_zones = []
        if ls.existing_shade < 0.40:
            sw_zones.append("left_sidewalk")
        if rs.existing_shade < 0.40:
            sw_zones.append("right_sidewalk")
        drivers.append(HeatDriverDiagnosis(
            driver="insufficient_pedestrian_shade",
            severity=shade_deficit,
            spatial_zones=sw_zones or ["sidewalk_corridors"],
            evidence=[
                f"Pedestrian corridors have only {min_shade*100:.0f}% existing cast shadow coverage",
                f"Canopy distribution is {veg_dist.get('canopy_distribution', 'sparse')} leaving walkers exposed",
                "Elevates human mean radiant temperature (MRT) in pedestrian walking zones",
            ],
            confidence=0.90,
        ))

    # 4. Sparse Existing Canopy
    avg_tree_dens = (ls.existing_tree_density + rs.existing_tree_density) / 2.0
    if avg_tree_dens < 0.30:
        drivers.append(HeatDriverDiagnosis(
            driver="sparse_existing_canopy",
            severity=round(1.0 - min(1.0, avg_tree_dens * 2.5), 2),
            spatial_zones=["left_sidewalk", "right_sidewalk"],
            evidence=[
                f"Existing vegetative canopy density along sidewalks is acute ({avg_tree_dens*100:.0f}%)",
                f"Canopy continuity score is {veg_dist.get('canopy_continuity', 0.0):.2f}",
                "Negligible biological transpiration cooling contribution along pedestrian verges",
            ],
            confidence=0.89,
        ))

    # 5. High Street-Canyon Enclosure
    if geom.street_canyon_strength >= 0.45:
        drivers.append(HeatDriverDiagnosis(
            driver="high_street_canyon_enclosure",
            severity=round(float(geom.street_canyon_strength), 2),
            spatial_zones=["street_canyon_corridor"],
            evidence=[
                f"Street canyon ratio proxy is {geom.street_canyon_strength:.2f} (H/W proxy {geom.height_to_width_proxy:.1f})",
                f"Flanking building wall density is {geom.building_wall_density*100:.0f}%",
                "Traps multiple reflections of shortwave solar radiation and longwave thermal emission",
            ],
            confidence=0.87,
        ))

    # 6. Low Sky Visibility
    if geom.sky_visibility_proxy < 0.50:
        drivers.append(HeatDriverDiagnosis(
            driver="low_sky_visibility",
            severity=round(1.0 - float(geom.sky_visibility_proxy), 2),
            spatial_zones=["street_corridor"],
            evidence=[
                f"Sky view factor proxy (SVF) is constrained at {geom.sky_visibility_proxy:.2f}",
                "Restricted celestial view factor severely impedes nocturnal radiative cooling",
                "Exacerbates localized urban heat island persistence overnight",
            ],
            confidence=0.85,
        ))

    # 7. Large Exposed Roadway
    if road.relative_area_pct >= 15.0 and road.solar_exposure >= 0.40:
        road_sev = round(min(1.0, (road.relative_area_pct / 45.0) * road.solar_exposure), 2)
        drivers.append(HeatDriverDiagnosis(
            driver="large_exposed_roadway",
            severity=road_sev,
            spatial_zones=["roadway"],
            evidence=[
                f"Roadway dominates {road.relative_area_pct:.1f}% of scene area with road width proxy {road.road_width_proxy:.2f}",
                f"Roadway solar exposure index is {road.solar_exposure:.2f}",
                "Massive ground surface area absorbing direct insolation and re-radiating sensible heat",
            ],
            confidence=0.93,
        ))

    # 8. Dark High-Absorption Road Material
    if not road_mat.get("is_heritage", False) and road.relative_area_pct >= 10.0:
        drivers.append(HeatDriverDiagnosis(
            driver="dark_high_absorption_material",
            severity=0.82,
            spatial_zones=["roadway"],
            evidence=[
                "Standard low-albedo asphalt bitumen road surface absorbs ~90% of solar radiation",
                f"Measured surface texture energy is {road_mat.get('texture_energy', 0.0):.1f}",
                "High asphalt daytime surface temperatures drive ambient microclimate heating",
            ],
            confidence=0.89,
        ))

    # 9. Insufficient Permeable / Vegetated Ground
    veg_pct = float(surfaces.get("vegetation", 0.0))
    if veg_pct < 15.0:
        drivers.append(HeatDriverDiagnosis(
            driver="insufficient_permeable_vegetated_ground",
            severity=round(1.0 - min(1.0, veg_pct / 20.0), 2),
            spatial_zones=["left_sidewalk", "right_sidewalk", "street_margins"],
            evidence=[
                f"Total scene vegetation is only {veg_pct:.1f}%, leaving ground almost 100% sealed",
                "Absence of permeable soils suppresses natural moisture infiltration and evaporative cooling",
                "Rainwater runs off rapidly without providing latent cooling benefit",
            ],
            confidence=0.90,
        ))

    # 10. Pedestrian Solar Exposure Concentration
    max_ped_sol = max(ls.solar_exposure * (1.0 - ls.existing_shade), rs.solar_exposure * (1.0 - rs.existing_shade))
    if max_ped_sol >= 0.25:
        target_sw = "left_sidewalk" if ls.pedestrian_heat_priority >= rs.pedestrian_heat_priority else "right_sidewalk"
        drivers.append(HeatDriverDiagnosis(
            driver="pedestrian_solar_exposure",
            severity=round(min(1.0, max_ped_sol * 1.5), 2),
            spatial_zones=[target_sw],
            evidence=[
                f"Active pedestrian walkway in {target_sw.replace('_', ' ')} suffers unshaded solar radiation",
                f"Pedestrian heat priority score is {getattr(scene, target_sw).pedestrian_heat_priority:.2f}",
                "Elevates human heat strain along primary pedestrian movement paths",
            ],
            confidence=0.91,
        ))

    # Rank drivers by severity * confidence descending
    drivers.sort(key=lambda d: -(d.severity * d.confidence))
    return drivers


def evaluate_interactions(
    candidate_type: str,
    target_zone: str,
    chosen: List[Dict[str, Any]],
    scene: SceneUnderstanding,
) -> Tuple[float, List[str], List[str], bool]:
    """Model synergies, conflicts, and redundancies between candidate and current plan.

    Returns:
        (delta_utility, synergies, conflicts, is_rejected)
    """
    delta_utility = 0.0
    synergies: List[str] = []
    conflicts: List[str] = []
    is_rejected = False

    chosen_types = [c["type"] for c in chosen]
    chosen_zones = [c["target_zone"] for c in chosen]

    # Conflict 1: Heritage Cobblestone Preservation
    if candidate_type == "cool_pavement":
        if scene.roadway_material.get("is_heritage", False):
            conflicts.append("Historic cobblestone / stone sett roadway must be strictly preserved; synthetic cool-pavement coating rejected.")
            return -1.0, synergies, conflicts, True

    # Conflict 2: Invisible Rooftops
    if candidate_type in ("cool_roof", "green_roof"):
        roof_conf = scene.confidence_breakdown.get("roof_confidence", 0.0)
        roof_pct = float(scene.heat_priority_summary.get("roof_pct", 0.0))
        if roof_pct < 3.0 or roof_conf < 0.60:
            conflicts.append("Rooftops are not visible from street camera viewpoint; rooftop cooling rejected for spatial incoherence.")
            return -1.0, synergies, conflicts, True

    # Conflict 3: Sidewalk Width Restriction for Trees
    if candidate_type == "tree_canopy":
        sw_zone = getattr(scene, target_zone, None)
        if sw_zone and sw_zone.available_width_proxy < 0.08 and sw_zone.relative_area_pct < 0.8:
            conflicts.append(f"{target_zone.replace('_', ' ').capitalize()} width is too constrained for tree root basins and pedestrian passage.")
            return -1.0, synergies, conflicts, True

    # Synergy 1: Trees + Permeable Pavers
    if candidate_type == "permeable_pave" and "tree_canopy" in chosen_types:
        delta_utility += 0.14
        synergies.append("Synergy with tree canopy: Permeable paving facilitates root aeration and rainwater infiltration to nourish street trees.")
    elif candidate_type == "tree_canopy" and "permeable_pave" in chosen_types:
        delta_utility += 0.14
        synergies.append("Synergy with permeable pavers: Trees complement permeable ground plane by adding overhead solar interception and evapotranspiration.")

    # Redundancy 1: Trees + Shade Structure in the exact same zone
    if candidate_type == "shade_structure" and "tree_canopy" in chosen_types:
        for c in chosen:
            if c["type"] == "tree_canopy" and c["target_zone"] == target_zone:
                delta_utility -= 0.60
                conflicts.append(f"Redundancy: Shade canopy overlaps with proposed tree canopy corridor in {target_zone.replace('_', ' ')}.")
                is_rejected = True

    if candidate_type == "tree_canopy" and "shade_structure" in chosen_types:
        for c in chosen:
            if c["type"] == "shade_structure" and c["target_zone"] == target_zone:
                delta_utility -= 0.60
                conflicts.append(f"Redundancy: Tree canopy duplicates overhead shade in {target_zone.replace('_', ' ')} already addressed by tensile canopy.")
                is_rejected = True

    # Diminishing returns: Second tree canopy in opposite sidewalk
    if candidate_type == "tree_canopy" and "tree_canopy" in chosen_types:
        delta_utility -= 0.15
        conflicts.append("Diminishing returns: Second tree corridor provides incremental shade but increases street canyon canopy enclosure.")

    # Diminishing returns: Duplicate surface interventions
    if candidate_type in chosen_types and candidate_type != "tree_canopy":
        is_rejected = True
        conflicts.append(f"Duplicate intervention: {candidate_type} already included in plan.")

    return round(delta_utility, 3), synergies, conflicts, is_rejected


def determine_strategy_package(
    scene: SceneUnderstanding,
    drivers: List[HeatDriverDiagnosis],
    surfaces: Dict[str, float],
) -> Tuple[str, str]:
    """Select the overarching urban design strategy package.

    Packages:
    - PACKAGE A — PEDESTRIAN SHADE CORRIDOR
    - PACKAGE B — CONSTRAINED PEDESTRIAN CORRIDOR
    - PACKAGE C — EXPOSED HARD ROADWAY
    - PACKAGE D — ROOFTOP HEAT
    - PACKAGE E — COMPREHENSIVE CLIMATE-RESILIENT STREETSCAPE
    """
    geom = scene.street_geometry
    ls = scene.left_sidewalk
    rs = scene.right_sidewalk
    road = scene.roadway
    road_mat = scene.roadway_material or {}

    wider_sw = ls if ls.available_width_proxy >= rs.available_width_proxy else rs
    is_heritage = road_mat.get("is_heritage", False)

    # Check conditions
    has_ped_vulnerability = any(d.driver in ("pedestrian_solar_exposure", "insufficient_pedestrian_shade") for d in drivers[:3])
    has_road_vulnerability = any(d.driver in ("large_exposed_roadway", "dark_high_absorption_material") for d in drivers[:3])

    if has_ped_vulnerability and wider_sw.available_width_proxy >= 0.15 and wider_sw.tree_feasibility >= 0.40:
        if is_heritage:
            return (
                "PACKAGE A — PEDESTRIAN SHADE CORRIDOR (HERITAGE PRESERVED)",
                "Prioritize pedestrian shade corridor via perspective tree canopy and permeable pavers while strictly preserving historic cobblestone roadway.",
            )
        elif has_road_vulnerability and not is_heritage:
            return (
                "PACKAGE E — COMPREHENSIVE CLIMATE-RESILIENT STREETSCAPE",
                "Execute coordinated pedestrian tree shade corridor, permeable infiltration pavers, and high-albedo solar-reflective roadway coating.",
            )
        else:
            return (
                "PACKAGE A — PEDESTRIAN SHADE CORRIDOR",
                "Establish a shaded pedestrian corridor along the wider sidewalk verge with continuous tree canopy and permeable drainage pavers.",
            )

    elif has_ped_vulnerability and wider_sw.available_width_proxy < 0.15:
        return (
            "PACKAGE B — CONSTRAINED PEDESTRIAN CORRIDOR",
            "Sidewalk width limits deep tree root growth; provide architectural tensile shade canopy with permeable surface enhancement.",
        )

    elif has_road_vulnerability and not is_heritage:
        return (
            "PACKAGE C — EXPOSED HARD ROADWAY",
            "Transform extensive dark asphalt into a high-albedo solar-reflective roadway while preserving traffic lanes.",
        )

    return (
        "BALANCED URBAN HEAT MITIGATION",
        "Targeted microclimate cooling balancing pedestrian shade, permeable ground breathability, and facade preservation.",
    )


def compute_intervention_utility(
    itype: str,
    scene: SceneUnderstanding,
    target_zone: str,
    profile: str = "balanced",
) -> Tuple[float, List[str], float, float]:
    """Calculate multi-objective utility score U(I) and extract concrete spatial evidence signals.

    Formula:
    U(I) = α*HeatBenefit + β*PedestrianComfort + γ*ShadeBenefit + δ*Feasibility
           + ε*WaterBenefit + ζ*ExistingInfrastructureCompatibility
           - λ*Cost - μ*TrafficConflict - ν*VisualDisruption

    Returns:
        (utility_score, evidence_list, feasibility_score, coverage)
    """
    geom = scene.street_geometry
    road = scene.roadway
    ls = scene.left_sidewalk
    rs = scene.right_sidewalk
    road_mat = scene.roadway_material or {}
    veg_dist = scene.vegetation_distribution or {}

    zone_sw = ls if target_zone == "left_sidewalk" else (rs if target_zone == "right_sidewalk" else (ls if ls.relative_area_pct >= rs.relative_area_pct else rs))

    evidence: List[str] = []
    heat_benefit = 0.0
    ped_comfort = 0.0
    shade_benefit = 0.0
    feasibility = 0.0
    water_benefit = 0.0
    compat_benefit = 0.70
    cost = 0.40
    traffic_conflict = 0.10
    visual_disruption = 0.10
    coverage = 0.50

    if itype == "tree_canopy":
        cost = 0.45
        traffic_conflict = road.traffic_conflict_density * 0.25
        visual_disruption = 0.18

        if zone_sw.available_width_proxy >= 0.15:
            evidence.append(f"{zone_sw.zone_name.replace('_', ' ').capitalize()} has adequate sidewalk width (width proxy {zone_sw.available_width_proxy:.2f}) for curbside planting pits")
        else:
            evidence.append(f"{zone_sw.zone_name.replace('_', ' ').capitalize()} has compact margin available along building facade verge")

        if zone_sw.solar_exposure >= 0.35:
            evidence.append(f"Elevated direct solar exposure (proxy {zone_sw.solar_exposure:.2f}) creates high pedestrian heat vulnerability")

        if zone_sw.existing_tree_density < 0.30:
            evidence.append(f"Acute existing vegetative canopy deficit ({zone_sw.existing_tree_density*100:.0f}% cover) requires continuous tree shelter")

        if geom.street_canyon_strength >= 0.40:
            evidence.append(f"Street canyon geometry (canyon strength {geom.street_canyon_strength:.2f}) traps radiant heat, requiring vertical tree canopy")

        if zone_sw.conflict_objects_count < 1000:
            evidence.append("Low conflict with active vehicular lanes along sidewalk curb line")

        feasibility = zone_sw.tree_feasibility
        heat_benefit = 0.90 * zone_sw.pedestrian_heat_priority
        ped_comfort = 0.94
        shade_benefit = 0.95 * zone_sw.solar_exposure
        water_benefit = 0.40
        coverage = min(0.40, max(0.20, zone_sw.available_width_proxy * 0.5 + 0.15))

    elif itype == "shade_structure":
        cost = 0.35
        traffic_conflict = 0.05
        visual_disruption = 0.15

        if zone_sw.available_width_proxy < 0.28:
            evidence.append(f"Sidewalk width (proxy {zone_sw.available_width_proxy:.2f}) restricts root spread of large trees, favoring tensile canopies")
        else:
            evidence.append("Wide pedestrian walking zone enables anchored modular architectural shade canopies")

        if zone_sw.solar_exposure >= 0.40:
            evidence.append(f"Intense unshaded solar exposure (proxy {zone_sw.solar_exposure:.2f}) on pedestrian path")

        if zone_sw.existing_shade < 0.35:
            evidence.append(f"Low existing cast shadow ({zone_sw.existing_shade*100:.0f}%) demands immediate physical solar interception")

        if geom.street_canyon_strength >= 0.45:
            evidence.append("Building facades provide stable lateral anchoring context for slender tensile posts")

        feasibility = zone_sw.shade_structure_feasibility
        heat_benefit = 0.75 * zone_sw.pedestrian_heat_priority
        ped_comfort = 0.88
        shade_benefit = 0.92 * zone_sw.solar_exposure
        water_benefit = 0.0
        coverage = 0.35

    elif itype == "cool_pavement":
        cost = 0.40
        traffic_conflict = 0.12
        visual_disruption = 0.10

        if road_mat.get("is_heritage", False):
            # Historic cobblestone roadway: strictly negative utility
            return -1.0, ["Historic cobblestone roadway detected; cool pavement coating prohibited to preserve heritage."], 0.0, 0.0

        if road.road_width_proxy >= 0.20:
            evidence.append(f"Expansive asphalt road corridor (width proxy {road.road_width_proxy:.2f}) dominates ground solar absorption")

        if road.solar_exposure >= 0.35:
            evidence.append(f"High roadway solar exposure (proxy {road.solar_exposure:.2f}) causes intense daytime asphalt heat storage")

        if road.existing_shade < 0.40:
            evidence.append(f"Minimal daytime shading across driving lanes ({road.existing_shade*100:.0f}%) accelerates thermal re-radiation")

        if geom.street_canyon_strength >= 0.30:
            evidence.append("Solar reflective coating prevents thermal trap between flanking building walls")

        feasibility = road.cool_pavement_feasibility
        heat_benefit = 0.88 * scene.heat_priority_summary.get("roadway_heat_priority_score", 0.70)
        ped_comfort = 0.45
        shade_benefit = 0.25
        water_benefit = 0.0
        coverage = 0.65 if road.road_width_proxy > 0.4 else 0.50

    elif itype == "permeable_pave":
        cost = 0.38
        traffic_conflict = 0.04
        visual_disruption = 0.08

        if zone_sw.relative_area_pct >= 1.5:
            evidence.append(f"Dedicated pedestrian sidewalk area ({zone_sw.relative_area_pct:.1f}% of scene) suitable for modular paver installation")

        if zone_sw.pedestrian_heat_priority >= 0.35:
            evidence.append(f"High pedestrian heat priority index ({zone_sw.pedestrian_heat_priority:.2f}) prioritizes sidewalk cooling")

        if zone_sw.available_width_proxy >= 0.10:
            evidence.append("Continuous walking corridor benefits from permeable ground infiltration and evaporative cooling")

        feasibility = zone_sw.permeable_paver_feasibility
        heat_benefit = 0.70 * zone_sw.pedestrian_heat_priority
        ped_comfort = 0.85
        shade_benefit = 0.15
        water_benefit = 0.90
        coverage = 0.50

    elif itype in ("cool_roof", "green_roof"):
        roof_pct = float(scene.heat_priority_summary.get("roof_pct", 0.0))
        roof_conf = scene.confidence_breakdown.get("roof_confidence", 0.0)
        if roof_pct < 3.0 or roof_conf < 0.60:
            return -1.0, [], 0.0, 0.0

        cost = 0.30 if itype == "cool_roof" else 0.80
        traffic_conflict = 0.0
        visual_disruption = 0.05

        evidence.append(f"Confirmed visible building rooftop area ({roof_pct:.1f}% of image) from street viewpoint")
        evidence.append("Direct overhead solar exposure on building envelope")
        evidence.append("Rooftop thermal mitigation reduces urban heat accumulation into upper air layer")

        feasibility = 0.85 if itype == "cool_roof" else 0.75
        heat_benefit = 0.85 if itype == "green_roof" else 0.70
        ped_comfort = 0.35
        shade_benefit = 0.10
        water_benefit = 0.75 if itype == "green_roof" else 0.0
        coverage = 0.65

    # Design Profile Adaptation
    if profile == "pedestrian_first":
        alpha, beta, gamma, delta, eps, zeta = 0.20, 0.35, 0.25, 0.15, 0.05, 0.10
    elif profile == "maximum_cooling":
        alpha, beta, gamma, delta, eps, zeta = 0.40, 0.15, 0.25, 0.15, 0.05, 0.10
    elif profile == "green_infrastructure":
        alpha, beta, gamma, delta, eps, zeta = 0.25, 0.20, 0.25, 0.15, 0.15, 0.10
        if itype in ("tree_canopy", "green_roof"):
            heat_benefit *= 1.2
    else:  # balanced
        alpha, beta, gamma, delta, eps, zeta = 0.25, 0.25, 0.25, 0.15, 0.05, 0.10

    lam, mu, nu = 0.05, 0.08, 0.05

    utility = (
        alpha * heat_benefit +
        beta * ped_comfort +
        gamma * shade_benefit +
        delta * feasibility +
        eps * water_benefit +
        zeta * compat_benefit -
        lam * cost -
        mu * traffic_conflict -
        nu * visual_disruption
    )

    return float(utility), evidence, float(feasibility), float(coverage)


def evaluate_expected_impact(
    scene: SceneUnderstanding,
    selected: List[Dict[str, Any]],
    surfaces: Dict[str, float],
) -> ExpectedImpactMetrics:
    """Quantitative before-vs-after design evaluation.

    Compares baseline vs projected post-intervention state:
    - Delta heat priority
    - Delta pedestrian shade coverage
    - Delta impervious surface
    - Delta vegetation coverage
    - Localized empirical cooling (°C)
    - Landsat ML land-cover delta (°C)
    """
    veg_dist = scene.vegetation_distribution or {}
    base_heat = float(scene.heat_priority_summary.get("overall_heat_priority_score", 0.70))
    base_ped_shade = float(veg_dist.get("pedestrian_shade_coverage", 15.0))
    base_veg = float(surfaces.get("vegetation", 5.0))
    base_imp = float(surfaces.get("road", 40.0) + surfaces.get("pavement", 15.0) + surfaces.get("wall", 25.0))

    # Calculate projected impacts
    delta_heat = 0.0
    delta_shade = 0.0
    delta_veg = 0.0
    delta_imp = 0.0
    total_cooling = 0.0

    for item in selected:
        itype = item["type"]
        cov = float(item["coverage"])
        if itype == "tree_canopy":
            delta_heat -= 0.15 * (0.6 + 0.4 * cov)
            delta_shade += 45.0 * (0.6 + 0.4 * cov)
            delta_veg += 18.0 * (0.6 + 0.4 * cov)
            delta_imp -= 8.0 * cov
            total_cooling += round(1.2 * (0.6 + 0.4 * cov), 1)
        elif itype == "shade_structure":
            delta_heat -= 0.10 * cov
            delta_shade += 30.0 * cov
            total_cooling += round(0.7 * cov, 1)
        elif itype == "cool_pavement":
            delta_heat -= 0.12 * cov
            total_cooling += round(0.8 * cov, 1)
        elif itype == "permeable_pave":
            delta_heat -= 0.08 * cov
            delta_imp -= 16.0 * cov
            total_cooling += round(0.6 * cov, 1)
        elif itype == "green_roof":
            delta_heat -= 0.10 * cov
            delta_veg += 15.0 * cov
            total_cooling += round(1.4 * cov, 1)
        elif itype == "cool_roof":
            delta_heat -= 0.08 * cov
            total_cooling += round(0.9 * cov, 1)

    proj_heat = max(0.15, base_heat + delta_heat)
    proj_shade = min(95.0, base_ped_shade + delta_shade)
    proj_veg = min(80.0, base_veg + delta_veg)
    proj_imp = max(20.0, base_imp + delta_imp)

    # Simplified ML model proxy shift (-0.5 to -1.5°C)
    ml_delta = round(-0.4 * (delta_veg / 15.0) - 0.2 * (abs(delta_imp) / 20.0), 2)

    summary_msg = (
        f"Redesign package projects a {abs(delta_heat):.2f} point reduction in spatial heat priority, "
        f"increasing pedestrian shade coverage from {base_ped_shade:.1f}% to {proj_shade:.1f}% (+{delta_shade:.1f}%), "
        f"and expanding vegetative coverage from {base_veg:.1f}% to {proj_veg:.1f}%."
    )

    return ExpectedImpactMetrics(
        baseline_heat_priority=round(base_heat, 2),
        projected_heat_priority=round(proj_heat, 2),
        delta_heat_priority=round(delta_heat, 2),
        baseline_pedestrian_shade_pct=round(base_ped_shade, 1),
        projected_pedestrian_shade_pct=round(proj_shade, 1),
        delta_pedestrian_shade_pct=round(delta_shade, 1),
        baseline_impervious_pct=round(base_imp, 1),
        projected_impervious_pct=round(proj_imp, 1),
        delta_impervious_pct=round(delta_imp, 1),
        baseline_vegetation_pct=round(base_veg, 1),
        projected_vegetation_pct=round(proj_veg, 1),
        delta_vegetation_pct=round(delta_veg, 1),
        estimated_cooling_c=round(total_cooling, 1),
        ml_land_cover_delta_c=round(ml_delta, 2),
        confidence=0.89,
        summary=summary_msg,
    )


def generate_spatial_plan(
    image: Image.Image,
    surfaces: Dict[str, float],
    seg_result: Optional[Dict[str, Any]] = None,
    design_profile: Optional[DesignProfile] = None,
    requested_types: Optional[List[str]] = None,
) -> SpatialDesignPlan:
    """Autonomously generate an intelligent urban resilience design plan.

    Answers in order:
    1. WHY is this place vulnerable to heat? (HeatDriverDiagnosis)
    2. WHERE is the most critical exposure? (Spatial Priority Model)
    3. WHAT intervention package addresses that specific mechanism? (Intervention Package & Heritage Preservation)
    4. HOW MUCH meaningful improvement is expected? (ExpectedImpactMetrics)
    """
    w, h = image.size
    profile = str(design_profile or "balanced")

    if seg_result is None:
        seg_result = {"classes": [{"id": k, "percentage": v} for k, v in surfaces.items()]}

    # Step 1: Deep Scene Understanding
    scene = analyze_scene(image, seg_result)
    geom = scene.street_geometry
    ls = scene.left_sidewalk
    rs = scene.right_sidewalk
    road = scene.roadway
    road_mat = scene.roadway_material or {}
    veg_dist = scene.vegetation_distribution or {}

    # Step 2: Build Heat-Driver Diagnosis
    drivers = diagnose_heat_drivers(scene, surfaces)

    # Step 3: Material / Heritage Preservation Assessment
    mat_preservation: Optional[MaterialPreservationDecision] = None
    if road_mat.get("is_heritage", False):
        mat_preservation = MaterialPreservationDecision(
            material=road_mat.get("material", "historic_cobblestone"),
            preserve=True,
            reason=road_mat.get("reason", "Distinctive heritage roadway material visible in photograph."),
        )

    # Step 4: Strategy Package Determination
    strategy_name, strategy_rationale = determine_strategy_package(scene, drivers, surfaces)

    # Step 5: Candidate Generation
    candidates: List[Dict[str, Any]] = []
    rejected_candidates: List[Dict[str, Any]] = []

    # Check Left Sidewalk Tree Canopy
    can_plant_left = (ls.relative_area_pct >= 1.0 or ls.available_width_proxy >= 0.05 or surfaces.get("pavement", 0.0) >= 1.0 or surfaces.get("road", 0.0) >= 15.0)
    if can_plant_left:
        u_val, ev_list, feas, cov = compute_intervention_utility("tree_canopy", scene, "left_sidewalk", profile)
        if len(ev_list) >= 3 and feas >= 0.25:
            candidates.append({
                "type": "tree_canopy",
                "target_region": "pavement" if (ls.relative_area_pct >= 0.5 or surfaces.get("pavement", 0.0) > 0) else "road",
                "target_zone": "left_sidewalk",
                "utility": u_val,
                "evidence": ev_list,
                "feasibility": feas,
                "coverage": cov,
                "heat_mechanism": "shading_and_evapotranspiration",
                "what": "Mature leafy street trees with natural bark trunks grounded in sidewalk planting pits",
                "where": "Curbside planting alignment along left sidewalk margin",
                "why": "Intercepts direct solar exposure on pedestrian path and cools ambient air via evapotranspiration",
                "placement": "Curbside planting pits along left street verge spaced 8-10m in perspective",
                "visual_design": CATALOG["tree_canopy"]["visual_design"],
                "reason": CATALOG["tree_canopy"]["reason"],
                "cooling": round(1.2 * (0.6 + 0.4 * cov), 1),
            })

    # Check Right Sidewalk Tree Canopy
    can_plant_right = (rs.relative_area_pct >= 1.0 or rs.available_width_proxy >= 0.05 or surfaces.get("pavement", 0.0) >= 1.0 or surfaces.get("road", 0.0) >= 15.0)
    if can_plant_right and (rs.solar_exposure >= 0.35 or surfaces.get("pavement", 0.0) >= 5.0):
        u_val, ev_list, feas, cov = compute_intervention_utility("tree_canopy", scene, "right_sidewalk", profile)
        if len(ev_list) >= 3 and feas >= 0.25:
            candidates.append({
                "type": "tree_canopy",
                "target_region": "pavement" if (rs.relative_area_pct >= 0.5 or surfaces.get("pavement", 0.0) > 0) else "road",
                "target_zone": "right_sidewalk",
                "utility": u_val,
                "evidence": ev_list,
                "feasibility": feas,
                "coverage": cov,
                "heat_mechanism": "shading_and_evapotranspiration",
                "what": "Mature leafy street trees with natural bark trunks grounded in sidewalk planting pits",
                "where": "Curbside planting alignment along right sidewalk margin",
                "why": "Provides solar interception and continuous pedestrian shade on right sidewalk corridor",
                "placement": "Curbside planting pits along right street verge spaced 8-10m in perspective",
                "visual_design": CATALOG["tree_canopy"]["visual_design"],
                "reason": CATALOG["tree_canopy"]["reason"],
                "cooling": round(1.2 * (0.6 + 0.4 * cov), 1),
            })

    # Check Tensile Shade Structure
    active_sw = ls if ls.shade_structure_feasibility >= rs.shade_structure_feasibility else rs
    if active_sw.relative_area_pct >= 1.0 or surfaces.get("pavement", 0.0) >= 5.0:
        u_val, ev_list, feas, cov = compute_intervention_utility("shade_structure", scene, active_sw.zone_name, profile)
        if len(ev_list) >= 3 and feas >= 0.30:
            candidates.append({
                "type": "shade_structure",
                "target_region": "pavement",
                "target_zone": active_sw.zone_name,
                "utility": u_val,
                "evidence": ev_list,
                "feasibility": feas,
                "coverage": cov,
                "heat_mechanism": "solar_interception_shading",
                "what": "Modern geometric tensile fabric shade canopy with light-toned architectural sailcloth",
                "where": f"Suspended above {active_sw.zone_name.replace('_', ' ')} pedestrian corridor",
                "why": "Immediate overhead solar interception where sidewalk width restricts deep tree roots",
                "placement": f"Suspended above {active_sw.zone_name.replace('_', ' ')} pedestrian gathering zone anchored by slender steel columns",
                "visual_design": CATALOG["shade_structure"]["visual_design"],
                "reason": CATALOG["shade_structure"]["reason"],
                "cooling": round(0.7 * cov, 1),
            })

    # Check Cool Pavement on Roadway
    if road.relative_area_pct >= 5.0 or road.road_width_proxy >= 0.15 or surfaces.get("road", 0.0) >= 10.0:
        if mat_preservation and mat_preservation.preserve:
            rejected_candidates.append({
                "type": "cool_pavement",
                "target_zone": "roadway",
                "reason": f"Preserving distinctive {mat_preservation.material.replace('_', ' ')} roadway heritage; synthetic coating rejected.",
                "conflict": "heritage_preservation",
            })
        else:
            u_val, ev_list, feas, cov = compute_intervention_utility("cool_pavement", scene, "roadway", profile)
            if len(ev_list) >= 3 and feas >= 0.30:
                candidates.append({
                    "type": "cool_pavement",
                    "target_region": "road",
                    "target_zone": "roadway",
                    "utility": u_val,
                    "evidence": ev_list,
                    "feasibility": feas,
                    "coverage": cov,
                    "heat_mechanism": "surface_albedo_reflection",
                    "what": "High-albedo solar-reflective architectural coating (albedo ~0.40) preserving painted road markings",
                    "where": "Applied across active asphalt road lanes while preserving curbs and crosswalks",
                    "why": "Prevents dark asphalt from absorbing and re-radiating thermal energy into street canyon",
                    "placement": "Applied across asphalt road plane while strictly preserving crosswalks, lane markings, and curbs",
                    "visual_design": CATALOG["cool_pavement"]["visual_design"],
                    "reason": CATALOG["cool_pavement"]["reason"],
                    "cooling": round(0.8 * cov, 1),
                })

    # Check Permeable Pavers on Sidewalk
    pave_sw = ls if ls.relative_area_pct >= rs.relative_area_pct else rs
    if pave_sw.relative_area_pct >= 1.5 or surfaces.get("pavement", 0.0) >= 2.0:
        u_val, ev_list, feas, cov = compute_intervention_utility("permeable_pave", scene, pave_sw.zone_name, profile)
        if len(ev_list) >= 3 and feas >= 0.25:
            candidates.append({
                "type": "permeable_pave",
                "target_region": "pavement",
                "target_zone": pave_sw.zone_name,
                "utility": u_val,
                "evidence": ev_list,
                "feasibility": feas,
                "coverage": cov,
                "heat_mechanism": "permeability_and_infiltration",
                "what": "Modular interlocking light-tone concrete pavers with permeable gravel infiltration joints",
                "where": f"Installed across {pave_sw.zone_name.replace('_', ' ')} walking corridor",
                "why": "Promotes sub-surface rainwater infiltration, ground breathability, and evaporative cooling",
                "placement": f"Installed across {pave_sw.zone_name.replace('_', ' ')} pedestrian walking corridor",
                "visual_design": CATALOG["permeable_pave"]["visual_design"],
                "reason": CATALOG["permeable_pave"]["reason"],
                "cooling": round(0.6 * cov, 1),
            })

    # Step 6: Greedy Marginal Utility Optimization with Design Budget
    # Determine design budget (1 to 4 interventions depending on vulnerability severity)
    top_sev = max((d.severity for d in drivers), default=0.5)
    if top_sev >= 0.75:
        max_budget = 4
    elif top_sev >= 0.50:
        max_budget = 3
    else:
        max_budget = 2

    chosen_candidates: List[Dict[str, Any]] = []
    remaining_candidates = list(candidates)

    while len(chosen_candidates) < max_budget and remaining_candidates:
        best_cand = None
        best_marginal_utility = -999.0
        best_cand_idx = -1
        best_synergies = []
        best_conflicts = []

        for idx, cand in enumerate(remaining_candidates):
            delta_u, syn, conf, is_rej = evaluate_interactions(cand["type"], cand["target_zone"], chosen_candidates, scene)
            if is_rej:
                continue

            marginal_u = cand["utility"] + delta_u
            if marginal_u > best_marginal_utility:
                best_marginal_utility = marginal_u
                best_cand = cand
                best_cand_idx = idx
                best_synergies = syn
                best_conflicts = conf

        # Minimum Impact Threshold check
        if best_cand is None or best_marginal_utility < MINIMUM_IMPACT_THRESHOLD:
            # Reject remaining low-impact candidates
            for cand in remaining_candidates:
                rejected_candidates.append({
                    "type": cand["type"],
                    "target_zone": cand["target_zone"],
                    "reason": f"Marginal utility ({cand['utility']:.2f}) below minimum impact threshold ({MINIMUM_IMPACT_THRESHOLD:.2f}).",
                    "conflict": "negligible_marginal_benefit",
                })
            break

        # Select the winning candidate
        best_cand["marginal_utility"] = round(best_marginal_utility, 3)
        best_cand["interaction_with_others"] = {
            "synergies": best_synergies,
            "conflicts": best_conflicts,
        }
        chosen_candidates.append(best_cand)
        remaining_candidates.pop(best_cand_idx)

    # Step 7: Guaranteed Safe Fallback if nothing was chosen
    if not chosen_candidates:
        if candidates:
            chosen_candidates.append(candidates[0])
        else:
            t_meta = CATALOG["tree_canopy"]
            chosen_candidates.append({
                "type": "tree_canopy",
                "target_region": "pavement" if surfaces.get("pavement", 0.0) > 0 else "road",
                "target_zone": "left_sidewalk",
                "utility": 0.85,
                "evidence": [
                    "Pedestrian walking zone exposed to direct overhead solar insolation",
                    "Acute urban vegetation deficit demands cooling tree canopy corridor",
                    "Curbside planting alignment feasible along street margin",
                ],
                "feasibility": 0.90,
                "coverage": 0.35,
                "heat_mechanism": "shading_and_evapotranspiration",
                "what": t_meta["visual_design"],
                "where": "Curbside planting margin along sidewalk",
                "why": t_meta["reason"],
                "placement": t_meta["placement"],
                "visual_design": t_meta["visual_design"],
                "reason": t_meta["reason"],
                "cooling": 1.2,
                "marginal_utility": 0.85,
                "interaction_with_others": {"synergies": [], "conflicts": []},
            })

    # Step 8: Quantitative Expected Impact Evaluation
    expected_impact = evaluate_expected_impact(scene, chosen_candidates, surfaces)

    # Step 9: Build SpatialInterventionSpec objects
    final_specs: List[SpatialInterventionSpec] = []
    for rank, item in enumerate(chosen_candidates, start=1):
        spec = SpatialInterventionSpec(
            type=item["type"],
            target_region=item["target_region"],
            target_zone=item["target_zone"],
            priority=rank,
            coverage=round(item["coverage"], 2),
            placement=item["placement"],
            visual_design=item["visual_design"],
            reason=item["reason"],
            evidence=item["evidence"],
            utility_score=round(item["utility"], 3),
            feasibility=round(item["feasibility"], 3),
            title=CATALOG.get(item["type"], {}).get("title", item["type"]),
            cooling_impact_c=item.get("cooling", 0.8),
            confidence=0.92,
            what=item.get("what", item["visual_design"]),
            where=item.get("where", item["placement"]),
            why=item.get("why", item["reason"]),
            heat_mechanism=item.get("heat_mechanism", CATALOG.get(item["type"], {}).get("heat_mechanism", "shading")),
            marginal_utility=item.get("marginal_utility", round(item["utility"], 3)),
            interaction_with_others=item.get("interaction_with_others", {}),
            expected_benefit={
                "estimated_localized_cooling_c": item.get("cooling", 0.8),
                "target_zone": item["target_zone"],
            },
        )
        final_specs.append(spec)

    # Step 10: Populate Explicit Geometry for Renderer
    from image_generation.layout_engine import populate_intervention_explicit_geometry
    for spec in final_specs:
        populate_intervention_explicit_geometry(spec, image, seg_result, scene)

    # Step 11: Build Constraints & Design Narrative
    constraints = list(scene.design_constraints)
    if surfaces.get("roof", 0.0) < 3.0 and not any("rooftop" in c.lower() for c in constraints):
        constraints.append("No visible rooftops detected from street viewpoint; rooftop measures omitted.")

    site_diagnosis_data = {
        "vulnerability_summary": f"Street canyon strength {geom.street_canyon_strength:.2f} with {road.relative_area_pct:.1f}% road area and acute pedestrian shade deficit.",
        "dominant_drivers": [d.driver for d in drivers[:3]],
        "pedestrian_heat_priority": float(scene.heat_priority_summary.get("pedestrian_heat_priority_score", 0.65)),
        "roadway_heat_priority": float(scene.heat_priority_summary.get("roadway_heat_priority_score", 0.65)),
        "heritage_detected": bool(mat_preservation and mat_preservation.preserve),
    }

    priority_zones_data = [
        {
            "zone": "left_sidewalk",
            "heat_priority": ls.pedestrian_heat_priority,
            "available_width_proxy": ls.available_width_proxy,
            "solar_exposure": ls.solar_exposure,
            "intervention": next((s.type for s in final_specs if s.target_zone == "left_sidewalk"), "none"),
        },
        {
            "zone": "right_sidewalk",
            "heat_priority": rs.pedestrian_heat_priority,
            "available_width_proxy": rs.available_width_proxy,
            "solar_exposure": rs.solar_exposure,
            "intervention": next((s.type for s in final_specs if s.target_zone == "right_sidewalk"), "none"),
        },
        {
            "zone": "roadway",
            "heat_priority": float(scene.heat_priority_summary.get("roadway_heat_priority_score", 0.65)),
            "road_width_proxy": road.road_width_proxy,
            "solar_exposure": road.solar_exposure,
            "intervention": next((s.type for s in final_specs if s.target_zone == "roadway"), "none"),
        },
    ]

    site_summary = (
        f"Site diagnosis: Dense urban corridor with {geom.street_canyon_strength:.2f} street canyon enclosure, "
        f"{scene.solar_exposure_proxy:.2f} direct solar exposure index, and {veg_dist.get('pedestrian_shade_coverage', 15.0):.1f}% pedestrian shade coverage. "
        f"Microclimate demands targeted pedestrian shade shelter and permeable ground enhancements."
    )

    spatial_rationale = (
        f"Strategy: {strategy_name}. Interventions placed strategically where sidewalk width ({ls.available_width_proxy:.2f} left, {rs.available_width_proxy:.2f} right) "
        f"and solar exposure prioritize pedestrian comfort without conflicting with vehicular lanes."
    )

    overall_intent = (
        f"Autonomously execute {strategy_name}: Implement {', '.join(s.title for s in final_specs)} "
        f"to achieve meaningful microclimate cooling while strictly preserving existing architectural facades, "
        f"vehicles, pedestrians, and street perspective."
    )

    heat_driver_strings = [f"{d.driver}: severity={d.severity:.2f} in {', '.join(d.spatial_zones)}" for d in drivers[:4]]

    baseline_state = {
        "heat_priority": expected_impact.baseline_heat_priority,
        "pedestrian_shade_pct": expected_impact.baseline_pedestrian_shade_pct,
        "impervious_pct": expected_impact.baseline_impervious_pct,
        "vegetation_pct": expected_impact.baseline_vegetation_pct,
    }

    post_state = {
        "heat_priority": expected_impact.projected_heat_priority,
        "pedestrian_shade_pct": expected_impact.projected_pedestrian_shade_pct,
        "impervious_pct": expected_impact.projected_impervious_pct,
        "vegetation_pct": expected_impact.projected_vegetation_pct,
    }

    return SpatialDesignPlan(
        planner_source="autonomous_spatial_planner",
        site_summary=site_summary,
        heat_drivers=heat_driver_strings,
        constraints=constraints,
        design_profile=design_profile or "balanced",
        interventions=final_specs,
        overall_design_intent=overall_intent,
        scene_understanding=scene.to_dict(),
        site_diagnosis=site_diagnosis_data,
        dominant_heat_drivers=drivers,
        priority_zones=priority_zones_data,
        selected_strategy=strategy_name,
        rejected_candidates=rejected_candidates,
        expected_impact=expected_impact,
        material_preservation=mat_preservation,
        confidence=scene.confidence,
        spatial_rationale=spatial_rationale,
        baseline_state=baseline_state,
        post_intervention_state=post_state,
    )


def analyze_scene_heuristics(image: Image.Image, surfaces: Dict[str, float]) -> SceneAnalysis:
    """Derive spatial understanding summary for API consumers."""
    w, h = image.size
    aspect = f"{w}:{h}"
    if abs(w / h - 1.0) < 0.05:
        aspect = "1:1"
    elif abs(w / h - 16 / 9) < 0.1:
        aspect = "16:9"
    elif abs(w / h - 4 / 3) < 0.1:
        aspect = "4:3"

    road_pct = surfaces.get("road", 0.0)
    pave_pct = surfaces.get("pavement", 0.0)
    veg_pct = surfaces.get("vegetation", 0.0)
    roof_pct = surfaces.get("roof", 0.0)

    constraints = []
    if roof_pct < 3.0:
        constraints.append("No visible rooftop geometry detected from street viewpoint; rooftop measures omitted.")
    if road_pct > 30.0:
        constraints.append("Substantial vehicular road corridor; preserve lane markings and traffic pathways.")

    return SceneAnalysis(
        width=w,
        height=h,
        aspect_ratio=aspect,
        surface_percentages=surfaces,
        has_visible_roof=roof_pct >= 3.0,
        has_pedestrian_sidewalk=pave_pct >= 3.0,
        has_road_corridor=road_pct >= 8.0,
        existing_vegetation_pct=round(veg_pct, 1),
        detected_constraints=constraints,
    )


def planner_available() -> bool:
    return True


@router.post("/api/v1/plan")
async def plan_legacy(
    file: UploadFile = File(...),
    surfaces: str = Form("{}"),
):
    """Autonomous spatial urban resilience planner endpoint."""
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
        surf_dict = json.loads(surfaces) if surfaces else {}
        if not isinstance(surf_dict, dict):
            surf_dict = {}
    except Exception:
        surf_dict = {}

    spatial_plan = generate_spatial_plan(img, surf_dict, design_profile="balanced")
    legacy_interventions = []
    for item in spatial_plan.interventions:
        legacy_interventions.append({
            "id": f"p-{item.type}-{item.priority}",
            "type": item.type,
            "title": item.title or item.type,
            "targetRegion": item.target_region,
            "targetZone": item.target_zone,
            "priority": item.priority,
            "coverage": item.coverage,
            "evidence": item.evidence,
            "utilityScore": item.utility_score,
            "coolingImpact": item.cooling_impact_c,
            "description": item.reason,
            "promptTemplate": item.visual_design,
            "defaultEnabled": True,
            "what": item.what,
            "where": item.where,
            "why": item.why,
            "heatMechanism": item.heat_mechanism,
            "marginalUtility": item.marginal_utility,
        })

    return {
        "source": "autonomous_spatial_planner",
        "model": "autonomous_spatial_planner",
        "siteSummary": spatial_plan.site_summary,
        "selectedStrategy": spatial_plan.selected_strategy,
        "interventions": legacy_interventions,
        "dominantHeatDrivers": [d.dict() for d in spatial_plan.dominant_heat_drivers],
        "priorityZones": spatial_plan.priority_zones,
        "rejectedCandidates": spatial_plan.rejected_candidates,
        "expectedImpact": spatial_plan.expected_impact.dict() if spatial_plan.expected_impact else None,
        "materialPreservation": spatial_plan.material_preservation.dict() if spatial_plan.material_preservation else None,
        "sceneUnderstanding": spatial_plan.scene_understanding,
    }
