"""Pydantic schemas and data models for Generative AI urban resilience image editing."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


DesignProfile = Literal[
    "balanced",
    "pedestrian_first",
    "maximum_cooling",
    "green_infrastructure",
    "low_cost",
]


class HeatDriverDiagnosis(BaseModel):
    """Structured spatial heat vulnerability driver diagnosis."""
    driver: str = Field(..., description="Heat mechanism identifier (e.g. pedestrian_solar_exposure, excessive_impervious_surface)")
    severity: float = Field(..., ge=0.0, le=1.0, description="Severity score in [0.0, 1.0]")
    spatial_zones: List[str] = Field(default_factory=list, description="Spatial zones where this driver is active")
    evidence: List[str] = Field(default_factory=list, description="Concrete spatial/environmental evidence signals")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Diagnostic confidence score")


class MaterialPreservationDecision(BaseModel):
    """Decision regarding the preservation of distinctive heritage or architectural surface materials."""
    material: str = Field(..., description="Identified material (e.g. historic_cobblestone, stone_setts, decorative_pavers)")
    preserve: bool = Field(default=True, description="Whether the material must be strictly preserved without synthetic coatings")
    reason: str = Field(..., description="Architectural or cultural heritage justification")


class ExpectedImpactMetrics(BaseModel):
    """Quantitative before-vs-after design evaluation metrics."""
    baseline_heat_priority: float = Field(default=0.75, description="Pre-intervention mean heat priority score in [0.0, 1.0]")
    projected_heat_priority: float = Field(default=0.45, description="Post-intervention projected mean heat priority score")
    delta_heat_priority: float = Field(default=-0.30, description="Projected reduction in heat priority score")
    baseline_pedestrian_shade_pct: float = Field(default=15.0, description="Pre-intervention pedestrian sidewalk shade percentage")
    projected_pedestrian_shade_pct: float = Field(default=65.0, description="Post-intervention pedestrian sidewalk shade percentage")
    delta_pedestrian_shade_pct: float = Field(default=50.0, description="Projected gain in pedestrian shade coverage")
    baseline_impervious_pct: float = Field(default=85.0, description="Pre-intervention impervious ground coverage")
    projected_impervious_pct: float = Field(default=60.0, description="Post-intervention impervious ground coverage")
    delta_impervious_pct: float = Field(default=-25.0, description="Projected reduction in impervious ground coverage")
    baseline_vegetation_pct: float = Field(default=5.0, description="Pre-intervention vegetative cover percentage")
    projected_vegetation_pct: float = Field(default=28.0, description="Post-intervention vegetative cover percentage")
    delta_vegetation_pct: float = Field(default=23.0, description="Projected gain in vegetative cover")
    estimated_cooling_c: float = Field(default=1.4, description="Deterministic empirical localized cooling estimate in °C")
    ml_land_cover_delta_c: float = Field(default=-0.8, description="Landsat LST ML model predicted temperature shift in °C")
    confidence: float = Field(default=0.88, description="Confidence in projected thermal resilience impact")
    summary: str = Field(default="", description="Narrative summary of expected spatial microclimate improvement")


class SpatialInterventionSpec(BaseModel):
    """Rich spatial design specification for a single urban cooling intervention."""
    type: str = Field(..., description="Intervention type identifier (e.g. tree_canopy, cool_pavement)")
    target_region: str = Field(..., description="Physical surface or zone (e.g. sidewalk, road, roof)")
    target_zone: Optional[str] = Field(default=None, description="Specific spatial zone (e.g. left_sidewalk, right_sidewalk, roadway)")
    priority: int = Field(default=1, description="Execution priority ranking (1 is highest)")
    coverage: float = Field(default=0.5, ge=0.05, le=1.0, description="Fraction of eligible surface to transform")
    placement: str = Field(..., description="Specific physical placement within the scene geometry")
    visual_design: str = Field(..., description="Visual styling, material characteristics, colors, and textures")
    reason: str = Field(..., description="Why this intervention is selected for this specific site")
    evidence: List[str] = Field(default_factory=list, description="Spatial and environmental evidence justifying intervention")
    utility_score: Optional[float] = Field(default=None, description="Decision ranking utility score")
    feasibility: float = Field(default=0.9, ge=0.0, le=1.0, description="Physical and urban feasibility score")
    spatial_constraints: List[str] = Field(default_factory=list, description="Intervention-specific spatial constraints")
    title: Optional[str] = Field(default=None, description="Human readable intervention title")
    cooling_impact_c: float = Field(default=0.8, description="Estimated localized cooling in degrees Celsius")
    confidence: float = Field(default=0.85, description="Confidence in intervention placement and benefit")

    # Autonomous Design Intelligence attributes (What / Where / Why / Heat Mechanism / Interactions)
    what: Optional[str] = Field(default=None, description="Precise architectural object and material specification")
    where: Optional[str] = Field(default=None, description="Exact spatial corridor, zone, and ground anchor placement")
    why: Optional[str] = Field(default=None, description="Specific microclimate problem this intervention solves")
    heat_mechanism: Optional[str] = Field(default=None, description="Physics mechanism: shading, albedo, evapotranspiration, or infiltration")
    marginal_utility: Optional[float] = Field(default=None, description="Marginal utility gained by adding this intervention to plan")
    interaction_with_others: Optional[Dict[str, Any]] = Field(default=None, description="Synergies, conflicts, and redundancy penalties")
    expected_benefit: Optional[Dict[str, Any]] = Field(default=None, description="Intervention-level projected microclimate benefits")

    # Explicit Geometric Layout Objects
    anchors: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Explicit grounded anchor points with x, y, relative_depth, scale, canopy_radius",
    )
    spacing: Optional[float] = Field(default=None, description="Perspective-adjusted inter-object spacing in pixels")
    corridor_polyline: Optional[List[List[float]]] = Field(default=None, description="Sidewalk or planting corridor polyline [[x,y], ...]")
    planting_points: Optional[List[Dict[str, Any]]] = Field(default=None, description="Explicit planting points with depth, scale, canopy, and pit specs")
    ground_anchor: Optional[List[float]] = Field(default=None, description="Primary ground contact anchor [x, y]")
    canopy_extent: Optional[List[int]] = Field(default=None, description="Bounding extent [x1, y1, x2, y2] for tree canopy")
    tree_height: Optional[int] = Field(default=None, description="Perspective-scaled tree height in pixels")
    trunk_width: Optional[int] = Field(default=None, description="Perspective-scaled trunk width in pixels")
    planting_pit: Optional[Dict[str, Any]] = Field(default=None, description="Porous sidewalk basin or pit dimensions")
    intended_shade_coverage: Optional[float] = Field(default=None, description="Estimated local pedestrian shade coverage percentage")
    relative_depth: Optional[float] = Field(default=None, description="Monocular relative depth (0.0=near, 1.0=far)")
    scale: Optional[float] = Field(default=None, description="Perspective scaling factor [0.25, 1.0]")
    anchor_points: Optional[List[List[float]]] = Field(default=None, description="Ground support anchor points for structures")
    footprint_polygon: Optional[List[List[float]]] = Field(default=None, description="Canopy or structure ground projection polygon")
    support_points: Optional[List[List[float]]] = Field(default=None, description="Ground column/support contact points")
    orientation: Optional[str] = Field(default=None, description="Structural orientation relative to street geometry")
    canopy_height: Optional[int] = Field(default=None, description="Elevation height of tensile canopy in pixels")
    ground_contact: Optional[List[List[float]]] = Field(default=None, description="Ground column contact points on sidewalk")
    intended_shadow_region: Optional[List[List[float]]] = Field(default=None, description="Projected ground shadow polygon")
    surface_quad: Optional[List[List[float]]] = Field(default=None, description="Perspective quadrilateral for surface textures [[x,y], ...]")
    surface_polygon: Optional[List[List[float]]] = Field(default=None, description="Explicit polygon boundary for surface interventions")
    roof_polygon: Optional[List[List[float]]] = Field(default=None, description="Explicit polygon boundary for rooftop interventions")
    extent: Optional[List[int]] = Field(default=None, description="Bounding extent of surface intervention [x1, y1, x2, y2]")
    material_type: Optional[str] = Field(default=None, description="Surface material texture specification")
    treatment_type: Optional[str] = Field(default=None, description="Surface treatment or coating specification")
    geometry_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Auxiliary geometric layout attributes")


class SpatialDesignPlan(BaseModel):
    """Complete urban resilience architectural design specification."""
    planner_source: str = Field(default="autonomous_spatial_planner", description="Source of design plan")
    site_summary: str = Field(..., description="Brief assessment of current site configuration and microclimate")
    heat_drivers: List[str] = Field(default_factory=list, description="Primary urban heat island causes identified in photo (legacy string format)")
    constraints: List[str] = Field(default_factory=list, description="Physical, structural, and traffic constraints to respect")
    design_profile: DesignProfile = Field(default="balanced", description="Selected urban design strategy")
    interventions: List[SpatialInterventionSpec] = Field(default_factory=list, description="Planned interventions")
    overall_design_intent: str = Field(..., description="Cohesive urban redesign narrative for the generative model")
    scene_understanding: Optional[Dict[str, Any]] = Field(default=None, description="Deep spatial scene understanding summary")

    # Autonomous Design Intelligence Upgrade
    site_diagnosis: Optional[Dict[str, Any]] = Field(default=None, description="Formal site vulnerability diagnosis")
    dominant_heat_drivers: List[HeatDriverDiagnosis] = Field(default_factory=list, description="Structured HeatDriverDiagnosis objects")
    priority_zones: List[Dict[str, Any]] = Field(default_factory=list, description="Ranked spatial priority zones")
    selected_strategy: Optional[str] = Field(default=None, description="Chosen urban design strategy package name")
    rejected_candidates: List[Dict[str, Any]] = Field(default_factory=list, description="Candidate interventions rejected with spatial/thermal reasons")
    expected_impact: Optional[ExpectedImpactMetrics] = Field(default=None, description="Before/after quantitative design evaluation")
    material_preservation: Optional[MaterialPreservationDecision] = Field(default=None, description="Heritage material preservation decision")
    confidence: float = Field(default=0.90, description="Overall planner confidence")
    spatial_rationale: Optional[str] = Field(default=None, description="Spatial reasoning for intervention placements")
    baseline_state: Optional[Dict[str, Any]] = Field(default=None, description="Baseline microclimate state")
    post_intervention_state: Optional[Dict[str, Any]] = Field(default=None, description="Projected post-intervention microclimate state")


class SceneAnalysis(BaseModel):
    """Perception layer output from SegFormer and computer vision heuristics."""
    width: int
    height: int
    aspect_ratio: str
    surface_percentages: Dict[str, float]
    has_visible_roof: bool
    has_pedestrian_sidewalk: bool
    has_road_corridor: bool
    existing_vegetation_pct: float
    detected_constraints: List[str]


class VisualizationOutput(BaseModel):
    """Outcome of generative image editing."""
    status: Literal["ready", "unavailable", "cached"]
    image_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    provider: str = "gemini"
    model: str = "gemini-3.1-flash-image"
    quality_tier: Literal["fast", "final"] = "fast"
    generation_time_ms: int = 0
    refinement_count: int = 0
    error_message: Optional[str] = None


class ValidationReport(BaseModel):
    """Quality and identity preservation assessment of the generated image."""
    is_valid: bool
    aspect_ratio_preserved: bool
    dimensions_valid: bool
    non_blank_verified: bool
    diff_mean: Optional[float] = None
    pct_changed: Optional[float] = None
    perceptual_score: Optional[float] = None
    masked_diff: Optional[float] = None
    unmasked_diff: Optional[float] = None
    architecture_preserved: Optional[bool] = None
    interventions_detected: Optional[List[str]] = None
    checks_passed: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class RefinementIntent(BaseModel):
    """Structured design intent extracted from natural-language instructions."""
    goal: str = Field(..., description="High-level design intent of the refinement")
    add: List[str] = Field(default_factory=list, description="Visual/architectural elements to add")
    remove: List[str] = Field(default_factory=list, description="Elements to remove or reduce")
    modify: List[str] = Field(default_factory=list, description="Elements to alter (scale, material, tone)")
    preserve: List[str] = Field(default_factory=list, description="Elements and architectural geometry to strictly preserve")
    spatial_constraints: List[str] = Field(default_factory=list, description="Physical placement and feasibility rules")


class RefinementResponse(BaseModel):
    """API response model for POST /api/v1/refine-design."""
    status: Literal["ready", "unavailable", "cached"]
    image_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    instruction: str
    provider: str = "gemini"
    model: str = "gemini-3.1-flash-image"
    quality_tier: Literal["fast", "final"] = "fast"
    generation_time_ms: int = 0
    refinement_count: int = 0
    intent: Optional[RefinementIntent] = None
    error_message: Optional[str] = None


class UnifiedRedesignResponse(BaseModel):
    """Comprehensive API response for POST /api/v1/analyze-and-redesign."""
    scene_analysis: SceneAnalysis
    design_plan: SpatialDesignPlan
    visualization: VisualizationOutput
    thermal_impact: Dict[str, Any]
    validation: ValidationReport
