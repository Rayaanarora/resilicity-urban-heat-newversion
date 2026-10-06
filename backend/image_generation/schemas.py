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


class SpatialInterventionSpec(BaseModel):
    """Rich spatial design specification for a single urban cooling intervention."""
    type: str = Field(..., description="Intervention type identifier (e.g. tree_canopy, cool_pavement)")
    target_region: str = Field(..., description="Physical surface or zone (e.g. sidewalk, road, roof)")
    priority: int = Field(default=1, description="Execution priority ranking (1 is highest)")
    coverage: float = Field(default=0.5, ge=0.05, le=1.0, description="Fraction of eligible surface to transform")
    placement: str = Field(..., description="Specific physical placement within the scene geometry")
    visual_design: str = Field(..., description="Visual styling, material characteristics, colors, and textures")
    reason: str = Field(..., description="Why this intervention is selected for this specific site")
    feasibility: float = Field(default=0.9, ge=0.0, le=1.0, description="Physical and urban feasibility score")
    title: Optional[str] = Field(default=None, description="Human readable intervention title")
    cooling_impact_c: float = Field(default=0.8, description="Estimated localized cooling in degrees Celsius")
    confidence: float = Field(default=0.85, description="Confidence in intervention placement and benefit")


class SpatialDesignPlan(BaseModel):
    """Complete urban resilience architectural design specification."""
    site_summary: str = Field(..., description="Brief assessment of current site configuration and microclimate")
    heat_drivers: List[str] = Field(default_factory=list, description="Primary urban heat island causes identified in photo")
    constraints: List[str] = Field(default_factory=list, description="Physical, structural, and traffic constraints to respect")
    design_profile: DesignProfile = Field(default="balanced", description="Selected urban design strategy")
    interventions: List[SpatialInterventionSpec] = Field(default_factory=list, description="Planned interventions")
    overall_design_intent: str = Field(..., description="Cohesive urban redesign narrative for the generative model")


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
