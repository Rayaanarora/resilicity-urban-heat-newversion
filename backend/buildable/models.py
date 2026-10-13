"""Pydantic models for buildable-design outputs (kept free of image_generation imports)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StreetMetrics(BaseModel):
    """Metric street dimensions estimated from the photo, with an uncertainty band."""
    road_width_m: float = Field(..., description="Carriageway width used as the scale reference")
    scale_source: str = Field(default="default", description="'user' if road_width_m was supplied, else 'default'")
    left_sidewalk_m: Optional[float] = Field(default=None, description="Estimated left sidewalk width (None if not measurable)")
    right_sidewalk_m: Optional[float] = Field(default=None, description="Estimated right sidewalk width (None if not measurable)")
    left_range_m: Optional[List[float]] = Field(default=None, description="[low, high] band for the left sidewalk")
    right_range_m: Optional[List[float]] = Field(default=None, description="[low, high] band for the right sidewalk")
    uncertainty_pct: int = Field(default=30, description="+/- percentage applied to every estimate")
    road_truncated: bool = Field(default=False, description="Carriageway runs off the frame, biasing the scale")
    rows_used: int = Field(default=0, description="Image rows used for the estimate")
    scale_available: bool = Field(default=False, description="A carriageway scale was measured (widths can be judged)")
    reliable: bool = Field(default=False, description="False when the estimate should not be used without a site survey")
    notes: List[str] = Field(default_factory=list)

    def sidewalk_m(self, zone: str) -> Optional[float]:
        return {"left_sidewalk": self.left_sidewalk_m, "right_sidewalk": self.right_sidewalk_m}.get(zone)


class ImplementationLineItem(BaseModel):
    """One bill-of-quantities line for a selected intervention, per street_length_m of street."""
    type: str
    title: str
    target_zone: Optional[str] = None
    quantity: float
    unit: str
    basis: str = Field(default="", description="How the quantity was derived")
    quantity_flagged: bool = Field(default=False, description="Quantity uses a default instead of a measurement")
    cost_inr_low: int
    cost_inr_high: int
    cooling_c: float = Field(default=0.0, description="Indicative localized cooling attributed to this item")
    cooling_per_lakh_c: float = Field(default=0.0, description="Cooling per lakh rupees at mid cost")
    phase: str
    phase_label: str
    lifespan_years: int
    maintenance: str
    permits: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    extra: Dict[str, float] = Field(default_factory=dict, description="Type-specific outputs, e.g. kWp")
    dimensions: Dict[str, Any] = Field(default_factory=dict, description="Catalog dimensions (for drawings)")


class ImplementationPhase(BaseModel):
    phase: str
    label: str
    items: List[str] = Field(default_factory=list)
    cost_inr_low: int = 0
    cost_inr_high: int = 0


class ImplementationPlan(BaseModel):
    """Bill of quantities, indicative costs, phasing and permits per street_length_m of street."""
    street_length_m: float
    line_items: List[ImplementationLineItem] = Field(default_factory=list)
    phases: List[ImplementationPhase] = Field(default_factory=list)
    total_cost_inr_low: int = 0
    total_cost_inr_high: int = 0
    total_cooling_c: float = 0.0
    cooling_per_lakh_c: float = 0.0
    budget_lakh: Optional[float] = None
    within_budget: Optional[bool] = None
    budget_excluded: List[Dict[str, Any]] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    clearances: Dict[str, float] = Field(default_factory=dict, description="Clearances used (for drawings)")
    source: str = ""
