"""Pydantic models for buildable-design outputs (kept free of image_generation imports)."""

from typing import List, Optional

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
