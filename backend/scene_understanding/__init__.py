"""Scene understanding module for autonomous spatial urban resilience planning."""

from .depth import estimate_relative_depth, compute_geometric_depth_fallback
from .geometry import analyze_street_corridor, estimate_vanishing_point
from .heat_priority import compute_spatial_heat_priority_map, render_heat_priority_colormap
from .representation import SceneUnderstanding, SidewalkZone, RoadwayZone, StreetGeometry, analyze_scene
from .solar import estimate_solar_and_shade

__all__ = [
    "estimate_relative_depth",
    "compute_geometric_depth_fallback",
    "estimate_vanishing_point",
    "analyze_street_corridor",
    "compute_spatial_heat_priority_map",
    "render_heat_priority_colormap",
    "estimate_solar_and_shade",
    "SceneUnderstanding",
    "SidewalkZone",
    "RoadwayZone",
    "StreetGeometry",
    "analyze_scene",
]
