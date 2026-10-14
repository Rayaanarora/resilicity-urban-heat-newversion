"""Optimiser candidates: gated-accepted items of each project with cost and cooling bands and exposure."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Pick at most one item from each exclusivity group.
STREET_TREES = {"tree_canopy", "parking_lane_tree"}
ROOF = {"cool_roof", "green_roof"}
OVERHEAD_SHADE = {"shade_structure", "solar_canopy", "facade_awning"}

PHASE_RANK = {"quick_win": 0, "medium": 1, "long_term": 2}


@dataclass(frozen=True)
class Candidate:
    key: str
    project_id: str
    project_name: str
    type: str
    title: str
    zone: Optional[str]
    group: str
    cost_inr: Dict[str, float]     # low / central / high
    cooling_c: Dict[str, float]    # low / central / high (literature)
    exposure: float                # people in the project's hotspot cell (constant in demo mode)
    exposure_source: str           # measured | assumed
    phase: str
    phase_label: str

    def value(self, objective: str, cooling_case: str = "central") -> float:
        c = self.cooling_c[cooling_case]
        return c * self.exposure if objective in ("person_c", "min_cost_for_target") else c


def exclusivity_group(project_id: str, itype: str, zone: Optional[str], key: str) -> str:
    if itype in STREET_TREES:
        return f"{project_id}|{zone}|street_trees"
    if itype in ROOF:
        return f"{project_id}|roof"
    if itype in OVERHEAD_SHADE:
        return f"{project_id}|{zone}|overhead_shade"
    return key


def candidates_for(project: Dict[str, Any], phase_limit: Optional[str] = None) -> List[Candidate]:
    """project is a projects.service.project_view() dict."""
    hotspot = project.get("hotspot") or {}
    exposure = float(hotspot.get("population") or 0.0)
    exposure_source = "measured" if project.get("data_mode") == "grid" else "assumed"
    max_rank = PHASE_RANK.get(phase_limit, 2) if phase_limit else 2
    out = []
    for item in project.get("accepted", []):
        if item.get("cooling_c") is None or PHASE_RANK.get(item["phase"], 2) > max_rank:
            continue
        key = f"{project['id']}:{item['type']}:{item.get('target_zone') or 'site'}"
        out.append(Candidate(
            key=key,
            project_id=project["id"],
            project_name=project["name"],
            type=item["type"],
            title=item["title"],
            zone=item.get("target_zone"),
            group=exclusivity_group(project["id"], item["type"], item.get("target_zone"), key),
            cost_inr={k: float(item["cost_inr"][k]) for k in ("low", "central", "high")},
            cooling_c={k: float(item["cooling_c"][k]) for k in ("low", "central", "high")},
            exposure=exposure,
            exposure_source=exposure_source,
            phase=item["phase"],
            phase_label=item["phase_label"],
        ))
    return out
