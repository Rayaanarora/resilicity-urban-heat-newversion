"""Typed access to the buildable intervention catalog defined in config.CATALOG."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import CATALOG, CONFIG_LABEL

PHASE_LABELS = {
    "quick_win": "Quick wins (0-3 months)",
    "medium": "Medium term (3-12 months)",
    "long_term": "Long term (12+ months)",
}


@dataclass(frozen=True)
class BuildableSpec:
    type: str
    title: str
    zone_kind: str
    dimensions: Dict[str, Any]
    spacing_m: Optional[float]
    min_clear_width_m: float
    unit: str
    cost_inr: Tuple[int, int]
    lifespan_years: int
    maintenance: str
    permits: List[str]
    phase: str
    cooling_c: Tuple[float, float, float]
    kontext: str
    source: str = field(default=CONFIG_LABEL)

    @property
    def phase_label(self) -> str:
        return PHASE_LABELS[self.phase]

    @property
    def cooling_central_c(self) -> float:
        return self.cooling_c[1]

    def kontext_phrase(self, zones: str) -> str:
        return self.kontext.format(zones=zones, spacing_m=self.spacing_m or 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "title": self.title,
            "zone_kind": self.zone_kind,
            "dimensions": self.dimensions,
            "spacing_m": self.spacing_m,
            "min_clear_width_m": self.min_clear_width_m,
            "unit": self.unit,
            "cost_inr": list(self.cost_inr),
            "lifespan_years": self.lifespan_years,
            "maintenance": self.maintenance,
            "permits": self.permits,
            "phase": self.phase,
            "phase_label": self.phase_label,
            "cooling_c": list(self.cooling_c),
            "source": self.source,
        }


_SPECS: Dict[str, BuildableSpec] = {t: BuildableSpec(type=t, **cfg) for t, cfg in CATALOG.items()}


def get_spec(itype: str) -> Optional[BuildableSpec]:
    return _SPECS.get(itype)


def catalog_types() -> List[str]:
    return list(_SPECS)
