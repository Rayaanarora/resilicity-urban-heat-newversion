"""Per-type cooling estimates shared by the planner, thermal impact and implementation plan."""

from typing import Dict, Optional, Tuple

from .catalog import get_spec
from .config import IMPACT, THERMAL


def cooling_band(itype: str, coverage: float) -> Optional[Tuple[float, float, float]]:
    """(low, central, high) indicative localized cooling in deg C, or None for an unknown type.

    Tree-like items ramp with coverage (base + slope * coverage); others scale linearly.
    """
    spec = get_spec(itype)
    if spec is None:
        return None
    cov = max(0.0, min(1.0, float(coverage)))
    if IMPACT.get(itype, {}).get("ramp"):
        base, slope = THERMAL["tree_coverage_ramp"]
        factor = base + slope * cov
    else:
        factor = cov
    lo, mid, hi = spec.cooling_c
    return round(lo * factor, 2), round(mid * factor, 2), round(hi * factor, 2)


def land_cover_shift(itype: str, coverage: float) -> Dict[str, float]:
    """Land-cover fraction change visible to the Landsat regression for this intervention."""
    out: Dict[str, float] = {}
    for frac, (per_cov, cap) in THERMAL["land_cover_shift"].get(itype, {}).items():
        s = round(min(float(coverage) * per_cov, cap), 3)
        out["f_built"] = out.get("f_built", 0.0) - s
        out[frac] = out.get(frac, 0.0) + s
    return out
