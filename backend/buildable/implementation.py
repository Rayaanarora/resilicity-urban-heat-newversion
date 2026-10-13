"""Implementation plan: bill of quantities, indicative cost, phasing and budget-capped selection.

Quantities are per CLEARANCES["street_length_m"] of street and per zone (one side for
sidewalk/kerb/facade items). Costs are the catalog's indicative unit ranges.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from .catalog import PHASE_LABELS, get_spec
from .config import BUDGET, CLEARANCES, CONFIG_LABEL, QUANTITY
from .models import ImplementationLineItem, ImplementationPhase, ImplementationPlan, StreetMetrics

INR_PER_LAKH = BUDGET["inr_per_lakh"]


def _sidewalk_width(zone: Optional[str], metrics: StreetMetrics) -> Tuple[float, bool]:
    width = metrics.sidewalk_m(zone or "")
    if width is None:
        return CLEARANCES["default_sidewalk_m_if_unmeasured"], True
    return width, False


def quantity_for(item: Dict[str, Any], metrics: StreetMetrics) -> Tuple[float, str, bool, Dict[str, float]]:
    """Return (quantity, basis, flagged, extra) for a chosen candidate."""
    itype, zone = item["type"], item.get("target_zone")
    spec = get_spec(itype)
    length = CLEARANCES["street_length_m"]
    usable = max(0.0, length - float(item.get("excluded_length_m", 0.0)))
    cov = float(item.get("coverage", 0.5))
    dims = spec.dimensions
    extra: Dict[str, float] = {}

    if itype in ("tree_canopy", "parking_lane_tree", "kerb_bioswale"):
        n = math.floor(usable / spec.spacing_m)
        basis = f"{usable:.0f} m usable frontage / {spec.spacing_m:.0f} m spacing"
        if usable < length:
            basis += f" ({length - usable:.0f} m junction setback excluded)"
        return float(n), basis, False, extra
    if itype == "cool_pavement":
        return (round(metrics.road_width_m * length * cov, 1),
                f"{metrics.road_width_m:.1f} m carriageway x {length:.0f} m x {cov:.0%} coverage", False, extra)
    if itype == "permeable_pave":
        width, flagged = _sidewalk_width(zone, metrics)
        return (round(width * length * cov, 1),
                f"{width:.1f} m footpath{' (default, not measured)' if flagged else ''} x {length:.0f} m x {cov:.0%}",
                flagged, extra)
    if itype in ("shade_structure", "solar_canopy"):
        run = QUANTITY["canopy_length_share"] * length
        area = round(run * dims["depth_m"], 1)
        if itype == "solar_canopy":
            extra["kwp"] = round(area * dims["kwp_per_m2"], 1)
        return area, f"{run:.0f} m run x {dims['depth_m']:.1f} m depth", False, extra
    if itype == "facade_awning":
        run = QUANTITY["awning_frontage_share"] * length
        return round(run, 1), f"{QUANTITY['awning_frontage_share']:.0%} of {length:.0f} m frontage", False, extra
    if itype == "bus_shelter":
        return float(QUANTITY["bus_shelters_per_length"]), "one shelter at an existing bus stop", False, extra
    if itype == "cool_wall":
        area = QUANTITY["cool_wall_frontage_share"] * length * dims["coated_height_m"]
        return (round(area, 1), f"{QUANTITY['cool_wall_frontage_share']:.0%} of {length:.0f} m frontage x "
                f"{dims['coated_height_m']:.0f} m height", False, extra)
    if itype == "green_facade":
        area = QUANTITY["green_facade_frontage_share"] * length * dims["trellis_height_m"]
        return round(area, 1), f"{QUANTITY['green_facade_frontage_share']:.0%} of frontage x trellis height", False, extra
    if itype in ("cool_roof", "green_roof"):
        area = length * dims["assumed_roof_depth_m"] * cov
        return (round(area, 1), f"{length:.0f} m frontage x assumed {dims['assumed_roof_depth_m']:.0f} m roof depth "
                f"x {cov:.0%} (roof area not measurable from the street)", True, extra)
    if itype == "misting_system":
        return round(QUANTITY["misting_length_share"] * length, 1), "share of street length at waiting areas", False, extra
    return 0.0, "no quantity rule", True, extra


def line_item(item: Dict[str, Any], metrics: StreetMetrics) -> ImplementationLineItem:
    spec = get_spec(item["type"])
    qty, basis, flagged, extra = quantity_for(item, metrics)
    lo, hi = (int(round(qty * c)) for c in spec.cost_inr)
    cooling = float(item.get("cooling", spec.cooling_central_c))
    mid_lakh = (lo + hi) / 2.0 / INR_PER_LAKH
    return ImplementationLineItem(
        type=spec.type, title=spec.title, target_zone=item.get("target_zone"),
        quantity=qty, unit=spec.unit, basis=basis, quantity_flagged=flagged,
        cost_inr_low=lo, cost_inr_high=hi, cooling_c=round(cooling, 2),
        cooling_per_lakh_c=round(cooling / mid_lakh, 3) if mid_lakh > 0 else 0.0,
        phase=spec.phase, phase_label=spec.phase_label, lifespan_years=spec.lifespan_years,
        maintenance=spec.maintenance, permits=list(spec.permits), conditions=list(item.get("conditions", [])),
        extra=extra,
        dimensions=dict(spec.dimensions),
    )


def mid_cost_lakh(item: Dict[str, Any], metrics: StreetMetrics) -> float:
    li = line_item(item, metrics)
    return (li.cost_inr_low + li.cost_inr_high) / 2.0 / INR_PER_LAKH


def select_within_budget(
    candidates: List[Dict[str, Any]],
    budget_lakh: float,
    metrics: StreetMetrics,
    interactions: Callable[[Dict[str, Any], List[Dict[str, Any]]], Tuple[float, List[str], List[str], bool]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Greedy cost-effectiveness selection under a budget cap.

    Ranks by cooling per lakh (mid cost), skipping items rejected by the interaction model or that
    would exceed the cap. Returns (chosen, budget_rejections).
    """
    ranked = sorted(
        candidates,
        key=lambda c: -(float(c.get("cooling", 0.0)) / max(mid_cost_lakh(c, metrics), 1e-6)),
    )
    chosen: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    spent = 0.0
    for cand in ranked:
        if len(chosen) >= BUDGET["max_interventions_with_budget"]:
            break
        delta_u, syn, conf, rejected = interactions(cand, chosen)
        if rejected:
            continue
        cost = mid_cost_lakh(cand, metrics)
        if spent + cost > budget_lakh:
            excluded.append({
                "type": cand["type"], "target_zone": cand.get("target_zone"), "rule": "over_budget",
                "conflict": "over_budget",
                "reason": f"Needs about {cost:.1f} lakh; only {budget_lakh - spent:.1f} of the {budget_lakh:.1f} lakh "
                          "budget remains after more cost-effective items.",
                "measured": {"mid_cost_lakh": round(cost, 2), "remaining_lakh": round(budget_lakh - spent, 2)},
            })
            continue
        cand["marginal_utility"] = round(cand.get("utility", 0.0) + delta_u, 3)
        cand["interaction_with_others"] = {"synergies": syn, "conflicts": conf}
        chosen.append(cand)
        spent += cost
    return chosen, excluded


def build_implementation_plan(
    chosen: List[Dict[str, Any]],
    metrics: StreetMetrics,
    budget_lakh: Optional[float] = None,
    budget_excluded: Optional[List[Dict[str, Any]]] = None,
) -> ImplementationPlan:
    items = [line_item(c, metrics) for c in chosen if get_spec(c["type"])]
    lo = sum(i.cost_inr_low for i in items)
    hi = sum(i.cost_inr_high for i in items)
    cooling = sum(i.cooling_c for i in items)
    mid_lakh = (lo + hi) / 2.0 / INR_PER_LAKH

    phases = []
    for phase, label in PHASE_LABELS.items():
        in_phase = [i for i in items if i.phase == phase]
        if in_phase:
            phases.append(ImplementationPhase(
                phase=phase, label=label, items=[i.title for i in in_phase],
                cost_inr_low=sum(i.cost_inr_low for i in in_phase),
                cost_inr_high=sum(i.cost_inr_high for i in in_phase),
            ))

    prerequisites = list(dict.fromkeys(p for i in items for p in i.permits + i.conditions))
    assumptions = [
        f"Quantities are per {CLEARANCES['street_length_m']:.0f} m of street, per side for footpath, kerb and facade items.",
        f"Costs and coefficients are {CONFIG_LABEL}.",
        "Cooling figures are indicative localized estimates and are summed without interaction effects.",
    ] + [f"Metric scale: {n}" for n in metrics.notes]

    return ImplementationPlan(
        street_length_m=CLEARANCES["street_length_m"],
        line_items=items, phases=phases,
        total_cost_inr_low=lo, total_cost_inr_high=hi,
        total_cooling_c=round(cooling, 2),
        cooling_per_lakh_c=round(cooling / mid_lakh, 3) if mid_lakh > 0 else 0.0,
        budget_lakh=budget_lakh,
        within_budget=(mid_lakh <= budget_lakh) if budget_lakh else None,
        budget_excluded=budget_excluded or [],
        prerequisites=prerequisites,
        assumptions=assumptions,
        clearances={k: CLEARANCES[k] for k in (
            "min_clear_walkway_m", "kerb_buffer_m", "min_vertical_clearance_m", "parking_lane_width_m")},
        source=CONFIG_LABEL,
    )
