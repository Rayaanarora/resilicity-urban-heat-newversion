"""Work-order content shared by the JSON, CSV and PDF exports."""

import datetime as dt
from typing import Any, Dict, List, Optional

from buildable.config import CONFIG_LABEL
from legacy_heat import card as legacy_card
from site_model import load_site_model

DISCLAIMER = (
    "Cooling figures are indicative literature ranges for each intervention, not predictions of a model, and are "
    "summed without interaction effects. The site heat model estimates pre-monsoon land surface temperature (the "
    "temperature of surfaces), not the air temperature people feel. Quantities come from a single photo and must be "
    "verified on site. Costs are indicative unit rates (" + CONFIG_LABEL + ")."
)


def model_cards() -> Dict[str, Any]:
    site = load_site_model()
    return {
        "site_model": {
            "use": "where is it hot (hotspot ranking)",
            "spatial_cv": site.metrics if site else None,
            "intervals_c": site.intervals if site else None,
            "features": site.features if site else None,
            "trained_on": site.card.get("trained_on") if site else None,
        },
        "legacy_land_cover_model": {
            "use": "legacy single-photo /heat flow only; low confidence; not used for intervention cooling",
            "spatial_cv": legacy_card.get("spatial_cv"),
        },
    }


def _line(item: Dict[str, Any]) -> Dict[str, Any]:
    q, cost, cool = item["quantity"], item["cost_inr"], item.get("cooling_c") or {}
    return {
        "intervention": item["title"],
        "type": item["type"],
        "side": item.get("target_zone"),
        "quantity": q["value"],
        "unit": q["unit"],
        "quantity_basis": q["basis"],
        "quantity_source": q["source"],
        "cost_inr_low": cost["low"],
        "cost_inr_central": cost["central"],
        "cost_inr_high": cost["high"],
        "cost_source": cost["source"],
        "cooling_c_low": cool.get("low"),
        "cooling_c_central": cool.get("central"),
        "cooling_c_high": cool.get("high"),
        "cooling_source": cool.get("source"),
        "phase": item["phase_label"],
        "lifespan_years": item["lifespan_years"],
        "permits": item["permits"],
        "conditions_to_verify": item["conditions"],
        "maintenance": item["maintenance"],
    }


def project_work_order(view: Dict[str, Any]) -> Dict[str, Any]:
    hotspot = view.get("hotspot") or {}
    metrics = view.get("street_metrics") or {}
    impl = view.get("implementation_plan") or {}
    return {
        "kind": "project_work_order",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "project": {"id": view["id"], "name": view["name"], "lat": view["lat"], "lon": view["lon"]},
        "data_mode": view.get("data_mode") or "unlinked",
        "hotspot": {
            "cell_id": hotspot.get("cell_id"),
            "rank": hotspot.get("rank"),
            "lst_pred_c": hotspot.get("lst_pred_c"),
            "lst_interval_90_c": hotspot.get("lst_interval_90_c"),
            "population": hotspot.get("population"),
            "sources": {"lst_pred_c": "model", "population": "measured" if view.get("data_mode") == "grid" else "assumed"},
        } if hotspot else None,
        "street": {
            "carriageway_m": metrics.get("road_width_m"),
            "carriageway_source": metrics.get("road_width_source"),
            "left_footpath_m": metrics.get("left_sidewalk_m"),
            "right_footpath_m": metrics.get("right_sidewalk_m"),
            "uncertainty_pct": metrics.get("uncertainty_pct"),
            "notes": metrics.get("notes", []),
        },
        "street_length_m": impl.get("street_length_m"),
        "interventions": [_line(i) for i in view.get("accepted", [])],
        "totals": {
            "cost_inr_low": impl.get("total_cost_inr_low"),
            "cost_inr_high": impl.get("total_cost_inr_high"),
            "cooling_c_central": impl.get("total_cooling_c"),
        },
        "rejected": view.get("rejected", []),
        "model_cards": model_cards(),
        "disclaimer": DISCLAIMER,
    }


def portfolio_work_order(result: Dict[str, Any], views: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """result is optimizer.service.optimize_portfolio output; views maps project id -> project_view."""
    projects: List[Dict[str, Any]] = []
    for sel in result.get("selected", []):
        view = views[sel["project_id"]]
        chosen = {i["key"] for i in sel["items"]}
        lines = [
            _line(a) for a in view["accepted"]
            if f"{view['id']}:{a['type']}:{a.get('target_zone') or 'site'}" in chosen
        ]
        projects.append({
            "project": {"id": view["id"], "name": view["name"]},
            "hotspot_rank": (view.get("hotspot") or {}).get("rank"),
            "interventions": lines,
            "rejected": view.get("rejected", []),
        })
    return {
        "kind": "portfolio_work_order",
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "objective": result.get("objective"),
        "budget_lakh": result.get("budget_lakh"),
        "data_mode": sorted({(v.get("data_mode") or "unlinked") for v in views.values()}),
        "totals": result.get("totals"),
        "robustness": result.get("robustness"),
        "phase_schedule": result.get("phase_schedule"),
        "projects": projects,
        "model_cards": model_cards(),
        "notes": result.get("notes", []),
        "disclaimer": DISCLAIMER,
    }


def boq_rows(order: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flat bill-of-quantities rows for CSV."""
    groups = [(order["project"], order["interventions"])] if order["kind"] == "project_work_order" else [
        (p["project"], p["interventions"]) for p in order["projects"]
    ]
    rows = []
    for project, lines in groups:
        for ln in lines:
            rows.append({
                "project_id": project["id"], "project": project["name"],
                **{k: ("; ".join(v) if isinstance(v, list) else v) for k, v in ln.items()},
            })
    return rows


def fmt_inr(n: Optional[float]) -> str:
    if n is None:
        return "-"
    return f"Rs {n / 100_000:.2f} L" if n >= 100_000 else f"Rs {n:,.0f}"
