"""Portfolio optimisation across projects under one budget."""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from buildable.catalog import PHASE_LABELS
from projects import db
from projects.service import project_view

from .candidates import Candidate, candidates_for
from .knapsack import Item, solve_dp, solve_greedy, solve_min_cost

LAKH = 100_000
OBJECTIVES = ("person_c", "max_cooling", "min_cost_for_target")
SWEEP_POINTS = 10

# Scenario = (cost case, cooling case)
SCENARIOS = {"central": ("central", "central"), "worst": ("high", "low"), "best": ("low", "high")}


def _items(cands: Sequence[Candidate], objective: str, cost_case: str, cooling_case: str) -> List[Item]:
    return [Item(c.key, c.group, c.cost_inr[cost_case], c.value(objective, cooling_case)) for c in cands]


def _summary(cands: Sequence[Candidate], idx: Sequence[int], objective: str) -> Dict[str, Any]:
    sel = [cands[i] for i in idx]
    cost = {k: float(sum(c.cost_inr[k] for c in sel)) for k in ("low", "central", "high")}
    cooling = {k: float(sum(c.cooling_c[k] for c in sel)) for k in ("low", "central", "high")}
    person = {k: float(sum(c.cooling_c[k] * c.exposure for c in sel)) for k in ("low", "central", "high")}
    lakh = cost["central"] / LAKH
    return {
        "cost_inr": cost,
        "cooling_c_sum": cooling,
        "person_c": person,
        "c_per_lakh": cooling["central"] / lakh if lakh else 0.0,
        "person_c_per_lakh": person["central"] / lakh if lakh else 0.0,
        "objective_value": float(sum(c.value(objective) for c in sel)),
        "n_items": len(sel),
    }


def _solve(cands: Sequence[Candidate], objective: str, budget_inr: float, scenario: str = "central",
           target: Optional[float] = None) -> Optional[List[int]]:
    cost_case, cooling_case = SCENARIOS[scenario]
    items = _items(cands, objective, cost_case, cooling_case)
    if objective == "min_cost_for_target":
        return solve_min_cost(items, target or 0.0)
    return solve_dp(items, budget_inr)


def optimize_candidates(
    cands: Sequence[Candidate],
    budget_lakh: float,
    objective: str = "person_c",
    target_person_c: Optional[float] = None,
) -> Dict[str, Any]:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if objective == "min_cost_for_target" and not target_person_c:
        raise ValueError("min_cost_for_target needs target_person_c")
    budget_inr = budget_lakh * LAKH

    idx = _solve(cands, objective, budget_inr, target=target_person_c)
    if idx is None:
        return {"feasible": False, "reason": "Even all candidates together do not reach the target.",
                "n_candidates": len(cands)}

    central = _summary(cands, idx, objective)
    greedy_idx = solve_greedy(_items(cands, objective, "central", "central"), budget_inr) if objective != "min_cost_for_target" else idx
    greedy = _summary(cands, greedy_idx, objective)

    robustness = {}
    for scen in ("worst", "best"):
        cost_case, cooling_case = SCENARIOS[scen]
        sel = [cands[i] for i in idx]
        robustness[scen] = {
            "assumption": f"cost {cost_case}, cooling {cooling_case}",
            "same_selection": {
                "cost_inr": float(sum(c.cost_inr[cost_case] for c in sel)),
                "person_c": float(sum(c.cooling_c[cooling_case] * c.exposure for c in sel)),
                "cooling_c_sum": float(sum(c.cooling_c[cooling_case] for c in sel)),
                "within_budget": objective == "min_cost_for_target" or sum(c.cost_inr[cost_case] for c in sel) <= budget_inr + 1e-6,
            },
        }
        re_idx = _solve(cands, objective, budget_inr, scen, target_person_c)
        if re_idx is not None:
            re_sel = [cands[i] for i in re_idx]
            robustness[scen]["reoptimised"] = {
                "items": [c.key for c in re_sel],
                "cost_inr": float(sum(c.cost_inr[cost_case] for c in re_sel)),
                "person_c": float(sum(c.cooling_c[cooling_case] * c.exposure for c in re_sel)),
            }

    selected = [cands[i] for i in idx]
    by_project: Dict[str, Dict[str, Any]] = {}
    for c in selected:
        p = by_project.setdefault(c.project_id, {"project_id": c.project_id, "name": c.project_name, "items": []})
        p["items"].append({
            "key": c.key, "type": c.type, "title": c.title, "zone": c.zone, "phase": c.phase,
            "cost_inr": c.cost_inr, "cooling_c": c.cooling_c, "exposure": c.exposure,
            "person_c": {k: c.cooling_c[k] * c.exposure for k in ("low", "central", "high")},
            "sources": {"cost_inr": "assumed", "cooling_c": "literature", "exposure": c.exposure_source},
        })

    schedule = []
    for phase, label in PHASE_LABELS.items():
        in_phase = [c for c in selected if c.phase == phase]
        if in_phase:
            schedule.append({"phase": phase, "label": label, "items": [c.key for c in in_phase],
                             "cost_inr_central": float(sum(c.cost_inr["central"] for c in in_phase))})

    return {
        "feasible": True,
        "objective": objective,
        "budget_lakh": budget_lakh,
        "n_candidates": len(cands),
        "selected": list(by_project.values()),
        "totals": central,
        "greedy_cross_check": {**greedy, "dp_at_least_greedy": central["objective_value"] >= greedy["objective_value"] - 1e-9},
        "robustness": robustness,
        "phase_schedule": schedule,
        "frontier": budget_sweep(cands, objective, budget_lakh) if objective != "min_cost_for_target" else [],
        "exposure_source": sorted({c.exposure_source for c in cands}),
        "notes": [
            "Cooling figures are literature bands (indicative), not model predictions.",
            "person-°C = people in the hotspot cell x indicative local cooling; population is a labelled constant in demo mode.",
            "Costs are indicative unit rates; replace with the local Schedule of Rates before use.",
        ],
    }


def budget_sweep(cands: Sequence[Candidate], objective: str, budget_lakh: float) -> List[Dict[str, float]]:
    """Efficient frontier: optimal objective at SWEEP_POINTS budget levels up to the cost of everything."""
    if not cands:
        return []
    everything = sum(c.cost_inr["central"] for c in cands) / LAKH
    top = max(everything, budget_lakh)
    levels = np.linspace(top / SWEEP_POINTS, top, SWEEP_POINTS)
    out = []
    for lv in levels:
        idx = _solve(cands, objective, lv * LAKH) or []
        s = _summary(cands, idx, objective)
        out.append({"budget_lakh": round(float(lv), 2), "objective_value": s["objective_value"],
                    "person_c": s["person_c"]["central"], "cost_inr": s["cost_inr"]["central"], "n_items": s["n_items"]})
    return out


def optimize_portfolio(
    project_ids: Sequence[str],
    budget_lakh: float,
    objective: str = "person_c",
    phase_limit: Optional[str] = None,
    target_person_c: Optional[float] = None,
) -> Dict[str, Any]:
    cands: List[Candidate] = []
    missing = []
    for pid in project_ids:
        p = db.get_project(pid)
        if p is None:
            missing.append(pid)
            continue
        cands.extend(candidates_for(project_view(p), phase_limit))
    if missing:
        raise KeyError(", ".join(missing))
    result = optimize_candidates(cands, budget_lakh, objective, target_person_c)
    return {**result, "project_ids": list(project_ids), "phase_limit": phase_limit}
