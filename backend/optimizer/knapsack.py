"""Exact multiple-choice knapsack (pick at most one item per group) plus a greedy cross-check.

Costs are discretised to UNIT_INR (Rs 0.1 lakh) and rounded *up*, so a solution feasible in units is always
feasible in rupees.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

UNIT_INR = 10_000  # Rs 0.1 lakh


@dataclass(frozen=True)
class Item:
    key: str
    group: str
    cost_inr: float
    value: float


def cost_units(cost_inr: float) -> int:
    return int(math.ceil(cost_inr / UNIT_INR - 1e-9))


def _groups(items: Sequence[Item]) -> List[List[int]]:
    order: Dict[str, List[int]] = {}
    for i, it in enumerate(items):
        order.setdefault(it.group, []).append(i)
    return list(order.values())


def _dp_table(items: Sequence[Item], budget_units: int) -> Tuple[np.ndarray, List[np.ndarray], List[List[int]]]:
    """dp[b] = best value with cost <= b units; choice[g][b] = item chosen in group g (-1 = none)."""
    groups = _groups(items)
    dp = np.zeros(budget_units + 1)
    choices: List[np.ndarray] = []
    for g in groups:
        best = dp.copy()
        pick = np.full(budget_units + 1, -1, dtype=int)
        for i in g:
            c, v = cost_units(items[i].cost_inr), items[i].value
            if c > budget_units or v <= 0:
                continue
            cand = np.full(budget_units + 1, -np.inf)
            cand[c:] = dp[: budget_units + 1 - c] + v
            better = cand > best + 1e-12
            best = np.where(better, cand, best)
            pick = np.where(better, i, pick)
        dp = best
        choices.append(pick)
    return dp, choices, groups


def _reconstruct(items: Sequence[Item], choices: List[np.ndarray], groups: List[List[int]], b: int) -> List[int]:
    chosen = []
    for g in range(len(groups) - 1, -1, -1):
        i = int(choices[g][b])
        if i >= 0:
            chosen.append(i)
            b -= cost_units(items[i].cost_inr)
    return sorted(chosen)


def solve_dp(items: Sequence[Item], budget_inr: float) -> List[int]:
    """Indices of the value-maximising selection within budget (at most one item per group)."""
    budget_units = int(math.floor(budget_inr / UNIT_INR + 1e-9))
    if not items or budget_units <= 0:
        return []
    dp, choices, groups = _dp_table(items, budget_units)
    return _reconstruct(items, choices, groups, budget_units)


def solve_greedy(items: Sequence[Item], budget_inr: float) -> List[int]:
    """Value-per-rupee greedy, respecting groups. A cross-check only; it can be suboptimal."""
    order = sorted(range(len(items)), key=lambda i: -items[i].value / max(items[i].cost_inr, 1e-9))
    spent, used, chosen = 0.0, set(), []
    for i in order:
        it = items[i]
        if it.value <= 0 or it.group in used or spent + it.cost_inr > budget_inr + 1e-6:
            continue
        chosen.append(i)
        used.add(it.group)
        spent += it.cost_inr
    return sorted(chosen)


def solve_min_cost(items: Sequence[Item], target_value: float) -> Optional[List[int]]:
    """Cheapest selection whose value reaches target_value, or None if even everything is not enough."""
    if target_value <= 0:
        return []
    total_units = sum(cost_units(it.cost_inr) for it in items)
    if not items or total_units <= 0:
        return None
    dp, choices, groups = _dp_table(items, total_units)
    reach = np.flatnonzero(dp >= target_value - 1e-9)
    if reach.size == 0:
        return None
    return _reconstruct(items, choices, groups, int(reach[0]))


def total(items: Sequence[Item], idx: Sequence[int]) -> Tuple[float, float]:
    return float(sum(items[i].cost_inr for i in idx)), float(sum(items[i].value for i in idx))
