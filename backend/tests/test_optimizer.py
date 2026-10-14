"""Budget optimiser: exact multiple-choice knapsack, cross-checks and the portfolio endpoint."""

import itertools
import random
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer.candidates import Candidate
from optimizer.knapsack import Item, solve_dp, solve_greedy, solve_min_cost, total
from optimizer.service import optimize_candidates


def _cand(key, group, cost, cool, exposure=1000.0, phase="medium"):
    return Candidate(key=key, project_id=key.split(":")[0], project_name="P", type="tree_canopy", title=key, zone="z",
                     group=group, cost_inr={"low": cost * 0.8, "central": cost, "high": cost * 1.25},
                     cooling_c={"low": cool * 0.5, "central": cool, "high": cool * 1.5}, exposure=exposure,
                     exposure_source="assumed", phase=phase, phase_label=phase)


def _brute(items, budget):
    best = 0.0
    for r in range(len(items) + 1):
        for combo in itertools.combinations(range(len(items)), r):
            groups = [items[i].group for i in combo]
            if len(groups) != len(set(groups)):
                continue
            c, v = total(items, combo)
            if c <= budget + 1e-6:
                best = max(best, v)
    return best


@pytest.mark.parametrize("seed", range(8))
def test_dp_matches_brute_force_and_beats_greedy(seed):
    rng = random.Random(seed)
    # Costs in whole Rs 0.1 lakh units so discretisation is exact.
    items = [Item(f"i{k}", f"g{rng.randint(0, 4)}", rng.randint(1, 30) * 10_000, rng.uniform(0.1, 5)) for k in range(9)]
    budget = rng.randint(5, 80) * 10_000
    dp = solve_dp(items, budget)
    cost, value = total(items, dp)
    assert cost <= budget
    assert value == pytest.approx(_brute(items, budget))
    assert value >= total(items, solve_greedy(items, budget))[1] - 1e-9


def test_budget_never_exceeded_with_fractional_costs():
    items = [Item(f"i{k}", f"g{k}", 10_000 * k + 1_234, 1.0 + k) for k in range(1, 12)]
    for budget in (25_000, 77_777, 310_000):
        assert total(items, solve_dp(items, budget))[0] <= budget


def test_exclusivity_groups_respected():
    cands = [_cand("p1:a", "p1|z|street_trees", 100_000, 1.0), _cand("p1:b", "p1|z|street_trees", 120_000, 1.2),
             _cand("p2:c", "p2:c", 50_000, 0.3)]
    res = optimize_candidates(cands, budget_lakh=10.0)
    keys = [i["key"] for p in res["selected"] for i in p["items"]]
    assert not ({"p1:a", "p1:b"} <= set(keys))
    assert "p1:b" in keys and "p2:c" in keys


def test_objective_is_monotonic_in_budget():
    rng = random.Random(3)
    cands = [_cand(f"p{k}:x", f"g{k % 4}", rng.randint(1, 20) * 50_000, rng.uniform(0.2, 2)) for k in range(12)]
    values = [optimize_candidates(cands, b)["totals"]["objective_value"] for b in (1, 2, 4, 8, 16, 64)]
    assert values == sorted(values)
    frontier = optimize_candidates(cands, 8)["frontier"]
    assert len(frontier) == 10 and [f["objective_value"] for f in frontier] == sorted(f["objective_value"] for f in frontier)


def test_empty_candidates_handled():
    res = optimize_candidates([], budget_lakh=5)
    assert res["feasible"] and res["selected"] == [] and res["totals"]["cost_inr"]["central"] == 0
    assert res["frontier"] == []


def test_min_cost_for_target_is_cheapest():
    items = [Item("a", "a", 30_000, 2.0), Item("b", "b", 20_000, 1.0), Item("c", "c", 50_000, 3.5), Item("d", "d", 10_000, 0.4)]
    idx = solve_min_cost(items, 3.0)
    cost, value = total(items, idx)
    assert value >= 3.0
    cheapest = min(total(items, combo)[0] for r in range(1, 5) for combo in itertools.combinations(range(4), r)
                   if total(items, combo)[1] >= 3.0)
    assert cost == cheapest
    assert solve_min_cost(items, 100.0) is None


def test_robustness_and_rates_reported():
    cands = [_cand("p1:a", "g1", 200_000, 1.0), _cand("p2:b", "g2", 300_000, 1.5)]
    res = optimize_candidates(cands, budget_lakh=5.0)
    assert res["robustness"]["worst"]["same_selection"]["person_c"] < res["totals"]["person_c"]["central"]
    assert res["robustness"]["best"]["same_selection"]["person_c"] > res["totals"]["person_c"]["central"]
    assert res["totals"]["c_per_lakh"] == pytest.approx(2.5 / 5.0)
    assert res["greedy_cross_check"]["dp_at_least_greedy"]


# ---------------------------------------------------------------------------- endpoint, on real projects
SAMPLES = Path(__file__).resolve().parent.parent.parent / "gen-ai" / "public" / "samples"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECTS_DB", str(tmp_path / "p.sqlite3"))
    from main import app
    return TestClient(app)


def test_portfolio_endpoint_on_three_projects(client):
    ids = []
    for name, width in (("sample_2_sidewalk_road.jpg", "14"), ("sample_5_commercial_campus.jpg", "9"), ("sample_1_dense_urban.jpg", "7")):
        with open(SAMPLES / name, "rb") as f:
            r = client.post("/api/v1/projects", files={"photo": (name, f, "image/jpeg")},
                            data={"lat": "13.05", "lon": "80.22", "road_width_m": width})
        assert r.status_code == 201
        ids.append(r.json()["id"])
    res = client.post("/api/v1/portfolio/optimize", json={"project_ids": ids, "budget_lakh": 20})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["totals"]["cost_inr"]["central"] <= 20 * 100_000
    assert data["n_candidates"] > 0 and len(data["frontier"]) == 10
    assert data["greedy_cross_check"]["dp_at_least_greedy"]
    assert client.post("/api/v1/portfolio/optimize", json={"project_ids": ["nope"], "budget_lakh": 5}).status_code == 404
    bad = client.post("/api/v1/portfolio/optimize", json={"project_ids": ids, "budget_lakh": 5, "objective": "min_cost_for_target"})
    assert bad.status_code == 422
