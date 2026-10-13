"""Tests for the buildable street-design package (metrics, gating, budget, implementation plan)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buildable import estimate_street_metrics
from buildable.config import METRICS


# ---------------------------------------------------------------------------
# 1. Metric estimation
# ---------------------------------------------------------------------------
def _street_masks(h=300, w=400, perspective=False, road=(100, 300), left=(40, 100), right=(300, 360)):
    """Road and sidewalk masks. With perspective, every span shrinks toward the top while the
    width ratios on each row stay the same, as for a camera looking along the street."""
    road_m = np.zeros((h, w), bool)
    left_m = np.zeros((h, w), bool)
    right_m = np.zeros((h, w), bool)
    cx = w / 2.0
    for y in range(h):
        s = (0.3 + 0.7 * y / (h - 1)) if perspective else 1.0
        def span(a, b):
            return int(round(cx + (a - cx) * s)), int(round(cx + (b - cx) * s))
        r0, r1 = span(*road)
        road_m[y, r0:r1] = True
        if left:
            l0, l1 = span(*left)
            left_m[y, l0:l1] = True
        if right:
            q0, q1 = span(*right)
            right_m[y, q0:q1] = True
    return road_m, left_m, right_m


def test_metrics_recovers_sidewalk_width_from_carriageway_scale():
    road, left, right = _street_masks()
    m = estimate_street_metrics(road, left, right, road_width_m=7.0)
    # 60 px sidewalk / 200 px road * 7 m = 2.1 m
    assert m.left_sidewalk_m == pytest.approx(2.1, abs=0.05)
    assert m.right_sidewalk_m == pytest.approx(2.1, abs=0.05)
    assert m.reliable and not m.road_truncated
    assert m.scale_source == "user"


def test_metrics_reports_thirty_percent_band():
    road, left, right = _street_masks()
    m = estimate_street_metrics(road, left, right, road_width_m=7.0)
    assert m.uncertainty_pct == METRICS["uncertainty_pct"] == 30
    lo, hi = m.left_range_m
    assert lo == pytest.approx(m.left_sidewalk_m * 0.7, abs=0.01)
    assert hi == pytest.approx(m.left_sidewalk_m * 1.3, abs=0.01)


def test_metrics_is_perspective_invariant_and_scales_with_reference():
    road, left, right = _street_masks(perspective=True)
    assert estimate_street_metrics(road, left, right, 7.0).left_sidewalk_m == pytest.approx(2.1, abs=0.1)
    assert estimate_street_metrics(road, left, right, 10.0).left_sidewalk_m == pytest.approx(3.0, abs=0.15)


def test_metrics_default_scale_is_flagged():
    road, left, right = _street_masks()
    m = estimate_street_metrics(road, left, right)
    assert m.road_width_m == METRICS["default_road_width_m"]
    assert m.scale_source == "default"
    assert any("road_width_m" in n for n in m.notes)


def test_metrics_missing_sidewalk_is_none_not_guessed():
    road, left, right = _street_masks(left=None)
    m = estimate_street_metrics(road, left, right, 7.0)
    assert m.left_sidewalk_m is None and m.left_range_m is None
    assert m.right_sidewalk_m is not None
    assert any("Left sidewalk not measurable" in n for n in m.notes)


def test_metrics_ignores_segmentation_specks():
    road, left, right = _street_masks(right=(300, 303))  # 3 px ~ 0.1 m
    assert estimate_street_metrics(road, left, right, 7.0).right_sidewalk_m is None


def test_metrics_truncated_road_widens_band_and_is_unreliable():
    road, left, right = _street_masks(road=(100, 400), right=None)
    m = estimate_street_metrics(road, left, right, 7.0)
    assert m.road_truncated and not m.reliable
    assert m.uncertainty_pct == METRICS["truncated_uncertainty_pct"]


def test_metrics_without_road_is_unavailable():
    z = np.zeros((300, 400), bool)
    m = estimate_street_metrics(z, z, z, 7.0)
    assert not m.reliable and m.left_sidewalk_m is None


# ---------------------------------------------------------------------------
# 2. Buildable catalog
# ---------------------------------------------------------------------------
from buildable import catalog_types, get_spec
from buildable.config import CONFIG_LABEL

NEW_TYPES = ["facade_awning", "bus_shelter", "solar_canopy", "kerb_bioswale", "parking_lane_tree", "cool_wall"]
EXISTING_TYPES = ["tree_canopy", "green_roof", "cool_roof", "cool_pavement", "permeable_pave", "shade_structure"]


@pytest.mark.parametrize("itype", NEW_TYPES + EXISTING_TYPES)
def test_catalog_entry_has_engineering_spec(itype):
    spec = get_spec(itype)
    assert spec is not None, itype
    assert spec.dimensions and spec.unit and spec.maintenance and spec.permits
    assert spec.lifespan_years > 0
    lo, hi = spec.cost_inr
    assert 0 < lo <= hi
    c_lo, c_mid, c_hi = spec.cooling_c
    assert 0 <= c_lo <= c_mid <= c_hi
    assert spec.phase in ("quick_win", "medium", "long_term")
    assert "{zones}" in spec.kontext
    assert spec.source == CONFIG_LABEL


def test_config_is_labelled_indicative():
    assert "indicative" in CONFIG_LABEL and "Schedule of Rates" in CONFIG_LABEL


def test_awning_depth_and_planner_text_come_from_config():
    import planner
    assert get_spec("facade_awning").dimensions["depth_m"] == pytest.approx(1.2)
    spacing = get_spec("tree_canopy").spacing_m
    assert f"{spacing:.0f} m apart" in planner.CATALOG["tree_canopy"]["placement"]
    assert set(planner.CATALOG) <= set(catalog_types())


# ---------------------------------------------------------------------------
# 3. Gating rules
# ---------------------------------------------------------------------------
from buildable import StreetMetrics
from buildable.config import CLEARANCES
from buildable.gating import SiteContext, gate
from buildable.observations import SiteObservations, observe_site

TREE_REQUIRED_M = get_spec("tree_canopy").dimensions["pit_m"] + CLEARANCES["kerb_buffer_m"] + CLEARANCES["min_clear_walkway_m"]


def _ctx(left=4.0, right=4.0, scale=True, obstacles=None, junction=None, climate="tropical_humid",
         hw=1.0, heritage=False, road=7.0):
    def band(v):
        return None if v is None else [round(v * 0.7, 2), round(v * 1.3, 2)]
    metrics = StreetMetrics(road_width_m=road, left_sidewalk_m=left, right_sidewalk_m=right,
                            left_range_m=band(left), right_range_m=band(right), scale_available=scale,
                            reliable=scale, rows_used=50 if scale else 0)
    obs = SiteObservations(
        obstacle_fractions=obstacles or {z: {"pole": 0.0, "vehicle": 0.0, "person": 0.0}
                                         for z in ("left_sidewalk", "right_sidewalk")},
        junction_suspected=junction or {"left_sidewalk": False, "right_sidewalk": False},
    )
    return SiteContext(metrics=metrics, observations=obs, climate_zone=climate, canyon_hw_proxy=hw,
                       canyon_strength=0.5, wall_density=0.3,
                       solar_exposure={"left_sidewalk": 0.7, "right_sidewalk": 0.7, "roadway": 0.7},
                       heritage_roadway=heritage, heritage_material="historic_cobblestone" if heritage else "")


def test_gate_tree_pit_needs_clear_walkway_after_pit():
    narrow = gate("tree_canopy", "left_sidewalk", _ctx(left=TREE_REQUIRED_M - 0.5))
    assert not narrow.allowed and narrow.rule == "tree_canopy_clear_width"
    assert f"{TREE_REQUIRED_M:.1f} m" in narrow.reason and narrow.measured["sidewalk_m"] < TREE_REQUIRED_M
    wide = gate("tree_canopy", "left_sidewalk", _ctx(left=TREE_REQUIRED_M * 1.6))
    assert wide.allowed and not wide.conditions


def test_gate_tree_borderline_width_requires_site_verification():
    res = gate("tree_canopy", "left_sidewalk", _ctx(left=TREE_REQUIRED_M + 0.1))
    assert res.allowed and any("verify on site" in c for c in res.conditions)


def test_gate_tree_rejected_when_road_measured_but_no_sidewalk():
    res = gate("tree_canopy", "left_sidewalk", _ctx(left=None))
    assert not res.allowed and res.rule == "no_measurable_sidewalk"


def test_gate_tree_without_scale_requires_survey():
    res = gate("tree_canopy", "left_sidewalk", _ctx(left=None, scale=False))
    assert res.allowed and any("survey" in c for c in res.conditions)


def test_gate_tree_junction_sight_line_setback():
    res = gate("tree_canopy", "left_sidewalk", _ctx(junction={"left_sidewalk": True, "right_sidewalk": False}))
    assert res.allowed
    assert res.partial_rejections and res.partial_rejections[0]["rule"] == "junction_sight_line"
    assert f"{CLEARANCES['junction_sight_clearance_m']:.0f} m" in res.partial_rejections[0]["reason"]


@pytest.mark.parametrize("group,limit_key", [("pole", "max_pole_strip_frac"), ("vehicle", "max_vehicle_strip_frac"),
                                             ("person", "max_person_strip_frac")])
def test_gate_tree_rejected_near_poles_vehicles_people(group, limit_key):
    fr = {"pole": 0.0, "vehicle": 0.0, "person": 0.0}
    fr[group] = CLEARANCES[limit_key] + 0.05
    res = gate("tree_canopy", "right_sidewalk",
               _ctx(obstacles={"left_sidewalk": dict(fr), "right_sidewalk": fr}))
    assert not res.allowed and res.rule == f"tree_near_{group}"


def test_gate_misting_rejected_in_humid_tropics_only():
    humid = gate("misting_system", "left_sidewalk", _ctx(climate="tropical_humid"))
    assert not humid.allowed and humid.rule == "misting_humid_climate"
    assert gate("misting_system", "left_sidewalk", _ctx(climate="arid")).allowed


def test_gate_cool_pavement_only_in_deep_canyons():
    req = CLEARANCES["deep_canyon_hw_proxy_min"]
    open_street = gate("cool_pavement", "roadway", _ctx(hw=req - 0.2))
    assert not open_street.allowed and open_street.rule == "cool_pavement_open_street"
    assert "pedestrians" in open_street.reason
    assert gate("cool_pavement", "roadway", _ctx(hw=req + 0.2)).allowed


def test_gate_heritage_roadway_still_rejected():
    res = gate("cool_pavement", "roadway", _ctx(hw=2.0, heritage=True))
    assert not res.allowed and res.rule == "heritage_preservation" and "cobblestone" in res.reason


def test_observe_site_detects_poles_in_kerb_strip():
    road, left, right = _street_masks()
    raw = np.zeros(road.shape, np.int32)
    raw[200:260, 300:304] = 93  # utility pole just right of the road, beside the right sidewalk
    obs = observe_site(raw, road, left, right)
    assert obs.obstacle_fractions["right_sidewalk"]["pole"] > 0
    assert obs.obstacle_fractions["left_sidewalk"]["pole"] == 0


def test_planner_writes_gating_reasons_to_rejected_candidates():
    from PIL import Image
    from planner import generate_spatial_plan
    w, h = 640, 480
    raw = np.full((h, w), 2, np.int32)
    raw[: int(h * 0.55), :] = 0
    raw[int(h * 0.55):, :60] = 3          # ~1.3 m sidewalks on a 7 m (320 px) road: too narrow for pits
    raw[int(h * 0.55):, 60:380] = 6
    raw[int(h * 0.55):, 380:440] = 3
    plan = generate_spatial_plan(Image.new("RGB", (w, h)), {"road": 35.0, "pavement": 10.0, "wall": 40.0},
                                 seg_result={"raw_preds": raw, "masks": [], "protected_objects": []}, road_width_m=7.0)
    tree_rejections = [r for r in plan.rejected_candidates if r["type"] == "tree_canopy"]
    assert tree_rejections and all(r["reason"] and r["rule"] for r in tree_rejections)
    assert all(iv.type != "tree_canopy" for iv in plan.interventions)


# ---------------------------------------------------------------------------
# 4. Implementation plan and budget selection
# ---------------------------------------------------------------------------
import math

from buildable import build_implementation_plan, select_within_budget
from buildable.config import BUDGET
from buildable.implementation import mid_cost_lakh

LENGTH = CLEARANCES["street_length_m"]


def _m(left=4.0, right=4.0, road=7.0):
    return StreetMetrics(road_width_m=road, left_sidewalk_m=left, right_sidewalk_m=right, scale_available=True)


def test_tree_quantity_uses_spacing_and_junction_setback():
    spacing = get_spec("tree_canopy").spacing_m
    plain = build_implementation_plan([{"type": "tree_canopy", "target_zone": "left_sidewalk"}], _m())
    assert plain.line_items[0].quantity == math.floor(LENGTH / spacing)
    setback = CLEARANCES["junction_sight_clearance_m"]
    near_junction = build_implementation_plan(
        [{"type": "tree_canopy", "target_zone": "left_sidewalk", "excluded_length_m": setback}], _m())
    assert near_junction.line_items[0].quantity == math.floor((LENGTH - setback) / spacing)
    assert "junction setback" in near_junction.line_items[0].basis


def test_line_item_costs_cooling_per_lakh_and_phases():
    items = [
        {"type": "cool_wall", "target_zone": "left_sidewalk", "cooling": 0.3},
        {"type": "tree_canopy", "target_zone": "left_sidewalk", "cooling": 1.2},
    ]
    plan = build_implementation_plan(items, _m())
    wall, tree = plan.line_items
    lo, hi = get_spec("cool_wall").cost_inr
    assert (wall.cost_inr_low, wall.cost_inr_high) == (round(wall.quantity * lo), round(wall.quantity * hi))
    mid_lakh = (tree.cost_inr_low + tree.cost_inr_high) / 2 / BUDGET["inr_per_lakh"]
    assert tree.cooling_per_lakh_c == pytest.approx(1.2 / mid_lakh, rel=1e-2)
    assert plan.total_cost_inr_low == wall.cost_inr_low + tree.cost_inr_low
    assert [p.phase for p in plan.phases] == ["quick_win", "medium"]
    assert any("Horticulture" in p for p in plan.prerequisites)
    assert plan.street_length_m == LENGTH


def test_unmeasured_quantities_are_flagged():
    plan = build_implementation_plan([{"type": "permeable_pave", "target_zone": "left_sidewalk", "coverage": 0.5}],
                                     _m(left=None))
    assert plan.line_items[0].quantity_flagged and "not measured" in plan.line_items[0].basis


def test_budget_selection_prefers_cost_effectiveness_within_cap():
    m = _m()
    cands = [
        {"type": "bus_shelter", "target_zone": "left_sidewalk", "cooling": 0.3, "utility": 0.9},
        {"type": "cool_wall", "target_zone": "left_sidewalk", "cooling": 0.3, "utility": 0.5},
        {"type": "tree_canopy", "target_zone": "left_sidewalk", "cooling": 1.2, "utility": 0.8},
    ]
    budget = mid_cost_lakh(cands[1], m) + mid_cost_lakh(cands[2], m) + 0.1   # not enough for the shelter too
    chosen, excluded = select_within_budget(cands, budget, m, lambda c, ch: (0.0, [], [], False))
    assert {c["type"] for c in chosen} == {"cool_wall", "tree_canopy"}
    assert sum(mid_cost_lakh(c, m) for c in chosen) <= budget
    assert excluded and excluded[0]["type"] == "bus_shelter" and excluded[0]["rule"] == "over_budget"


def test_budget_selection_respects_interaction_rejections():
    m = _m()
    cands = [{"type": "tree_canopy", "target_zone": "left_sidewalk", "cooling": 1.2},
             {"type": "cool_wall", "target_zone": "left_sidewalk", "cooling": 0.3}]
    chosen, _ = select_within_budget(cands, 1000.0, m, lambda c, ch: (0.0, [], [], c["type"] == "cool_wall"))
    assert [c["type"] for c in chosen] == ["tree_canopy"]


def test_planner_budget_mode_stays_within_cap():
    from PIL import Image
    from planner import generate_spatial_plan
    w, h = 640, 480
    raw = np.full((h, w), 2, np.int32)
    raw[: int(h * 0.55), :] = 0
    raw[int(h * 0.55):, :160] = 3
    raw[int(h * 0.55):, 160:480] = 6
    raw[int(h * 0.55):, 480:] = 3
    seg = {"raw_preds": raw, "masks": [], "protected_objects": []}
    plan = generate_spatial_plan(Image.new("RGB", (w, h)), {"road": 35.0, "pavement": 20.0, "wall": 35.0},
                                 seg_result=seg, road_width_m=7.0, budget_lakh=5.0)
    ip = plan.implementation_plan
    assert ip is not None and ip.budget_lakh == 5.0 and ip.within_budget
    assert (ip.total_cost_inr_low + ip.total_cost_inr_high) / 2 <= 5.0 * BUDGET["inr_per_lakh"]


# ---------------------------------------------------------------------------
# 5. Integration: thermal coefficients, new candidate types, interactions
# ---------------------------------------------------------------------------
from buildable import cooling_band


@pytest.mark.parametrize("itype", NEW_TYPES + EXISTING_TYPES)
def test_every_catalog_type_has_nonzero_cooling(itype):
    lo, mid, hi = cooling_band(itype, 0.5)
    assert 0 < lo <= mid <= hi


def test_thermal_impact_reports_bands_and_flags_unknown_types():
    from main import card, compute_thermal_impact
    surfaces = {"road": 40.0, "wall": 25.0, "pavement": 20.0, "vegetation": 5.0}
    out = compute_thermal_impact(surfaces, [
        {"type": "facade_awning", "coverage": 0.6},
        {"type": "cool_wall", "coverage": 0.7},
        {"type": "teleporter", "coverage": 0.5},
    ])
    assert out["totalCoolingReductionC"] > 0
    assert {p["type"] for p in out["perIntervention"]} == {"facade_awning", "cool_wall"}
    assert out["unmodelledInterventions"] == ["teleporter"] and "teleporter" in out["warning"]
    lo, hi = out["coolingRangeC"]
    assert lo <= out["totalCoolingReductionC"] <= hi
    assert out["mlModelR2"] == pytest.approx(card["spatial_cv"]["R2"], abs=0.01)
    d_lo, d_hi = out["mlModelDeltaRangeC"]
    assert d_hi - d_lo == pytest.approx(2 * card["spatial_cv"]["RMSE"], abs=0.02)
    assert out["confidence"] == "low"


def _street_seg(left_px, road_px, right_px, w=640, h=480):
    raw = np.full((h, w), 2, np.int32)
    raw[: int(h * 0.55), :] = 0
    y = int(h * 0.55)
    raw[y:, :left_px] = 3
    raw[y:, left_px:left_px + road_px] = 6
    raw[y:, left_px + road_px:left_px + road_px + right_px] = 3
    return {"raw_preds": raw, "masks": [], "protected_objects": []}


def test_narrow_sidewalks_on_wide_road_get_parking_lane_trees():
    from PIL import Image
    from planner import generate_spatial_plan
    # 1.3 m footpaths on a 10 m carriageway: pits do not fit, a build-out leaves 8 m.
    plan = generate_spatial_plan(Image.new("RGB", (640, 480)), {"road": 40.0, "pavement": 8.0, "wall": 40.0},
                                 seg_result=_street_seg(42, 320, 42), road_width_m=10.0)
    types = [iv.type for iv in plan.interventions]
    assert "tree_canopy" not in types
    assert "parking_lane_tree" in types
    assert any(r["type"] == "tree_canopy" and r["rule"] == "tree_canopy_clear_width" for r in plan.rejected_candidates)


def test_interactions_reject_redundant_and_conflicting_pairs():
    from planner import evaluate_interactions
    from scene_understanding import analyze_scene
    from PIL import Image
    scene = analyze_scene(Image.new("RGB", (640, 480)), _street_seg(160, 320, 160))
    tree_left = [{"type": "tree_canopy", "target_zone": "left_sidewalk"}]
    assert evaluate_interactions("parking_lane_tree", "left_sidewalk", tree_left, scene)[3]
    assert evaluate_interactions("solar_canopy", "left_sidewalk", tree_left, scene)[3]
    assert not evaluate_interactions("solar_canopy", "right_sidewalk", tree_left, scene)[3]
    awning_left = [{"type": "facade_awning", "target_zone": "left_sidewalk"}]
    assert evaluate_interactions("shade_structure", "left_sidewalk", awning_left, scene)[3]
    assert not evaluate_interactions("facade_awning", "right_sidewalk", awning_left, scene)[3]
