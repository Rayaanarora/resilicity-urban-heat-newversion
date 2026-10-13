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
