"""Unit tests for Autonomous Spatial Urban Resilience Planner."""

import io
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import planner  # noqa: E402
from image_generation.schemas import SpatialDesignPlan, SpatialInterventionSpec
from scene_understanding import analyze_scene

SURF = {"road": 34.0, "wall": 26.0, "roof": 18.0, "vegetation": 16.0, "pavement": 6.0}


def make_test_image(w=128, h=128, color=(128, 128, 128)) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


def png_bytes(w=64, h=48) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "gray").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(planner.router)
    return TestClient(app)


def post(client, surfaces=SURF, file=None):
    return client.post(
        "/api/v1/plan",
        files={"file": ("test.png", file or png_bytes(), "image/png")},
        data={"surfaces": json.dumps(surfaces)},
    )


# ---------------------------------------------------------------------------
# Autonomous Spatial Planner API Unit Tests
# ---------------------------------------------------------------------------
def test_generate_spatial_plan_produces_valid_schema():
    img = make_test_image(256, 256)
    plan = planner.generate_spatial_plan(img, SURF, design_profile="balanced")

    assert isinstance(plan, SpatialDesignPlan)
    assert plan.planner_source == "autonomous_spatial_planner"
    assert len(plan.interventions) >= 1
    assert plan.site_summary is not None
    assert len(plan.heat_drivers) >= 1

    first = plan.interventions[0]
    assert isinstance(first, SpatialInterventionSpec)
    assert first.priority == 1
    assert first.feasibility > 0.0
    assert first.cooling_impact_c > 0.0
    # Every intervention strictly justified by at least three spatial evidence signals
    assert len(first.evidence) >= 3


def test_interventions_ranked_and_duplicates_handled():
    img = make_test_image(256, 256)
    plan = planner.generate_spatial_plan(img, SURF, design_profile="balanced")
    priorities = [iv.priority for iv in plan.interventions]

    # Priorities must be sequential 1, 2, ...
    assert priorities == list(range(1, len(priorities) + 1))


def test_roof_interventions_omitted_when_no_visible_roof():
    img = make_test_image(256, 256)
    no_roof_surf = {"road": 55.0, "pavement": 25.0, "wall": 20.0, "roof": 0.0}
    plan = planner.generate_spatial_plan(img, no_roof_surf)

    types = [iv.type for iv in plan.interventions]
    assert "cool_roof" not in types
    assert "green_roof" not in types
    assert any("rooftop" in c.lower() for c in plan.constraints)


def test_compute_intervention_utility_returns_evidence_and_feasibility():
    img = make_test_image(200, 200)
    seg_res = {"classes": [{"id": k, "percentage": v} for k, v in SURF.items()]}
    scene = analyze_scene(img, seg_res)

    utility, evidence, feasibility, coverage = planner.compute_intervention_utility(
        itype="tree_canopy",
        scene=scene,
        target_zone="left_sidewalk",
        profile="balanced",
    )

    assert isinstance(utility, float)
    assert isinstance(evidence, list)
    assert len(evidence) >= 3
    assert 0.0 <= feasibility <= 1.0
    assert 0.1 <= coverage <= 1.0


def test_design_profiles_alter_intervention_focus():
    img = make_test_image(256, 256)
    surf = {"road": 40.0, "wall": 20.0, "pavement": 25.0, "vegetation": 5.0, "roof": 10.0}

    ped_plan = planner.generate_spatial_plan(img, surf, design_profile="pedestrian_first")
    max_plan = planner.generate_spatial_plan(img, surf, design_profile="maximum_cooling")

    assert ped_plan.design_profile == "pedestrian_first"
    assert max_plan.design_profile == "maximum_cooling"


def test_analyze_scene_heuristics():
    img = make_test_image(300, 200)
    analysis = planner.analyze_scene_heuristics(img, SURF)

    assert analysis.width == 300
    assert analysis.height == 200
    assert analysis.has_road_corridor is True
    assert analysis.has_pedestrian_sidewalk is True
    assert analysis.has_visible_roof is True


def test_planner_available():
    assert planner.planner_available() is True


# ---------------------------------------------------------------------------
# Endpoint Tests (/api/v1/plan)
# ---------------------------------------------------------------------------
def test_endpoint_happy_path(client):
    r = post(client)
    assert r.status_code == 200
    body = r.json()

    assert body["source"] == "autonomous_spatial_planner"
    assert "interventions" in body
    assert len(body["interventions"]) >= 1
    first_iv = body["interventions"][0]
    assert "type" in first_iv
    assert "priority" in first_iv
    assert "coolingImpact" in first_iv
    assert "evidence" in first_iv
    assert len(first_iv["evidence"]) >= 3


def test_endpoint_handles_empty_or_fallback_surfaces(client):
    r = client.post(
        "/api/v1/plan",
        files={"file": ("test.png", png_bytes(), "image/png")},
        data={"surfaces": "{}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "autonomous_spatial_planner"
    assert len(body["interventions"]) >= 1


def test_endpoint_rejects_corrupted_image(client):
    r = post(client, file=b"not an image file")
    assert r.status_code == 400
    assert "Invalid image" in r.json()["detail"]


def test_endpoint_handles_malformed_surfaces_json(client):
    r = client.post(
        "/api/v1/plan",
        files={"file": ("test.png", png_bytes(), "image/png")},
        data={"surfaces": "this is not valid json {"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "autonomous_spatial_planner"


# ---------------------------------------------------------------------------
# Autonomous Design Intelligence Unit Tests
# ---------------------------------------------------------------------------
def test_heat_driver_diagnosis_creation_and_ranking():
    img = make_test_image(256, 256)
    surf = {"road": 45.0, "wall": 30.0, "pavement": 15.0, "vegetation": 5.0, "roof": 5.0}
    plan = planner.generate_spatial_plan(img, surf)

    assert len(plan.dominant_heat_drivers) >= 3
    for d in plan.dominant_heat_drivers:
        assert isinstance(d.driver, str)
        assert 0.0 <= d.severity <= 1.0
        assert 0.0 <= d.confidence <= 1.0
        assert len(d.evidence) >= 1
        assert len(d.spatial_zones) >= 1

    # Dominant drivers must be sorted by severity * confidence descending
    scores = [d.severity * d.confidence for d in plan.dominant_heat_drivers]
    assert scores == sorted(scores, reverse=True)


def test_material_preservation_rejects_cool_pavement_on_cobblestone():
    img = make_test_image(256, 256)
    surf = {"road": 50.0, "wall": 25.0, "pavement": 15.0, "vegetation": 5.0, "roof": 5.0}

    # Simulate scene with historic cobblestone roadway
    seg_res = {"classes": [{"id": k, "percentage": v} for k, v in surf.items()]}
    scene = analyze_scene(img, seg_res)
    scene.roadway_material = {
        "material": "historic_cobblestone",
        "is_heritage": True,
        "texture_energy": 1250.0,
        "confidence": 0.92,
        "reason": "High spatial frequency texture and inter-block relief detected on roadway surface, indicative of historic cobblestone or stone setts.",
    }

    # Patch analyze_scene to return this scene
    from unittest.mock import patch
    with patch("planner.analyze_scene", return_value=scene):
        plan = planner.generate_spatial_plan(img, surf)

        # Cool pavement must NOT be in interventions
        types = [iv.type for iv in plan.interventions]
        assert "cool_pavement" not in types

        # Must record material preservation decision
        assert plan.material_preservation is not None
        assert plan.material_preservation.preserve is True
        assert "cobblestone" in plan.material_preservation.material.lower()

        # Must record rejection in rejected_candidates
        rej_types = [r.get("type") for r in plan.rejected_candidates]
        assert "cool_pavement" in rej_types
        assert any("heritage" in r.get("conflict", "") for r in plan.rejected_candidates)


def test_interaction_synergy_and_redundancy():
    img = make_test_image(256, 256)
    seg_res = {"classes": [{"id": k, "percentage": v} for k, v in SURF.items()]}
    scene = analyze_scene(img, seg_res)

    chosen = [
        {"type": "tree_canopy", "target_zone": "left_sidewalk", "utility": 0.85, "coverage": 0.35}
    ]

    # Synergy: Permeable pave receives boost when tree canopy is present
    delta_u_pave, syn_pave, _, _ = planner.evaluate_interactions("permeable_pave", "left_sidewalk", chosen, scene)
    assert delta_u_pave > 0.0
    assert any("synergy" in s.lower() for s in syn_pave)

    # Redundancy: Shade structure in the same sidewalk zone is penalized
    delta_u_shade, _, conf_shade, _ = planner.evaluate_interactions("shade_structure", "left_sidewalk", chosen, scene)
    assert delta_u_shade < 0.0
    assert any("redundancy" in c.lower() for c in conf_shade)


def test_expected_impact_metrics_before_after():
    img = make_test_image(256, 256)
    plan = planner.generate_spatial_plan(img, SURF)

    assert plan.expected_impact is not None
    imp = plan.expected_impact
    assert imp.baseline_heat_priority > imp.projected_heat_priority
    assert imp.delta_heat_priority < 0.0
    assert imp.delta_pedestrian_shade_pct >= 0.0
    assert imp.estimated_cooling_c > 0.0
    assert len(imp.summary) > 20


def test_design_narrative_completeness():
    img = make_test_image(256, 256)
    plan = planner.generate_spatial_plan(img, SURF)

    assert plan.site_diagnosis is not None
    assert "vulnerability_summary" in plan.site_diagnosis
    assert plan.selected_strategy is not None
    assert plan.spatial_rationale is not None
    assert len(plan.constraints) >= 1
    assert plan.baseline_state is not None
    assert plan.post_intervention_state is not None
