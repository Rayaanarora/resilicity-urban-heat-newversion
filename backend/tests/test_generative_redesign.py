"""Tests for Generative AI urban redesign pipeline, validation, and Gemini provider."""

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from image_generation import (
    GeminiImageEditingProvider,
    build_redesign_prompt,
    build_refinement_prompt,
    validate_image_output,
    SpatialDesignPlan,
    SpatialInterventionSpec,
)
from image_generation.schemas import DesignProfile
from main import app, compute_cache_key, compute_thermal_impact
from planner import generate_spatial_plan, analyze_scene_heuristics


@pytest.fixture
def client():
    return TestClient(app)


def make_test_image(w=200, h=150, color=(120, 130, 140)):
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Image Validation & Aspect Ratio Preservation Tests
# ---------------------------------------------------------------------------
def test_validate_image_output_success():
    orig = (400, 300)
    gen = Image.new("RGB", (400, 300), color=(100, 150, 200))
    # Add gradient so it's not zero-variance
    import numpy as np
    arr = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
    gen = Image.fromarray(arr)

    is_valid, report, norm_img = validate_image_output(gen, orig)
    assert is_valid is True
    assert report.aspect_ratio_preserved is True
    assert norm_img.size == (400, 300)


def test_validate_image_output_rejects_blank():
    orig = (200, 200)
    blank = Image.new("RGB", (200, 200), color=(0, 0, 0))
    is_valid, report, _ = validate_image_output(blank, orig)
    assert is_valid is False
    assert report.non_blank_verified is False


def test_validate_image_normalizes_aspect_ratio_without_distortion():
    orig = (600, 300)  # 2:1 aspect ratio
    # Generated image is square (300, 300)
    import numpy as np
    arr = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
    gen = Image.fromarray(arr)

    is_valid, report, norm_img = validate_image_output(gen, orig)
    assert is_valid is True
    nw, nh = norm_img.size
    # Normalized image should have 2:1 aspect ratio
    assert abs(nw / float(nh) - 2.0) < 0.05


# ---------------------------------------------------------------------------
# Spatial Planner & Design Profile Tests
# ---------------------------------------------------------------------------
def test_spatial_planner_generates_valid_schema():
    img = Image.new("RGB", (640, 480))
    surfaces = {"road": 35.0, "wall": 25.0, "pavement": 15.0, "vegetation": 5.0, "roof": 0.0}

    plan = generate_spatial_plan(img, surfaces, design_profile="balanced")
    assert isinstance(plan, SpatialDesignPlan)
    assert len(plan.interventions) > 0
    assert plan.design_profile == "balanced"
    assert "No visible rooftops" in " ".join(plan.constraints)

    first_iv = plan.interventions[0]
    assert first_iv.type in ("tree_canopy", "cool_pavement", "permeable_pave")
    assert first_iv.feasibility > 0.5
    assert first_iv.cooling_impact_c > 0.0


def test_design_profiles_influence_plan():
    img = Image.new("RGB", (640, 480))
    surfaces = {"road": 40.0, "wall": 20.0, "pavement": 20.0, "vegetation": 5.0, "roof": 5.0}

    ped_plan = generate_spatial_plan(img, surfaces, design_profile="pedestrian_first")
    types_ped = [i.type for i in ped_plan.interventions]
    assert "tree_canopy" in types_ped or "permeable_pave" in types_ped

    max_plan = generate_spatial_plan(img, surfaces, design_profile="maximum_cooling")
    assert max_plan.design_profile == "maximum_cooling"


# ---------------------------------------------------------------------------
# Prompt Synthesis Tests
# ---------------------------------------------------------------------------
def test_prompt_builder_enforces_identity_and_interventions():
    spec = SpatialInterventionSpec(
        type="tree_canopy",
        target_region="pavement",
        priority=1,
        coverage=0.4,
        placement="sidewalk curb",
        visual_design="lush native shade trees",
        reason="heat relief",
    )
    plan = SpatialDesignPlan(
        site_summary="Hot urban canyon",
        heat_drivers=["Dark asphalt"],
        constraints=["Preserve facade"],
        design_profile="balanced",
        interventions=[spec],
        overall_design_intent="Create shaded corridor",
    )

    prompt = build_redesign_prompt(plan)
    assert "PRESERVE THE SCENE IDENTITY" in prompt
    assert "Tree Canopy" in prompt
    assert "Preserve facade" in prompt


# ---------------------------------------------------------------------------
# Caching & Thermal Separation Tests
# ---------------------------------------------------------------------------
def test_deterministic_cache_key():
    img1 = b"image_data_one"
    img2 = b"image_data_two"
    plan_dict = {"interventions": [{"type": "tree_canopy"}]}

    k1 = compute_cache_key(img1, plan_dict, "fast")
    k2 = compute_cache_key(img1, plan_dict, "fast")
    k3 = compute_cache_key(img2, plan_dict, "fast")

    assert k1 == k2
    assert k1 != k3


def test_thermal_impact_calculation():
    surfaces = {"road": 40.0, "wall": 20.0, "pavement": 20.0, "vegetation": 5.0, "roof": 5.0}
    interventions = [
        {"type": "tree_canopy", "coverage": 0.5},
        {"type": "cool_pavement", "coverage": 0.7},
    ]
    thermal = compute_thermal_impact(surfaces, interventions)
    assert thermal["totalCoolingReductionC"] > 0.5
    assert "baselineSurfaceTempC" in thermal
    assert "disclaimer" in thermal


# ---------------------------------------------------------------------------
# Security & Secret Leakage Prevention Tests
# ---------------------------------------------------------------------------
def test_health_endpoint_never_leaks_api_key(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert "gemini_configured" in data
    assert isinstance(data["gemini_configured"], bool)
    # Check that no string value looks like the API key
    serialized = json.dumps(data)
    assert "Ab8RN6" not in serialized
    assert "key" not in data or data.get("key") is None


# ---------------------------------------------------------------------------
# End-to-End API Redesign Tests with Mocked Gemini Provider
# ---------------------------------------------------------------------------
def test_analyze_and_redesign_with_mock_gemini(client):
    fake_img = Image.new("RGB", (300, 200), color=(100, 120, 140))
    import numpy as np
    arr = np.random.randint(50, 220, (200, 300, 3), dtype=np.uint8)
    fake_generated = Image.fromarray(arr)

    with patch.object(
        GeminiImageEditingProvider,
        "edit",
        new=AsyncMock(return_value=(fake_generated, None)),
    ):
        file_bytes = make_test_image(300, 200)
        res = client.post(
            "/api/v1/analyze-and-redesign",
            files={"image": ("test.jpg", file_bytes, "image/jpeg")},
            data={"design_profile": "balanced", "quality_tier": "fast"},
        )
        assert res.status_code == 200
        payload = res.json()
        assert "scene_analysis" in payload
        assert "design_plan" in payload
        assert "visualization" in payload
        assert payload["visualization"]["status"] == "ready"
        assert payload["visualization"]["image_url"].startswith("data:image/jpeg;base64,")
        assert payload["thermal_impact"]["totalCoolingReductionC"] > 0


def test_analyze_and_redesign_handles_provider_failure_gracefully(client):
    with patch.object(
        GeminiImageEditingProvider,
        "edit",
        new=AsyncMock(return_value=(None, "Gemini API quota exceeded for image generation.")),
    ):
        file_bytes = make_test_image(300, 200)
        res = client.post(
            "/api/v1/analyze-and-redesign",
            files={"image": ("test.jpg", file_bytes, "image/jpeg")},
            data={"design_profile": "maximum_cooling", "quality_tier": "fast"},
        )
        assert res.status_code == 200
        payload = res.json()
        # Returns design plan and thermal impact, but clearly states visualization unavailable
        assert payload["visualization"]["status"] == "unavailable"
        assert "quota exceeded" in payload["visualization"]["error_message"].lower()
        assert payload["scene_analysis"]["width"] == 300


# ---------------------------------------------------------------------------
# Open-Ended Refinement & Intent Parsing Tests
# ---------------------------------------------------------------------------
from image_generation.intent_parser import parse_refinement_intent_rules


def test_intent_parser_compound_instruction():
    instruction = (
        "Add four trees along the left sidewalk, replace the pavement with light permeable pavers, "
        "add two benches under the trees, and keep all buildings unchanged."
    )
    intent = parse_refinement_intent_rules(instruction)
    assert any("tree" in a.lower() for a in intent.add)
    assert any("paver" in a.lower() for a in intent.add) or any("paver" in m.lower() for m in intent.modify)
    assert any("benche" in a.lower() or "seating" in a.lower() for a in intent.add)
    assert any("building" in p.lower() for p in intent.preserve)


def test_intent_parser_removal_and_reduction():
    instruction = "Remove the pergola and make the trees less dense"
    intent = parse_refinement_intent_rules(instruction)
    assert any("pergola" in r.lower() for r in intent.remove)
    assert any("tree" in r.lower() for r in intent.remove) or any("scale" in m.lower() or "spacing" in m.lower() for m in intent.modify)


def test_intent_parser_plausibility_adaptation():
    instruction = "Add a giant tree in the middle of the road"
    intent = parse_refinement_intent_rules(instruction)
    # Must adapt to non-obstructing locations (curbs, verges, median) rather than active traffic lanes
    assert any("traffic" in c.lower() or "lane" in c.lower() or "curb" in c.lower() for c in intent.spatial_constraints)


def test_refine_prompt_builder_includes_constraints():
    intent = parse_refinement_intent_rules("Keep storefronts unchanged and add shaded seating")
    prompt = build_refinement_prompt("Keep storefronts unchanged and add shaded seating", intent=intent)
    assert "CURRENT SCENE:" in prompt
    assert "PRESERVE (CRITICAL ARCHITECTURAL CONSTRAINTS):" in prompt
    assert "storefronts" in prompt.lower()
    assert "regenerate the city" in prompt.lower()


def test_refine_design_endpoint_success_and_caching(client):
    import numpy as np
    fake_img = Image.fromarray(np.random.randint(60, 200, (200, 300, 3), dtype=np.uint8))
    mock_edit = AsyncMock(return_value=(fake_img, None))

    with patch.object(GeminiImageEditingProvider, "edit", new=mock_edit):
        file_bytes = make_test_image(300, 200)
        res = client.post(
            "/api/v1/refine-design",
            files={"image": ("redesign.jpg", file_bytes, "image/jpeg")},
            data={"instruction": "Add two more trees along the sidewalk", "quality_tier": "fast"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ready"
        assert data["image_url"].startswith("data:image/jpeg;base64,")
        assert data["instruction"] == "Add two more trees along the sidewalk"
        assert "intent" in data
        assert mock_edit.call_count == 1

        # Second identical call should hit the cache without calling edit again!
        res_cached = client.post(
            "/api/v1/refine-design",
            files={"image": ("redesign.jpg", file_bytes, "image/jpeg")},
            data={"instruction": "Add two more trees along the sidewalk", "quality_tier": "fast"},
        )
        assert res_cached.status_code == 200
        assert mock_edit.call_count == 1  # Unchanged! Cached!


def test_refine_design_endpoint_handles_failure_gracefully(client):
    with patch.object(
        GeminiImageEditingProvider,
        "edit",
        new=AsyncMock(return_value=(None, "Gemini quota exhausted")),
    ):
        file_bytes = make_test_image(300, 200)
        res = client.post(
            "/api/v1/refine-design",
            files={"image": ("redesign.jpg", file_bytes, "image/jpeg")},
            data={"instruction": "Turn this into a pocket park", "quality_tier": "fast"},
        )
        assert res.status_code == 200
        data = res.json()
        # Returns unavailable status with error message without crashing
        assert data["status"] == "unavailable"
        assert "quota" in data["error_message"].lower()
        assert data["image_url"] is None
