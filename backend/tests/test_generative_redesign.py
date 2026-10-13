"""Tests for the generative redesign pipeline: planner, prompts, Kontext engine, caching and API."""

import asyncio
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
    HuggingFaceSpaceKontextProvider,
    RedesignEngine,
    build_kontext_redesign_prompt,
    build_kontext_refinement_prompt,
    build_redesign_prompt,
    build_refinement_prompt,
    validate_image_output,
    SpatialDesignPlan,
    SpatialInterventionSpec,
)
from image_generation.schemas import DesignProfile
import main
from main import app, compute_thermal_impact
from planner import generate_spatial_plan, analyze_scene_heuristics


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Isolated engine: only the HF Space provider (mocked per test) and a throwaway cache.
    monkeypatch.delenv("REDESIGN_OFFLINE", raising=False)
    monkeypatch.setattr(main, "redesign_engine", RedesignEngine(["hf_space"], cache_dir=str(tmp_path)))
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
def _measurable_street_seg(w=640, h=480):
    """Walls above, then 3.5 m sidewalks either side of a 7 m road (160/320/160 px)."""
    import numpy as np
    raw = np.full((h, w), 2, dtype=np.int32)          # sky
    raw[: int(h * 0.55), :160] = 0                    # wall
    raw[: int(h * 0.55), 480:] = 0
    raw[int(h * 0.55):, :160] = 3                     # sidewalk
    raw[int(h * 0.55):, 160:480] = 6                  # road
    raw[int(h * 0.55):, 480:] = 3
    return {"raw_preds": raw, "masks": [], "protected_objects": []}


def test_spatial_planner_generates_valid_schema():
    img = Image.new("RGB", (640, 480))
    surfaces = {"road": 35.0, "wall": 25.0, "pavement": 15.0, "vegetation": 5.0, "roof": 0.0}

    plan = generate_spatial_plan(img, surfaces, seg_result=_measurable_street_seg(), design_profile="balanced",
                                 road_width_m=7.0)
    assert plan.street_metrics.left_sidewalk_m and plan.street_metrics.left_sidewalk_m > 3.0
    assert isinstance(plan, SpatialDesignPlan)
    assert len(plan.interventions) > 0
    assert plan.design_profile == "balanced"
    assert "No visible rooftops" in " ".join(plan.constraints)

    first_iv = plan.interventions[0]
    assert first_iv.type in ("tree_canopy", "cool_pavement", "permeable_pave")
    assert first_iv.feasibility > 0.5
    assert first_iv.cooling_impact_c > 0.0


def test_unscaled_site_items_carry_a_survey_condition():
    # No carriageway in the photo: widths cannot be judged, so trees need a site survey first.
    # (An all-zero raw_preds would be ADE class 0 = wall everywhere, so use sky instead.)
    import numpy as np
    sky = {"raw_preds": np.full((480, 640), 2, np.int32), "masks": [], "protected_objects": []}
    plan = generate_spatial_plan(Image.new("RGB", (640, 480)), {"road": 35.0, "pavement": 15.0}, seg_result=sky)
    assert not plan.street_metrics.scale_available
    sidewalk_items = [iv for iv in plan.interventions if iv.target_zone in ("left_sidewalk", "right_sidewalk")]
    assert sidewalk_items
    assert all(any("survey" in c for c in iv.spatial_constraints) for iv in sidewalk_items)


def test_design_profiles_influence_plan():
    img = Image.new("RGB", (640, 480))
    surfaces = {"road": 40.0, "wall": 20.0, "pavement": 20.0, "vegetation": 5.0, "roof": 5.0}

    seg = _measurable_street_seg()
    ped_plan = generate_spatial_plan(img, surfaces, seg_result=seg, design_profile="pedestrian_first", road_width_m=7.0)
    types_ped = [i.type for i in ped_plan.interventions]
    assert "tree_canopy" in types_ped or "permeable_pave" in types_ped

    max_plan = generate_spatial_plan(img, surfaces, seg_result=seg, design_profile="maximum_cooling", road_width_m=7.0)
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


def test_kontext_prompt_preserves_architecture_and_lists_zones():
    specs = [
        SpatialInterventionSpec(type="tree_canopy", target_region="pavement", target_zone="left_sidewalk",
                                priority=1, coverage=0.4, placement="curb", visual_design="trees", reason="shade"),
        SpatialInterventionSpec(type="tree_canopy", target_region="pavement", target_zone="right_sidewalk",
                                priority=2, coverage=0.4, placement="curb", visual_design="trees", reason="shade"),
        SpatialInterventionSpec(type="cool_pavement", target_region="road", target_zone="roadway",
                                priority=3, coverage=0.6, placement="road", visual_design="cool", reason="albedo"),
    ]
    plan = SpatialDesignPlan(site_summary="s", heat_drivers=[], constraints=[], design_profile="balanced",
                             interventions=specs, overall_design_intent="cool it")
    prompt = build_kontext_redesign_prompt(plan, {"vehicle": 12.0, "person": 0.2})
    assert prompt.startswith("Add a continuous row of large mature street trees")
    assert "Keep the same buildings, shops, vehicles" in prompt
    assert "people" not in prompt  # below the detection threshold
    assert "street trees" in prompt and "both sidewalks" in prompt
    assert "cool reflective paving" in prompt and "the road" in prompt
    assert prompt.count("street trees") == 1  # tree zones merged into one clause


def test_kontext_refinement_prompt_keeps_rest_of_scene():
    prompt = build_kontext_refinement_prompt("Add a fountain in the plaza.")
    assert prompt.startswith("Add a fountain in the plaza.")
    assert "Keep everything else exactly the same" in prompt


def _noisy(w=300, h=200, seed=0):
    import numpy as np
    return Image.fromarray(np.random.default_rng(seed).integers(40, 220, (h, w, 3), dtype=np.uint8))


def test_engine_disk_cache_avoids_second_provider_call(tmp_path):
    engine = RedesignEngine(["hf_space"], cache_dir=str(tmp_path))
    src, out = _noisy(seed=1), _noisy(seed=2)
    mock_edit = AsyncMock(return_value=(out, None))
    with patch.object(HuggingFaceSpaceKontextProvider, "edit", new=mock_edit):
        first = asyncio.run(engine.generate(src, b"img-bytes", "prompt"))
        # A fresh engine on the same directory simulates a backend restart.
        second = asyncio.run(RedesignEngine(["hf_space"], cache_dir=str(tmp_path)).generate(src, b"img-bytes", "prompt"))
    assert first.image is not None and not first.cached
    assert second.cached and second.image.size == src.size
    assert mock_edit.call_count == 1


def test_engine_offline_mode_never_calls_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("REDESIGN_OFFLINE", "1")
    engine = RedesignEngine(["hf_space"], cache_dir=str(tmp_path))
    mock_edit = AsyncMock(return_value=(_noisy(seed=3), None))
    with patch.object(HuggingFaceSpaceKontextProvider, "edit", new=mock_edit):
        res = asyncio.run(engine.generate(_noisy(seed=4), b"other", "prompt"))
    assert res.image is None and "Offline mode" in res.error
    assert mock_edit.call_count == 0


def test_engine_falls_back_to_next_provider(tmp_path, monkeypatch):
    from image_generation import FalKontextProvider
    monkeypatch.setenv("FAL_KEY", "test-key")
    engine = RedesignEngine(["hf_space", "fal"], cache_dir=str(tmp_path))
    with patch.object(HuggingFaceSpaceKontextProvider, "edit", new=AsyncMock(return_value=(None, "quota reached"))),          patch.object(FalKontextProvider, "edit", new=AsyncMock(return_value=(_noisy(seed=5), None))):
        res = asyncio.run(engine.generate(_noisy(seed=6), b"fallback", "prompt"))
    assert res.image is not None and res.provider == "fal"


def test_engine_rejects_unchanged_output(tmp_path):
    engine = RedesignEngine(["hf_space"], cache_dir=str(tmp_path))
    src = _noisy(seed=7)
    with patch.object(HuggingFaceSpaceKontextProvider, "edit", new=AsyncMock(return_value=(src.copy(), None))):
        res = asyncio.run(engine.generate(src, b"same", "prompt"))
    assert res.image is None and "rejected" in res.error


# ---------------------------------------------------------------------------
# Caching & Thermal Separation Tests
# ---------------------------------------------------------------------------
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
# End-to-End API Redesign Tests with Mocked Kontext Provider
# ---------------------------------------------------------------------------
def test_analyze_and_redesign_with_mock_kontext(client, monkeypatch):
    fake_img = Image.new("RGB", (300, 200), color=(100, 120, 140))
    import numpy as np
    arr = np.random.randint(50, 220, (200, 300, 3), dtype=np.uint8)
    fake_generated = Image.fromarray(arr)

    with patch.object(
        HuggingFaceSpaceKontextProvider,
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


def test_analyze_and_redesign_handles_provider_failure_gracefully(client, monkeypatch):
    with patch.object(
        HuggingFaceSpaceKontextProvider,
        "edit",
        new=AsyncMock(return_value=(None, "Hugging Face Space GPU quota exceeded.")),
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

    with patch.object(HuggingFaceSpaceKontextProvider, "edit", new=mock_edit):
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
        HuggingFaceSpaceKontextProvider,
        "edit",
        new=AsyncMock(return_value=(None, "Hugging Face Space GPU quota exhausted")),
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
