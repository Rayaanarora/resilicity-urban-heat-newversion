"""Tests for Design Intelligence V2: Autonomous Urban Designer.

Validates Requirements 1-12 from DESIGN INTELLIGENCE V2:
1. DesignIntent data structure & schema constraints.
2. Design Strategist reasoning from scene diagnosis.
3. Mechanism-first intervention selection and rejection tracking.
4. Intent-aware spatial layout and clearance guarantees.
5. Tree anchor geometry (ground anchor, trunk corridor, canopy, pit).
6. Hard negative design rules injection into renderer prompts.
7. Design Critic evaluation (presence, spatial, preservation, unauthorized planter box detection).
8. Bounded retry adaptation recommendations.
9. End-to-end integration with SpatialDesignPlan and UnifiedRedesignResponse.
10. Strict 100% local, no-cloud compliance.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_generation.schemas import (
    DesignIntent,
    DesignCritique,
    SpatialDesignPlan,
    SpatialInterventionSpec,
    UnifiedRedesignResponse,
    SceneAnalysis,
    VisualizationOutput,
    ValidationReport,
)
from image_generation.design_critic import evaluate_design_critique
from image_generation.controlled_generation import build_intervention_prompt
from image_generation.layout_engine import compute_tree_row_layout
from planner import synthesize_design_intent, generate_spatial_plan
from scene_understanding import analyze_scene


def test_design_intent_schema():
    """Verify DesignIntent instantiates with all required fields and valid defaults."""
    intent = DesignIntent(
        primary_objective="Create continuous pedestrian shade without reducing roadway capacity",
        secondary_objectives=["increase vegetation", "increase pedestrian thermal comfort"],
        dominant_heat_mechanisms=["direct_solar_exposure", "insufficient_pedestrian_shade"],
        target_zones=["left_sidewalk"],
        pedestrian_priority=0.92,
        shade_priority=0.96,
        vegetation_priority=0.82,
        permeability_priority=0.61,
        albedo_priority=0.45,
        preservation_requirements=["preserve roadway geometry", "preserve building facades"],
        hard_constraints=["do not obstruct active roadway", "do not block building entrances"],
        acceptable_interventions=["mature street trees", "tree pits"],
        rejected_interventions=["large raised planter barriers", "decorative flower strips without shade"],
        desired_spatial_pattern="linear tree corridor",
        desired_continuity=0.85,
        visual_character="natural mature urban roadside trees flush with sidewalk",
        hard_negative_rules=["planter box", "giant planter", "flower bed"],
        confidence=0.92,
        rationale="Pedestrian shade deficit on wide sidewalk requires mature street trees.",
    )
    assert intent.primary_objective.startswith("Create continuous pedestrian shade")
    assert "left_sidewalk" in intent.target_zones
    assert intent.pedestrian_priority > 0.90
    assert "planter box" in intent.hard_negative_rules


def test_design_strategist_synthesizes_intent():
    """Verify synthesize_design_intent creates scene-specific intent from scene perception."""
    w, h = 800, 600
    img = Image.new("RGB", (w, h), (120, 120, 120))
    seg_result = {
        "surfaces": {"road": 35.0, "pavement": 25.0, "wall": 25.0, "sky": 15.0},
        "raw_preds": np.zeros((h, w), dtype=np.int32),
        "masks": [],
    }
    # Mark left side as pavement (class 3) and right side as road (class 6)
    seg_result["raw_preds"][:, :300] = 3
    seg_result["raw_preds"][:, 300:] = 6

    scene = analyze_scene(img, seg_result)
    intent = synthesize_design_intent(
        scene=scene,
        surfaces={"road": 35.0, "pavement": 25.0, "wall": 25.0, "sky": 15.0},
        design_profile="pedestrian_first",
    )

    assert isinstance(intent, DesignIntent)
    assert len(intent.dominant_heat_mechanisms) > 0
    assert "roadway" in " ".join(intent.preservation_requirements).lower()
    assert any("planter" in r.lower() for r in intent.rejected_interventions)
    assert intent.shade_priority >= 0.70


def test_planner_integrates_design_intent_and_rejects_violating_candidates():
    """Verify generate_spatial_plan includes design_intent and rejected_candidates."""
    w, h = 800, 600
    img = Image.new("RGB", (w, h), (140, 140, 140))
    seg_result = {
        "surfaces": {"road": 30.0, "pavement": 25.0, "wall": 30.0, "sky": 15.0},
        "raw_preds": np.zeros((h, w), dtype=np.int32),
        "masks": [],
    }
    seg_result["raw_preds"][:, :250] = 3   # sidewalk
    seg_result["raw_preds"][:, 250:600] = 6 # road
    seg_result["raw_preds"][:, 600:] = 0   # wall

    plan = generate_spatial_plan(
        image=img,
        surfaces={"road": 30.0, "pavement": 25.0, "wall": 30.0, "sky": 15.0},
        seg_result=seg_result,
        design_profile="pedestrian_first",
    )

    assert plan.design_intent is not None
    assert isinstance(plan.design_intent, DesignIntent)
    assert len(plan.interventions) > 0

    # Ensure tree interventions have explicit anchor geometry
    tree_specs = [iv for iv in plan.interventions if iv.type == "tree_canopy"]
    if tree_specs:
        tree = tree_specs[0]
        assert tree.anchors is not None
        assert len(tree.anchors) >= 1
        anc = tree.anchors[0]
        assert "x" in anc and "y" in anc
        assert "canopy_radius" in anc
        assert "planting_pit" in anc


def test_tree_anchor_spatial_layout_clearance():
    """Verify compute_tree_row_layout places anchors safely within the sidewalk verge away from curb and wall."""
    w, h = 800, 600
    sidewalk_mask = np.zeros((h, w), dtype=bool)
    road_mask = np.zeros((h, w), dtype=bool)
    wall_mask = np.zeros((h, w), dtype=bool)
    protected_mask = np.zeros((h, w), dtype=bool)

    # Left sidewalk: x from 100 to 300, y from 300 to 600
    # Storefront wall: x from 0 to 100
    # Road: x from 300 to 700
    wall_mask[:, :100] = True
    sidewalk_mask[300:, 100:300] = True
    road_mask[300:, 300:700] = True

    layout = compute_tree_row_layout(
        width=w,
        height=h,
        sidewalk_mask=sidewalk_mask,
        road_mask=road_mask,
        protected_mask=protected_mask,
        wall_mask=wall_mask,
    )

    anchors = layout.get("anchors", [])
    assert len(anchors) > 0, "Should generate tree anchors on available sidewalk"

    for anc in anchors:
        ax = anc["x"]
        ay = anc["y"]
        # Ground anchor must be strictly on sidewalk (100 < x < 300, y >= 300)
        assert 100 <= ax <= 300, f"Anchor x={ax} outside sidewalk zone [100, 300]"
        assert ay >= 300, f"Anchor y={ay} outside sidewalk zone y>=300"
        # Must maintain safe clearance from road (x=300) and wall (x=100)
        assert ax >= 120, f"Anchor x={ax} too close to storefront wall"
        assert ax <= 280, f"Anchor x={ax} too close to road curb"


def test_build_intervention_prompt_injects_hard_negatives():
    """Verify build_intervention_prompt includes hard negative rules from DesignIntent."""
    intent = DesignIntent(
        primary_objective="Pedestrian shade",
        secondary_objectives=[],
        dominant_heat_mechanisms=["solar_exposure"],
        target_zones=["left_sidewalk"],
        pedestrian_priority=0.9,
        shade_priority=0.9,
        vegetation_priority=0.8,
        permeability_priority=0.5,
        albedo_priority=0.5,
        preservation_requirements=[],
        hard_constraints=[],
        acceptable_interventions=["trees"],
        rejected_interventions=["planter box"],
        desired_spatial_pattern="linear corridor",
        desired_continuity=0.8,
        visual_character="mature trees",
        hard_negative_rules=[
            "planter box",
            "giant planter",
            "raised planter",
            "rectangular flower box",
            "concrete barrier",
        ],
    )

    prompt, neg_prompt = build_intervention_prompt("tree_canopy", design_intent=intent)
    assert "mature roadside shade tree" in prompt
    assert "planter box" in neg_prompt
    assert "giant planter" in neg_prompt
    assert "concrete barrier" in neg_prompt


def test_design_critic_passes_realistic_tree():
    """Verify evaluate_design_critique passes when a realistic green tree with texture is generated."""
    w, h = 512, 512
    orig = Image.new("RGB", (w, h), (140, 140, 140))
    gen = orig.copy()
    draw = ImageDraw.Draw(gen)

    # Draw leafy green canopy in upper half
    draw.ellipse([150, 80, 360, 280], fill=(34, 139, 34))
    # Add texture/leaves
    for i in range(160, 350, 15):
        for j in range(90, 270, 15):
            draw.point((i, j), fill=(20, 100, 20))
    # Draw brown trunk in lower half
    draw.rectangle([240, 270, 270, 460], fill=(101, 67, 33))

    mask = Image.new("L", (w, h), 0)
    d_mask = ImageDraw.Draw(mask)
    d_mask.ellipse([150, 80, 360, 280], fill=255)
    d_mask.rectangle([240, 270, 270, 460], fill=255)

    critique = evaluate_design_critique(
        original_crop=orig,
        generated_crop=gen,
        mask=mask,
        intervention_type="tree_canopy",
    )

    assert critique.intervention_presence_score >= 0.50
    assert critique.preservation_score >= 0.80
    assert critique.unauthorized_change_score >= 0.70
    assert critique.passed is True


def test_design_critic_flags_planter_box():
    """Verify evaluate_design_critique detects unauthorized rectangular planter box structures."""
    w, h = 512, 512
    orig = Image.new("RGB", (w, h), (180, 180, 180))
    gen = orig.copy()
    draw = ImageDraw.Draw(gen)

    # Draw a solid rectangular planter box in lower 35% of crop
    draw.rectangle([80, 360, 430, 480], fill=(60, 60, 60), outline=(255, 255, 255), width=3)
    # Add multiple horizontal dividing ridges typical of concrete planter boxes
    for y_line in [380, 400, 420, 440, 460]:
        draw.line([(80, y_line), (430, y_line)], fill=(255, 255, 255), width=2)

    mask = Image.new("L", (w, h), 0)
    d_mask = ImageDraw.Draw(mask)
    d_mask.rectangle([80, 360, 430, 480], fill=255)

    critique = evaluate_design_critique(
        original_crop=orig,
        generated_crop=gen,
        mask=mask,
        intervention_type="tree_canopy",
    )

    assert "planter_box_detected" in critique.failure_reasons
    assert critique.unauthorized_change_score < 0.85
    assert critique.retry_recommendation is not None
    assert "planter" in critique.retry_recommendation.lower()


def test_design_critic_protects_preserved_objects():
    """Verify evaluate_design_critique rejects modifications to protected pixels."""
    w, h = 512, 512
    orig = Image.new("RGB", (w, h), (150, 150, 150))
    gen = Image.new("RGB", (w, h), (255, 0, 0))  # radically altered

    mask = Image.new("L", (w, h), 255)
    protected_mask = np.ones((h, w), dtype=bool)  # Entire region is protected

    critique = evaluate_design_critique(
        original_crop=orig,
        generated_crop=gen,
        mask=mask,
        intervention_type="tree_canopy",
        protected_mask_crop=protected_mask,
    )

    assert "protected_region_violation" in critique.failure_reasons
    assert critique.preservation_score < 0.70
    assert critique.passed is False
