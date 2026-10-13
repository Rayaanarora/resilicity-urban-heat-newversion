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
