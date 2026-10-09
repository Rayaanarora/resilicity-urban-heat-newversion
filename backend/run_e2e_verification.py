"""End-to-end verification script for Design Intelligence V2.

Runs the complete autonomous pipeline on urban_street_before.png:
1. Perception: SegFormer semantic segmentation & monocular depth.
2. Heat Diagnosis: Heat drivers & spatial priorities.
3. Design Strategist: Synthesize DesignIntent answering A..I.
4. Spatial Designer: Anchor layout respecting clearances.
5. Controlled Generation: Tree-by-tree inpainting with Depth ControlNet.
6. Design Critic: Quantitative critique & verification.
7. Verification of all debug artifacts (01..07, tree_01..*, final_*).
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("resilicity.e2e_test")

from segmentation.model import SegFormerEngine
from scene_understanding import analyze_scene
from planner import generate_spatial_plan
from image_generation.multi_pass import run_autonomous_multi_pass_redesign
from image_generation.sd15_controlnet_provider import LocalSD15ControlNetInpaintingProvider


async def main():
    img_path = Path(__file__).resolve().parent.parent / "gen-ai" / "public" / "samples" / "urban_street_before.png"
    if not img_path.exists():
        logger.error("Sample image not found at %s", img_path)
        return

    logger.info("Loading input image from %s", img_path)
    img = Image.open(img_path).convert("RGB")
    logger.info("Image size: %dx%d", img.width, img.height)

    out_dir = Path(__file__).resolve().parent / "data" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Perception (SegFormer)
    logger.info("Running SegFormer segmentation...")
    seg_engine = SegFormerEngine.get_instance()
    seg_res = seg_engine.segment_image(img)
    surfaces = seg_res.get("surfaces", {})
    if not surfaces:
        surfaces = {m["className"]: m["areaPercentage"] for m in seg_res.get("masks", [])}
        seg_res["surfaces"] = surfaces
    seg_engine.offload_to_cpu()
    logger.info("Detected surfaces: %s", surfaces)

    # 2. Scene Understanding
    logger.info("Analyzing 3D spatial scene understanding...")
    scene = analyze_scene(img, seg_res)

    # 3. Autonomous Design Strategist & Planner
    logger.info("Generating autonomous spatial design plan with Design Intent...")
    plan = generate_spatial_plan(
        image=img,
        surfaces=surfaces,
        seg_result=seg_res,
        design_profile="pedestrian_first",
    )
    logger.info("Selected strategy: %s", plan.selected_strategy)
    logger.info("Design Intent primary objective: %s", plan.design_intent.primary_objective if plan.design_intent else "N/A")
    logger.info("Interventions count: %d", len(plan.interventions))
    for iv in plan.interventions:
        anchors = getattr(iv, "anchors", []) or []
        logger.info("  Intervention: %s (anchors=%d)", iv.type, len(anchors))

    # 4. Multi-Pass Controlled Generative Inpainting
    logger.info("Initializing SD 1.5 + Depth ControlNet provider...")
    provider = LocalSD15ControlNetInpaintingProvider.get_instance()
    logger.info("Provider loaded on GPU: %s", provider.gpu_name)

    t0 = time.time()
    final_img, val_report, multi_pass_meta = await run_autonomous_multi_pass_redesign(
        image=img,
        plan=plan,
        seg_result=seg_res,
        sdxl=provider,
        scene_understanding=scene,
        quality_tier="fast",
        save_debug=True,
        debug_dir=out_dir,
        base_seed=42,
    )
    elapsed = time.time() - t0
    logger.info("Redesign finished in %.2fs", elapsed)

    # 5. Check Artifact Suite
    required_artifacts = [
        "01_original.png",
        "02_heat_priority.png",
        "03_heat_drivers.json",
        "04_design_intent.json",
        "05_candidate_interventions.json",
        "06_selected_strategy.json",
        "07_spatial_layout.json",
        "final_before.png",
        "final_after.png",
        "final_design_intent.json",
        "final_design_critique.json",
    ]

    all_exist = True
    for fname in required_artifacts:
        fpath = out_dir / fname
        if fpath.exists():
            logger.info("✓ Found artifact: %s (%d bytes)", fname, fpath.stat().st_size)
        else:
            logger.error("✗ Missing artifact: %s", fname)
            all_exist = False

    critique_data = multi_pass_meta.get("design_critique", {})
    logger.info("=" * 60)
    logger.info("FINAL DESIGN CRITIQUE VERDICT:")
    logger.info("  Passed: %s", critique_data.get("passed"))
    logger.info("  Overall Score: %.2f", critique_data.get("overall_score", 0.0))
    logger.info("  Presence Score: %.2f", critique_data.get("intervention_presence_score", 0.0))
    logger.info("  Spatial Compliance: %.2f", critique_data.get("spatial_compliance_score", 0.0))
    logger.info("  Preservation Score: %.2f", critique_data.get("preservation_score", 0.0))
    logger.info("  Unauthorized Change Score: %.2f", critique_data.get("unauthorized_change_score", 0.0))
    logger.info("  Failure Reasons: %s", critique_data.get("failure_reasons", []))
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
