"""Autonomous End-to-End Visual Redesign Validation Script."""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("resilicity.final_test")

from image_generation import (
    LocalSD15InpaintingProvider,
    LocalSDXLInpaintingProvider,
    run_autonomous_multi_pass_redesign,
    validate_image_output,
)
from planner import generate_spatial_plan
from scene_understanding import analyze_scene
from segmentation import SegFormerEngine

DEBUG_DIR = Path(__file__).resolve().parent / "data" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATH = Path(__file__).resolve().parents[1] / "gen-ai" / "public" / "samples" / "urban_street_before.png"


async def main():
    t0 = time.time()
    logger.info("Loading street image from %s", IMAGE_PATH)
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Source image not found: {IMAGE_PATH}")

    pil_img = Image.open(IMAGE_PATH).convert("RGB")
    logger.info("Image loaded: size=%s", pil_img.size)

    # 1. Perception Layer (SegFormer semantic segmentation)
    logger.info("Running SegFormer semantic segmentation...")
    seg_engine = SegFormerEngine.get_instance()
    seg_res = seg_engine.segment_image(pil_img, include_raw_arrays=True)
    seg_engine.offload_to_cpu()

    surfaces = {}
    for m in seg_res.get("masks", []):
        surfaces[m["className"]] = m["areaPercentage"]
    logger.info("Detected surfaces: %s", surfaces)

    # 2. Scene Understanding
    logger.info("Executing deep spatial scene understanding...")
    scene = analyze_scene(pil_img, seg_res)
    logger.info(
        "Scene understanding: canyon_strength=%.2f, sun_direction=%s, left_sw_area=%.1f%%, right_sw_area=%.1f%%, road_area=%.1f%%",
        scene.street_geometry.street_canyon_strength,
        scene.sun_direction_proxy,
        scene.left_sidewalk.relative_area_pct,
        scene.right_sidewalk.relative_area_pct,
        scene.roadway.relative_area_pct,
    )

    # 3. Autonomous Spatial Urban Resilience Planner
    logger.info("Generating autonomous spatial urban design plan...")
    plan = generate_spatial_plan(pil_img, surfaces, seg_result=seg_res, design_profile="balanced")
    logger.info("Selected interventions: %s", [f"{iv.type} (p={iv.priority}, zone={iv.target_zone})" for iv in plan.interventions])

    # 4. Local SD 1.5 Inpainting Engine
    logger.info("Initializing Local SD 1.5 Inpainting Engine on CUDA...")
    sd15 = LocalSD15InpaintingProvider.get_instance()
    status = sd15.get_status()
    logger.info("Generator status: GPU=%s, VRAM=%.2f GB, Model=%s", status["gpu_name"], status["vram_gb"], status["model_path"])

    # 5. Run Complete 5-Stage Urban Redesign Pipeline
    logger.info("Executing 5-Stage Autonomous Urban Redesign Pipeline...")
    edited_img, report, meta = await run_autonomous_multi_pass_redesign(
        image=pil_img,
        plan=plan,
        seg_result=seg_res,
        sdxl=sd15,
        scene_understanding=scene,
        quality_tier="fast",
        save_debug=True,
        debug_dir=DEBUG_DIR,
        base_seed=42,
    )

    # 6. Ensure all Part 32 debug artifacts exist
    req_artifacts = [
        "00_original.png",
        "01_segmentation.png",
        "02_depth.png",
        "03_spatial_heat_priority.png",
        "04_protected_objects.png",
        "05_scene_layout.png",
        "06_layout_plan.json",
        "07_tree_layout.png",
        "08_tree_draft.png",
        "09_tree_crop_input.png",
        "10_tree_crop_mask.png",
        "11_tree_crop_output.png",
        "12_shade_layout.png",
        "13_shade_draft.png",
        "14_shade_crop_input.png",
        "15_shade_crop_output.png",
        "16_road_draft.png",
        "17_sidewalk_draft.png",
        "18_harmonized_scene.png",
        "19_protected_recomposite.png",
        "20_final_redesign.png",
        "validation.json",
        "generation_trace.json",
    ]

    logger.info("=" * 60)
    logger.info("PART 32 DEBUG ARTIFACT SUITE VERIFICATION:")
    all_present = True
    for art in req_artifacts:
        p = DEBUG_DIR / art
        exists = p.exists()
        if not exists:
            all_present = False
        size_kb = (p.stat().st_size / 1024) if exists else 0
        logger.info("  %s: %s (%.1f KB)", art, "EXISTS" if exists else "MISSING", size_kb)
    logger.info("=" * 60)

    logger.info(
        "PIPELINE COMPLETED in %.2fs: is_valid=%s, diff_mean=%.2f, masked_diff=%.2f, pct_changed=%.1f%%, all_artifacts=%s",
        time.time() - t0,
        report.is_valid if report else False,
        report.diff_mean if report and report.diff_mean else 0.0,
        report.masked_diff if report and report.masked_diff else 0.0,
        report.pct_changed if report and report.pct_changed else 0.0,
        all_present,
    )


if __name__ == "__main__":
    asyncio.run(main())
