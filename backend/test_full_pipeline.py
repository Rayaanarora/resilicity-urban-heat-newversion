"""End-to-End Validation Script for the Spatially Intelligent Autonomous Redesign System.

Tests the full pipeline on real street photographs:
1. SegFormer semantic perception (surface classes + protected object layer)
2. Monocular depth estimation (Depth-Anything-V2-Small / perspective fallback)
3. Classical CV geometry (horizon, vanishing point, sidewalk separation, canyon H/W proxy, SVF proxy)
4. Solar & shade reasoning (direct sunlight vs shaded zones)
5. Spatial Heat Priority Map (continuous prioritization index)
6. Autonomous Spatial Planner (multi-factor utility optimization, >=3 spatial evidence signals)
7. Autonomous Multi-Pass SDXL Inpainting (sequential passes, cumulative mask protection, per-pass prompts)
8. Multi-stage validation and debug export
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure backend root is on sys.path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from segmentation import SegFormerEngine
from scene_understanding import analyze_scene
from planner import generate_spatial_plan
from image_generation import (
    LocalSDXLInpaintingProvider,
    run_autonomous_multi_pass_redesign,
    validate_image_output,
)

TEST_IMG_PATH = HERE.parent / "gen-ai" / "public" / "samples" / "sample_1_dense_urban.jpg"
DEBUG_DIR = HERE / "data" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


async def main():
    print("=" * 70)
    print("RESILICITY: TESTING AUTONOMOUS SPATIAL MULTI-PASS PIPELINE")
    print("=" * 70)

    if not TEST_IMG_PATH.exists():
        print(f"Error: Test image not found at {TEST_IMG_PATH}")
        return

    orig_image = Image.open(TEST_IMG_PATH).convert("RGB")
    w, h = orig_image.size
    print(f"Input image loaded: {TEST_IMG_PATH.name} ({w}x{h})")

    # Step 1: Perception Layer
    print("\n--- STEP 1: SEGFORMER PERCEPTION (LAYER 1 + LAYER 2) ---")
    t0 = time.time()
    seg_engine = SegFormerEngine.get_instance()
    seg_result = seg_engine.segment_image(orig_image)
    seg_engine.offload_to_cpu()
    surfaces = {m["className"]: m["areaPercentage"] for m in seg_result.get("masks", [])}
    print(f"Perception completed in {time.time() - t0:.2f}s")
    print("Clean Surface Classes:", surfaces)
    print("Protected Objects Detected:", [f"{p['id']} ({p['percentage']}%)" for p in seg_result.get("protected_objects", [])])

    # Step 2: Deep Spatial Scene Understanding
    print("\n--- STEP 2: DEEP SPATIAL SCENE UNDERSTANDING ---")
    t1 = time.time()
    scene = analyze_scene(orig_image, seg_result)
    geom = scene.street_geometry
    ls = scene.left_sidewalk
    rs = scene.right_sidewalk
    road = scene.roadway

    print(f"Scene understanding derived in {time.time() - t1:.2f}s")
    print(f"  Street Canyon Strength: {geom.street_canyon_strength:.2f}")
    print(f"  Height-to-Width (H/W):   {geom.height_to_width_proxy:.2f}")
    print(f"  Sky View Factor (SVF):   {geom.sky_visibility_proxy:.2f}")
    print(f"  Vanishing Point:         {geom.vanishing_point}")
    print(f"  Visual Solar Exposure:   {scene.solar_exposure_proxy:.2f}")
    print(f"  Existing Shade Proxy:    {scene.existing_shade_proxy:.2f}")
    print(f"  Left Sidewalk:  width={ls.available_width_proxy:.2f}, solar={ls.solar_exposure:.2f}, tree_feas={ls.tree_feasibility:.2f}")
    print(f"  Right Sidewalk: width={rs.available_width_proxy:.2f}, solar={rs.solar_exposure:.2f}, tree_feas={rs.tree_feasibility:.2f}")
    print(f"  Roadway:        width={road.road_width_proxy:.2f}, solar={road.solar_exposure:.2f}, cool_feas={road.cool_pavement_feasibility:.2f}")
    print(f"  Heat Priority:  {scene.heat_priority_summary}")

    # Step 3: Autonomous Spatial Planner
    print("\n--- STEP 3: AUTONOMOUS SPATIAL PLANNER ---")
    plan = generate_spatial_plan(orig_image, surfaces, seg_result=seg_result, design_profile="balanced")
    print(f"Planner Source: {plan.planner_source}")
    print(f"Site Summary: {plan.site_summary}")
    print(f"Total Selected Interventions: {len(plan.interventions)}")
    for iv in plan.interventions:
        print(f"  - [{iv.priority}] {iv.title} on {iv.target_zone or iv.target_region} (Utility={iv.utility_score:.2f}, Feasibility={iv.feasibility:.2f})")
        print(f"    Evidence ({len(iv.evidence)} signals):")
        for ev in iv.evidence:
            print(f"      * {ev}")

    # Step 4: Autonomous Multi-Pass SDXL Inpainting
    print("\n--- STEP 4: AUTONOMOUS MULTI-PASS SDXL INPAINTING ---")
    sdxl = LocalSDXLInpaintingProvider.get_instance()
    t2 = time.time()
    final_img, report, meta = await run_autonomous_multi_pass_redesign(
        image=orig_image,
        plan=plan,
        seg_result=seg_result,
        sdxl=sdxl,
        scene_understanding=scene,
        quality_tier="fast",
        save_debug=True,
        debug_dir=DEBUG_DIR,
        base_seed=42,
    )
    gen_time = time.time() - t2
    print(f"Multi-pass generation finished in {gen_time:.2f}s")
    print("Multi-pass Metadata:", json.dumps(meta, indent=2))

    if final_img is not None:
        out_path = DEBUG_DIR / "final_redesign.png"
        final_img.save(out_path)
        print(f"\nFinal redesign saved to {out_path}")

        # Compute numerical comparison
        orig_arr = np.array(orig_image.resize(final_img.size)).astype(np.float32)
        gen_arr = np.array(final_img).astype(np.float32)
        abs_diff = np.abs(gen_arr - orig_arr)
        mean_diff = float(np.mean(abs_diff))
        pct_changed = float(np.count_nonzero(np.max(abs_diff, axis=2) > 15.0) / (final_img.width * final_img.height) * 100.0)

        print("\n" + "=" * 50)
        print("NUMERICAL VALIDATION SUMMARY:")
        print(f"  Global Image Mean Difference: {mean_diff:.2f} / 255")
        print(f"  Percentage Pixels Changed:    {pct_changed:.2f}%")
        print(f"  Cumulative Mask Coverage:     {meta.get('cumulative_mask_coverage', 0.0):.2f}%")
        print(f"  Passes Accepted:              {meta.get('passes_accepted')}/{meta.get('passes_executed')}")
        print(f"  Validation Checks Passed:     {report.checks_passed if report else []}")
        print(f"  Validation Warnings:          {report.warnings if report else []}")
        print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
