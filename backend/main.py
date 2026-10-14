"""ResiliCity backend: serves perception, spatial urban planning, thermal impact,
and FLUX.1 Kontext generative redesign of street photos for urban heat resilience.

Run:  uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import asyncio
import io
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging
import random
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger("resilicity.main")

HERE = Path(__file__).parent

# Load .env server-side safely without exposing to client
env_path = HERE / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from image_generation import (
    GenerationResult,
    RedesignEngine,
    build_kontext_redesign_prompt,
    build_kontext_refinement_prompt,
    illustrated_zones,
    parse_refinement_intent,
    UnifiedRedesignResponse,
    RefinementIntent,
    RefinementResponse,
    SceneAnalysis,
    SpatialDesignPlan,
    VisualizationOutput,
    ValidationReport,
)
from inpainting import ResilientInpainter, encode_image_to_base64
from legacy_heat import (  # noqa: F401  (re-exported: tests and clients import these from main)
    FRAC,
    RANGES,
    bundle,
    card,
    compute_thermal_impact,
    model,
    predict,
    to_fractions,
)
from planner import (
    generate_spatial_plan,
    analyze_scene_heuristics,
    planner_available,
    router as planner_router,
)
from analysis_service import run_site_analysis
from hotspots import router as hotspots_router
from projects import router as projects_router
from optimizer import router as portfolio_router
from scene_understanding import analyze_scene, render_heat_priority_colormap
from segmentation import SegFormerEngine

inpainter = ResilientInpainter()

# One edit call per redesign through the provider chain in IMAGE_PROVIDERS
# (default: free Hugging Face Kontext Space, then fal.ai), with a persistent disk cache.
redesign_engine = RedesignEngine()


DEBUG_DIR = HERE / "data" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def save_segmentation_visualization(image: Image.Image, seg_result: Dict[str, Any], out_path: Path) -> None:
    """Render colored semantic segmentation overlay on top of original image for visual debugging."""
    from PIL import ImageDraw
    w, h = image.size
    overlay = image.copy().convert("RGBA")
    draw_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_layer)

    COLOR_MAP = {
        "road": (220, 38, 38, 120),       # Red
        "pavement": (234, 88, 12, 130),   # Orange
        "wall": (217, 119, 6, 120),       # Amber
        "roof": (147, 51, 234, 130),      # Purple
        "vegetation": (22, 163, 74, 140), # Green
        "sky": (2, 132, 199, 100),        # Sky blue
        "other": (148, 163, 184, 90),     # Slate
    }

    for m in seg_result.get("masks", []):
        cls_name = m.get("className", "other")
        color = COLOR_MAP.get(cls_name, (148, 163, 184, 90))
        for poly in m.get("polygons", []):
            if len(poly) >= 3:
                pixel_pts = [
                    (int(round(pt[0] / 100.0 * w)), int(round(pt[1] / 100.0 * h)))
                    for pt in poly
                ]
                draw.polygon(pixel_pts, fill=color)

    combined = Image.alpha_composite(overlay, draw_layer)
    combined.convert("RGB").save(out_path)



# Initialize SegFormer singleton
seg_engine = SegFormerEngine.get_instance()

app = FastAPI(title="ResiliCity Generative AI Platform", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(planner_router)
app.include_router(hotspots_router)
app.include_router(projects_router)
app.include_router(portfolio_router)

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/health")
def health():
    seg_info = seg_engine.get_status()
    engine_status = redesign_engine.status()
    return {
        "ok": True,
        "generative_provider": ",".join(redesign_engine.provider_names),
        "image_generation": engine_status,
        "heat_model_available": bundle is not None and "model" in bundle,
        "segmenter_available": seg_info["available"],
        "segmenter_model": seg_info["model_name"],
        "segmenter_device": seg_info["device"],
        "segmenter_source": seg_info["source"],
        "planner_available": planner_available(),
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
    }


@app.post("/api/v1/analyze-and-redesign")
async def analyze_and_redesign(
    image: UploadFile = File(...),
    design_profile: Optional[str] = Form(None),
    requested_interventions: Optional[str] = Form(None),
    quality_tier: str = Form("fast"),
    refinement_prompt: Optional[str] = Form(None),
    road_width_m: Optional[float] = Form(None),
    climate_zone: Optional[str] = Form(None),
    budget_lakh: Optional[float] = Form(None),
):
    """End-to-end Autonomous Generative AI urban resilience pipeline.
    
    1. Scene Analysis (SegFormer semantic perception)
    2. Autonomous Spatial Urban Design Planning (no user toggles required)
    3. Spatial Inpainting Mask Construction (pedestrian curb envelopes, roadway, roofs)
    4. Whole-scene FLUX.1 Kontext redesign (single edit call, disk-cached)
    5. Visual validation (output must visibly differ from the original)
    6. Estimated Microclimate Thermal Impact Computation
    """
    t0 = time.time()
    try:
        content = await image.read()
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(413, "Image file too large (max 20 MB)")
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid image file: {e}")

    # 1-3. Segmentation, scene understanding and the gated plan (shared with project creation)
    site = run_site_analysis(pil_img, design_profile, road_width_m, climate_zone, budget_lakh)
    surfaces, seg_res, scene, plan, analysis = site.surfaces, site.seg_result, site.scene, site.plan, site.analysis

    # 4. Generative redesign: one FLUX.1 Kontext edit of the whole photo, driven by the plan.
    #    Results are cached on disk by the engine, so re-running the same photo costs nothing.
    plan_dict = plan.dict()
    prompt = build_kontext_redesign_prompt(plan, scene.to_dict().get("protected_objects_summary"))
    if refinement_prompt and prompt:
        prompt = f"{prompt} Additionally: {refinement_prompt.strip()}"

    if prompt:
        result = await redesign_engine.generate(pil_img, content, prompt, quality_tier=quality_tier)
    else:
        # Nothing buildable was selected: do not spend generation quota on an empty edit.
        result = GenerationResult(
            None, "none", "",
            error="No buildable interventions were selected for this street; see the rejected candidates for reasons.",
        )
    elapsed_ms = int((time.time() - t0) * 1000)
    thermal = site.thermal

    # 5. Debug artifact bundle
    try:
        pil_img.save(DEBUG_DIR / "original.png")
        save_segmentation_visualization(pil_img, seg_res, DEBUG_DIR / "segmentation.png")
        if scene.heat_priority_map is not None:
            render_heat_priority_colormap(scene.heat_priority_map).save(DEBUG_DIR / "spatial_heat_priority.png")
        if result.image is not None:
            result.image.save(DEBUG_DIR / "final_redesign.png")
        (DEBUG_DIR / "redesign_prompt.txt").write_text(prompt, encoding="utf-8")
        (DEBUG_DIR / "scene_understanding.json").write_text(json.dumps(scene.to_dict(), indent=2), encoding="utf-8")
        (DEBUG_DIR / "design_plan.json").write_text(json.dumps(plan_dict, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not export debug artifacts: %s", e)

    if result.image is not None:
        vis_output = {
            "status": "ready",
            "cached": result.cached,
            "image_url": encode_image_to_base64(result.image, quality=92),
            "width": result.image.width,
            "height": result.image.height,
            "provider": result.provider,
            "model": result.model,
            "quality_tier": quality_tier,
            "generation_time_ms": elapsed_ms,
            "refinement_count": 0,
            "error_message": None,
            # Kontext draws only the top-priority items; the full plan is in implementation_plan.
            "illustrated_interventions": list(illustrated_zones(plan)),
        }
        validation = result.validation.dict() if result.validation else {
            "is_valid": True,
            "aspect_ratio_preserved": True,
            "dimensions_valid": True,
            "non_blank_verified": True,
            "checks_passed": ["Served from redesign cache"],
            "warnings": [],
        }
    else:
        logger.error("Redesign generation failed: %s", result.error)
        vis_output = {
            "status": "unavailable",
            "image_url": None,
            "width": pil_img.width,
            "height": pil_img.height,
            "provider": result.provider,
            "model": result.model,
            "quality_tier": quality_tier,
            "generation_time_ms": elapsed_ms,
            "refinement_count": 0,
            "error_message": result.error or "Image generation failed.",
        }
        validation = {
            "is_valid": False,
            "aspect_ratio_preserved": True,
            "dimensions_valid": False,
            "non_blank_verified": False,
            "checks_passed": [],
            "warnings": [result.error or "Image generation failed"],
        }

    return {
        "scene_analysis": analysis.dict(),
        "scene_understanding": scene.to_dict(),
        "design_plan": plan_dict,
        "visualization": vis_output,
        "thermal_impact": thermal,
        "validation": validation,
        "design_intent": plan.design_intent.dict() if getattr(plan, "design_intent", None) else None,
        "design_critique": plan.design_critique.dict() if getattr(plan, "design_critique", None) else None,
    }


@app.post("/api/v1/refine-design")
async def refine_design(
    image: UploadFile = File(...),
    instruction: Optional[str] = Form(None),
    refinement_prompt: Optional[str] = Form(None),
    quality_tier: str = Form("fast"),
    context: Optional[str] = Form(None),
    refinement_history: Optional[str] = Form(None),
):
    """Refine a previously generated redesign with open-ended natural language guidance.
    
    Transforms the current generated design according to the user's arbitrary instruction,
    interprets structured intent, and applies a FLUX.1 Kontext edit that changes only
    what was asked while preserving architectural identity and camera perspective.
    """
    t0 = time.time()
    instr_text = (instruction or refinement_prompt or "").strip()
    if not instr_text:
        raise HTTPException(400, "Refinement instruction cannot be empty.")

    try:
        content = await image.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    # Structured intent is returned to the UI; the edit itself is a short Kontext instruction.
    intent = parse_refinement_intent(instr_text, context=context)
    intent_dict = intent.dict() if hasattr(intent, "dict") else intent.model_dump()
    prompt = build_kontext_refinement_prompt(instr_text)

    result = await redesign_engine.generate(
        pil_img, content, prompt, quality_tier=quality_tier, require_visible_change=False
    )
    elapsed_ms = int((time.time() - t0) * 1000)

    response_data = {
        "status": "ready" if result.image is not None else "unavailable",
        "cached": result.cached,
        "image_url": encode_image_to_base64(result.image, quality=92) if result.image is not None else None,
        "width": result.image.width if result.image is not None else pil_img.width,
        "height": result.image.height if result.image is not None else pil_img.height,
        "instruction": instr_text,
        "provider": result.provider,
        "model": result.model,
        "quality_tier": quality_tier,
        "generation_time_ms": elapsed_ms,
        "intent": intent_dict,
        # On failure the frontend keeps showing the current image.
        "error_message": None if result.image is not None else (result.error or "Refinement generation unavailable."),
    }
    return response_data


# ---------------------------------------------------------------------------
# Backwards-compatible legacy endpoints
# ---------------------------------------------------------------------------
class HeatReq(BaseModel):
    surfaces: dict[str, float]
    shifts: dict[str, float] = {}


@app.post("/api/v1/heat")
def heat(req: HeatReq):
    before = to_fractions(req.surfaces)
    after = {k: max(0.0, before[k] + req.shifts.get(k, 0.0)) for k in FRAC}
    s = sum(after.values())
    after = {k: v / s for k, v in after.items()}

    out_of_range = []
    if RANGES:
        out_of_range = [k for k in FRAC if k in RANGES and not RANGES[k][0] - 1e-9 <= before[k] <= RANGES[k][1] + 1e-9]

    p0, p1 = predict(before), predict(after)
    delta = p1 - p0
    rmse = card["spatial_cv"]["RMSE"]
    expected_cooling = (req.shifts.get("f_tree", 0) + req.shifts.get("f_shrub", 0) + req.shifts.get("f_grass", 0)
                        - req.shifts.get("f_built", 0)) > 0
    wrong_sign = expected_cooling and delta > 0.01
    noise_threshold = 0.05
    within_noise = abs(delta) < noise_threshold
    return {
        "lstBeforeC": p0,
        "lstAfterC": p1,
        "deltaC": delta,
        "rmseC": rmse,
        "outOfRange": out_of_range,
        "wrongSign": bool(wrong_sign),
        "withinNoise": bool(within_noise),
        "reliable": not (wrong_sign or within_noise),
    }


@app.post("/api/v1/segment")
async def segment(file: UploadFile = File(...)):
    if not seg_engine.is_loaded:
        raise HTTPException(503, f"Segmentation model unavailable: {seg_engine.load_error or 'Model not initialized'}")
    try:
        content = await file.read()
        if len(content) > 15 * 1024 * 1024:
            raise HTTPException(413, "Image file too large (max 15 MB)")
        img = Image.open(io.BytesIO(content)).convert("RGB")
        res = seg_engine.segment_image(img, include_raw_arrays=False)
        seg_engine.offload_to_cpu()
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Segmentation error: {e}")


@app.post("/api/v1/inpaint", deprecated=True)
async def inpaint_scene(
    file: UploadFile = File(...),
    interventions: str = Form("[]"),
    polygons: str | None = Form(None),
):
    """[DEPRECATED] Legacy procedural inpainting endpoint.
    
    Warning: Retained strictly for legacy test compatibility.
    Primary autonomous generation route is /api/v1/analyze-and-redesign.
    """
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    try:
        interventions_list = json.loads(interventions) if interventions else []
    except Exception:
        interventions_list = []

    polygons_by_class: dict[str, list[list[tuple[float, float]]]] = {}
    if polygons:
        try:
            polygons_by_class = json.loads(polygons)
        except Exception:
            pass

    if not polygons_by_class and seg_engine.is_loaded:
        try:
            seg_res = seg_engine.segment_image(img)
            for m in seg_res.get("masks", []):
                cls_name = m.get("className") or m.get("label")
                if cls_name:
                    polygons_by_class.setdefault(cls_name, []).extend(m.get("polygons", []))
        except Exception as e:
            warnings.warn(f"Segmentation failed for fallback: {e}")

    try:
        inpainted_img = inpainter.inpaint(img, polygons_by_class, interventions_list)
        orig_arr = np.array(img).astype(np.float32)
        inpaint_arr = np.array(inpainted_img).astype(np.float32)
        diff_mean = float(np.abs(inpaint_arr - orig_arr).mean())

        data_url = encode_image_to_base64(inpainted_img)
        return {
            "ok": True,
            "imageUrl": data_url,
            "interventions_count": len(interventions_list),
            "diff_mean": diff_mean,
            "polygons_detected": list(polygons_by_class.keys()),
        }
    except Exception as e:
        raise HTTPException(500, f"Inpainting generation error: {e}")
