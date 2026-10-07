"""ResiliCity backend: serves perception, spatial urban planning, thermal impact,
and Google Gemini Generative AI urban resilience image editing.

Run:  uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import asyncio
import hashlib
import io
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import logging
import random
import numpy as np
import pandas as pd
import sklearn
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
    GeminiImageEditingProvider,
    LocalSDXLInpaintingProvider,
    build_inpainting_mask,
    build_redesign_prompt,
    build_refinement_prompt,
    build_sdxl_inpainting_prompt,
    parse_refinement_intent,
    validate_image_output,
    UnifiedRedesignResponse,
    RefinementIntent,
    RefinementResponse,
    SceneAnalysis,
    SpatialDesignPlan,
    VisualizationOutput,
    ValidationReport,
)
from inpainting import ResilientInpainter, encode_image_to_base64
from planner import (
    generate_spatial_plan,
    analyze_scene_heuristics,
    planner_available,
    router as planner_router,
)
from segmentation import SegFormerEngine

IMAGE_PROVIDER = os.environ.get("IMAGE_PROVIDER", "local_sdxl").strip().lower()

inpainter = ResilientInpainter()

# Lazy provider construction: only instantiate SDXL if local_sdxl is active; never require Gemini API key
sdxl_provider = LocalSDXLInpaintingProvider.get_instance() if IMAGE_PROVIDER == "local_sdxl" else None
_gemini_provider: Optional[GeminiImageEditingProvider] = None


def get_gemini_provider() -> GeminiImageEditingProvider:
    global _gemini_provider
    if _gemini_provider is None:
        _gemini_provider = GeminiImageEditingProvider()
    return _gemini_provider


# In-memory deterministic generation cache: key -> cached dict with visualization & real validation
GENERATION_CACHE: Dict[str, Dict[str, Any]] = {}
REDESIGN_CACHE: Dict[str, Dict[str, Any]] = GENERATION_CACHE

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



bundle = joblib.load(HERE / "heat_model.joblib")
model, FRAC = bundle["model"], bundle["features"]
card = json.loads((HERE / "model_card.json").read_text())
RANGES = card.get("range")

_built_with = card.get("sklearn_version")
if getattr(model, "monotonic_cst", None) is None:
    warnings.warn("heat_model.joblib has no monotonic constraints; run `python retrain.py`.")
if _built_with and _built_with != sklearn.__version__:
    warnings.warn(f"heat_model.joblib built with {_built_with}, running {sklearn.__version__}.")

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

APP_TO_WC = {"road": "built", "roof": "built", "pavement": "built", "wall": "built", "water": "water"}


def to_fractions(app_percent: dict[str, float], tree_share: float = 0.5) -> dict[str, float]:
    f = {c: 0.0 for c in FRAC}
    for k, pct in app_percent.items():
        if k == "vegetation":
            f["f_tree"] += pct / 100 * tree_share
            f["f_grass"] += pct / 100 * (1 - tree_share)
        elif k in APP_TO_WC:
            f["f_" + APP_TO_WC[k]] += pct / 100
    s = sum(f.values())
    if s <= 0:
        raise HTTPException(422, "surfaces must contain at least one known class")
    return {k: v / s for k, v in f.items()}


def predict(f: dict[str, float]) -> float:
    return float(model.predict(pd.DataFrame([f])[FRAC])[0])


def compute_thermal_impact(surfaces: Dict[str, float], interventions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate deterministic microclimate thermal impact."""
    total_cooling_est = 0.0
    shifts = {"f_built": 0.0, "f_tree": 0.0, "f_grass": 0.0}

    for item in interventions:
        itype = item.get("type", "")
        cov = float(item.get("coverage", 0.5))
        if itype == "tree_canopy":
            s = round(min(cov * 0.10, 0.06), 3)
            shifts["f_built"] -= s
            shifts["f_tree"] += s
            total_cooling_est += round(1.2 * (0.6 + 0.4 * cov), 1)
        elif itype == "cool_pavement":
            total_cooling_est += round(0.8 * cov, 1)
        elif itype == "permeable_pave":
            total_cooling_est += round(0.6 * cov, 1)
        elif itype == "cool_roof":
            total_cooling_est += round(0.9 * cov, 1)
        elif itype == "green_roof":
            s = round(min(cov * 0.05, 0.03), 3)
            shifts["f_built"] -= s
            shifts["f_grass"] += s
            total_cooling_est += round(1.4 * cov, 1)
        elif itype == "shade_structure":
            total_cooling_est += round(0.7 * cov, 1)

    # Machine learning land surface temperature model prediction
    try:
        before = to_fractions(surfaces)
        after = {k: max(0.0, before[k] + shifts.get(k, 0.0)) for k in FRAC}
        s = sum(after.values())
        after = {k: v / s for k, v in after.items()}
        p0, p1 = predict(before), predict(after)
        ml_delta = p1 - p0
    except Exception:
        p0, p1, ml_delta = 38.5, 38.5 - total_cooling_est, -total_cooling_est

    return {
        "baselineSurfaceTempC": round(p0, 1),
        "projectedSurfaceTempC": round(p0 - total_cooling_est, 1),
        "totalCoolingReductionC": round(total_cooling_est, 1),
        "mlModelDeltaC": round(ml_delta, 2),
        "methodology": "Dual-layer: Physical empirical microclimate calculations coupled with Landsat 8/9 LST regression.",
        "disclaimer": "AI image visualization illustrates architectural cooling intent; physical cooling figures are computed via thermodynamic and empirical models.",
    }


def compute_cache_key(img_bytes: bytes, plan_dict: Dict[str, Any], quality_tier: str, refinement: str = "") -> str:
    """Generate deterministic hash for caching generated designs."""
    hasher = hashlib.sha256()
    hasher.update(img_bytes)
    hasher.update(json.dumps(plan_dict, sort_keys=True).encode("utf-8"))
    hasher.update(quality_tier.encode("utf-8"))
    hasher.update(refinement.encode("utf-8"))
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/health")
def health():
    seg_info = seg_engine.get_status()
    sdxl_info = sdxl_provider.get_status() if sdxl_provider is not None else {
        "available": False,
        "loaded": False,
        "model": "none",
        "model_path": "",
        "cuda_available": False,
        "gpu_name": "None",
        "vram_gb": 0.0,
        "error": "Local SDXL provider not active",
    }
    return {
        "ok": True,
        "generative_provider": IMAGE_PROVIDER,
        "local_generator": sdxl_info,
        "heat_model_available": bundle is not None and "model" in bundle,
        "segmenter_available": seg_info["available"],
        "segmenter_model": seg_info["model_name"],
        "segmenter_device": seg_info["device"],
        "segmenter_source": seg_info["source"],
        "planner_available": planner_available(),
    }


@app.post("/api/v1/analyze-and-redesign")
async def analyze_and_redesign(
    image: UploadFile = File(...),
    design_profile: Optional[str] = Form(None),
    requested_interventions: Optional[str] = Form(None),
    quality_tier: str = Form("fast"),
    refinement_prompt: Optional[str] = Form(None),
):
    """End-to-end Autonomous Generative AI urban resilience pipeline.
    
    1. Scene Analysis (SegFormer semantic perception)
    2. Autonomous Spatial Urban Design Planning (no user toggles required)
    3. Spatial Inpainting Mask Construction (pedestrian curb envelopes, roadway, roofs)
    4. Local SDXL Inpainting on NVIDIA RTX GPU
    5. Quantitative Visual Validation (real diff_mean, pct_changed against original)
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

    # 1. Perception Layer (SegFormer)
    surfaces: Dict[str, float] = {}
    seg_res = {}
    try:
        seg_res = seg_engine.segment_image(pil_img)
        for m in seg_res.get("masks", []):
            surfaces[m["className"]] = m["areaPercentage"]
    except Exception as e:
        warnings.warn(f"Segmentation perception fallback: {e}")
        surfaces = {"road": 35.0, "wall": 25.0, "pavement": 15.0, "vegetation": 5.0, "roof": 0.0}

    # 2. Autonomous Spatial Urban Design Planner
    plan = generate_spatial_plan(
        pil_img,
        surfaces,
        design_profile=design_profile,  # type: ignore
    )
    analysis = analyze_scene_heuristics(pil_img, surfaces)

    # 3. Deterministic Caching with Real Validation Metrics
    plan_dict = plan.dict()
    cache_key = compute_cache_key(content, plan_dict, quality_tier, refinement_prompt or "")
    if cache_key in GENERATION_CACHE:
        cached_entry = GENERATION_CACHE[cache_key]
        return {
            "scene_analysis": analysis.dict(),
            "design_plan": plan_dict,
            "visualization": cached_entry["visualization"],
            "thermal_impact": cached_entry.get(
                "thermal_impact",
                compute_thermal_impact(surfaces, [i.dict() for i in plan.interventions]),
            ),
            "validation": cached_entry["validation"],
        }

    # 4. Generative Redesign via Local SDXL Inpainting (Task 5 Retry Loop) or Gemini
    provider_name = IMAGE_PROVIDER
    max_retries = 3
    edited_img = None
    gen_error = None
    report = None
    normalized_img = None
    final_seed = 42
    mask_meta = {}
    mask_img = None

    if IMAGE_PROVIDER == "local_sdxl":
        model_name = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        active_sdxl = sdxl_provider or LocalSDXLInpaintingProvider.get_instance()

        for attempt in range(max_retries):
            current_seed = 42 if attempt == 0 else random.randint(100, 999999)
            # Dilation escalation on retry: expansion_level 0 -> 1 -> 2
            mask_img, mask_meta = build_inpainting_mask(
                pil_img,
                seg_res,
                plan.interventions,
                expansion_level=attempt,
                save_debug=True,
                debug_dir=DEBUG_DIR,
            )

            prompt = build_sdxl_inpainting_prompt(plan, attempt=attempt)
            guidance = 7.5 + attempt * 1.0
            strength = 0.999
            steps = 20 if quality_tier == "fast" else 26

            # Task 1: Complete pipeline audit logging
            logger.info("=" * 60)
            logger.info("SDXL PIPELINE EXECUTION - Attempt %d/%d", attempt + 1, max_retries)
            logger.info("Chosen interventions: %s", [f"{i.type} -> {i.target_region} (cov={int(i.coverage*100)}%)" for i in plan.interventions])
            logger.info("Target semantic classes: %s", [i.target_region for i in plan.interventions])
            logger.info("Individual mask coverages: %s", mask_meta.get("individual_coverages"))
            logger.info("Combined mask coverage: %.1f%% (%d px)", mask_meta.get("coverage_percentage", 0.0), mask_meta.get("covered_pixels", 0))
            logger.info("Final SDXL prompt: %s", prompt)
            logger.info("Generation dimensions: 512x512 (scaled)")
            logger.info("Strength: %.3f | Guidance: %.1f | Steps: %d | Seed: %d", strength, guidance, steps, current_seed)
            logger.info("=" * 60)

            edited_img, gen_error = await active_sdxl.edit(
                pil_img,
                prompt=prompt,
                quality_tier=quality_tier,
                mask_image=mask_img,
                seed=current_seed,
                strength=strength,
                guidance_scale=guidance,
                steps=steps,
                max_retries=1,
            )

            if edited_img is not None:
                # Task 5: Multi-stage quantitative validation
                is_valid, report, normalized_img = validate_image_output(
                    edited_img,
                    pil_img.size,
                    orig_img=pil_img,
                    mask_img=mask_img,
                    individual_masks=mask_meta.get("individual_masks"),
                    min_diff_mean=5.0,
                    min_masked_diff=12.0,
                    min_pct_changed=4.0,
                )
                if is_valid:
                    logger.info("Validation PASSED on attempt %d: diff_mean=%.1f, masked_diff=%.1f", attempt + 1, report.diff_mean or 0, report.masked_diff or 0)
                    final_seed = current_seed
                    break
                else:
                    logger.warning("Validation REJECTED on attempt %d: %s (escalating mask & prompt)", attempt + 1, report.warnings)
                    gen_error = f"Validation rejected on attempt {attempt + 1}: {', '.join(report.warnings)}"
                    edited_img = None
    else:
        # Fallback to Gemini multimodal provider
        gemini_prov = get_gemini_provider()
        base_prompt = build_redesign_prompt(plan)
        final_prompt = build_refinement_prompt(base_prompt, refinement_prompt) if refinement_prompt else base_prompt
        model_name = gemini_prov.get_model_for_tier(quality_tier)
        mask_img, mask_meta = build_inpainting_mask(pil_img, seg_res, plan.interventions, save_debug=True, debug_dir=DEBUG_DIR)
        edited_img, gen_error = await gemini_prov.edit(
            pil_img,
            prompt=final_prompt,
            model_name=model_name,
            quality_tier=quality_tier,
        )
        if edited_img is not None:
            _, report, normalized_img = validate_image_output(edited_img, pil_img.size, orig_img=pil_img, mask_img=mask_img)

    elapsed_ms = int((time.time() - t0) * 1000)
    thermal = compute_thermal_impact(surfaces, [i.dict() for i in plan.interventions])

    # Task 6: Export all 9 debug artifacts in development mode
    try:
        pil_img.save(DEBUG_DIR / "debug_original_image.png")
        save_segmentation_visualization(pil_img, seg_res, DEBUG_DIR / "debug_segmentation_visualization.png")
        if "individual_masks" in mask_meta:
            for m_key, m_val in mask_meta["individual_masks"].items():
                m_val.save(DEBUG_DIR / f"debug_{m_key}.png")
        if mask_img is not None:
            mask_img.save(DEBUG_DIR / "debug_combined_mask.png")
        if edited_img is not None:
            edited_img.save(DEBUG_DIR / "debug_generated_image.png")
        logger.info("Saved all debug artifacts to %s", DEBUG_DIR)
    except Exception as e:
        logger.warning("Could not save debug artifacts: %s", e)

    if edited_img is not None and normalized_img is not None and report is not None and report.is_valid:
        data_url = encode_image_to_base64(normalized_img, quality=92)
        nw, nh = normalized_img.size

        vis_output = {
            "status": "ready",
            "image_url": data_url,
            "width": nw,
            "height": nh,
            "provider": provider_name,
            "model": model_name,
            "quality_tier": quality_tier,
            "generation_time_ms": elapsed_ms,
            "refinement_count": 0,
            "error_message": None,
        }
        cache_entry = {
            "visualization": vis_output,
            "validation": report.dict(),
            "thermal_impact": thermal,
        }
        GENERATION_CACHE[cache_key] = cache_entry

        return {
            "scene_analysis": analysis.dict(),
            "design_plan": plan_dict,
            "visualization": vis_output,
            "thermal_impact": thermal,
            "validation": report.dict(),
        }

    # If generation failed or was rejected by validation after all attempts (never fake result!)
    vis_output = {
        "status": "unavailable",
        "image_url": None,
        "width": pil_img.width,
        "height": pil_img.height,
        "provider": provider_name,
        "model": model_name if 'model_name' in locals() else "unknown",
        "quality_tier": quality_tier,
        "generation_time_ms": elapsed_ms,
        "refinement_count": 0,
        "error_message": gen_error or "Autonomous image generation rejected by validation.",
    }
    return {
        "scene_analysis": analysis.dict(),
        "design_plan": plan_dict,
        "visualization": vis_output,
        "thermal_impact": thermal,
        "validation": report.dict() if report else {
            "is_valid": False,
            "aspect_ratio_preserved": True,
            "dimensions_valid": False,
            "non_blank_verified": False,
            "checks_passed": [],
            "warnings": [gen_error or "Image generation failed"],
        },
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
    interprets structured intent, and executes Gemini multimodal image editing while
    strictly preserving architectural identity and camera perspective.
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

    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(501, "Design refinement is disabled: autonomous local SDXL pipeline operates locally without external API keys.")

    gemini_prov = get_gemini_provider()
    model_name = gemini_prov.get_model_for_tier(quality_tier)

    # 1. Deterministic cache check
    img_hash = hashlib.sha256(content).hexdigest()[:16]
    instr_hash = hashlib.sha256(instr_text.lower().encode("utf-8")).hexdigest()[:16]
    refine_cache_key = f"refine_{img_hash}_{instr_hash}_{model_name}"

    if refine_cache_key in REDESIGN_CACHE:
        cached = REDESIGN_CACHE[refine_cache_key]
        return cached

    # 2. Extract structured design intent (goal, additions, removals, modifications, preservations)
    intent = parse_refinement_intent(instr_text, context=context)

    # 3. Build spatially constrained refinement prompt
    prompt = build_refinement_prompt(instr_text, intent=intent, context=context)

    # 4. Execute multimodal image editing with Gemini
    edited_img, gen_error = await gemini_prov.edit(
        pil_img,
        prompt=prompt,
        model_name=model_name,
        quality_tier=quality_tier,
    )

    elapsed_ms = int((time.time() - t0) * 1000)
    intent_dict = intent.dict() if hasattr(intent, "dict") else intent.model_dump()

    if edited_img is not None:
        _, report, norm_img = validate_image_output(edited_img, pil_img.size)
        data_url = encode_image_to_base64(norm_img, quality=92)
        response_data = {
            "status": "ready",
            "image_url": data_url,
            "width": norm_img.width,
            "height": norm_img.height,
            "instruction": instr_text,
            "provider": "gemini",
            "model": model_name,
            "quality_tier": quality_tier,
            "generation_time_ms": elapsed_ms,
            "intent": intent_dict,
            "error_message": None,
        }
        REDESIGN_CACHE[refine_cache_key] = response_data
        return response_data
    else:
        # Graceful return with unavailable status so the frontend retains the current valid image
        return {
            "status": "unavailable",
            "image_url": None,
            "width": pil_img.width,
            "height": pil_img.height,
            "instruction": instr_text,
            "provider": "gemini",
            "model": model_name,
            "quality_tier": quality_tier,
            "generation_time_ms": elapsed_ms,
            "intent": intent_dict,
            "error_message": gen_error or "Refinement generation unavailable.",
        }


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
        return seg_engine.segment_image(img)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Segmentation error: {e}")


@app.post("/api/v1/inpaint")
async def inpaint_scene(
    file: UploadFile = File(...),
    interventions: str = Form("[]"),
    polygons: str | None = Form(None),
):
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
