"""ResiliCity backend: serves the Landsat-calibrated heat model (and, optionally, the SegFormer segmenter).

Run:  uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import io
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from inpainting import ResilientInpainter, encode_image_to_base64
from planner import planner_available, router as planner_router
from segmentation import SegFormerEngine

inpainter = ResilientInpainter()

HERE = Path(__file__).parent
bundle = joblib.load(HERE / "heat_model.joblib")
model, FRAC = bundle["model"], bundle["features"]
card = json.loads((HERE / "model_card.json").read_text())
RANGES = card.get("range")  # optional: {"f_tree": [min, max], ...}

# Startup checks: a model without monotonic constraints can predict that greening warms a site, and a
# pickle loaded under a different scikit-learn version is not guaranteed to behave. Warn loudly.
_built_with = card.get("sklearn_version")
if getattr(model, "monotonic_cst", None) is None:
    warnings.warn("heat_model.joblib has no monotonic constraints; greening scenarios may predict warming. "
                  "Run `python retrain.py <resilicity_samples.csv>`.")
if _built_with and _built_with != sklearn.__version__:
    warnings.warn(f"heat_model.joblib was built with scikit-learn {_built_with}, running {sklearn.__version__}.")

# Initialize SegFormer singleton
seg_engine = SegFormerEngine.get_instance()

app = FastAPI(title="ResiliCity API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(planner_router)

# App surface class -> ESA WorldCover class used by the model
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


class HeatReq(BaseModel):
    surfaces: dict[str, float]  # {"road": 34, "wall": 26, ...} percentages
    shifts: dict[str, float] = {}  # {"f_built": -0.05, "f_tree": 0.05}


@app.get("/api/v1/health")
def health():
    seg_info = seg_engine.get_status()
    return {
        "ok": True,
        "heat_model_available": bundle is not None and "model" in bundle,
        "segmenter_available": seg_info["available"],
        "segmenter_model": seg_info["model_name"],
        "segmenter_device": seg_info["device"],
        "segmenter_source": seg_info["source"],
        "inpainter_available": True,
        "inpainter_engine": "resilient_procedural_cv",
        "model": card.get("target"),
        "n_samples": card.get("n_samples"),
        "spatial_cv": card.get("spatial_cv"),
        "planner_available": planner_available(),
        "segmenter": seg_info["available"],  # backward compatibility
    }


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
    wrong_sign = expected_cooling and delta > 0.01   # model says warming for a greening plan
    # Marginal sensitivity threshold: a paired delta from the same model is sensitive down to ~0.05 °C,
    # whereas RMSE (2.47 °C) is the global cross-validation point prediction error.
    noise_threshold = 0.05
    within_noise = abs(delta) < noise_threshold
    return {
        "lstBeforeC": p0,
        "lstAfterC": p1,
        "deltaC": delta,  # negative = cooler
        "rmseC": rmse,
        "outOfRange": out_of_range,
        "wrongSign": bool(wrong_sign),
        "withinNoise": bool(within_noise),
        "reliable": not (wrong_sign or within_noise),
    }


@app.post("/api/v1/segment")
async def segment(file: UploadFile = File(...)):
    if not seg_engine.is_loaded:
        raise HTTPException(
            503,
            f"Segmentation model unavailable: {seg_engine.load_error or 'Model not initialized'}",
        )

    try:
        content = await file.read()
        if len(content) > 15 * 1024 * 1024:
            raise HTTPException(413, "Image file too large (max 15 MB)")
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid or unreadable image file: {e}")

    try:
        result = seg_engine.segment_image(img)
        return result
    except Exception as e:
        raise HTTPException(500, f"Segmentation inference error: {e}")


@app.post("/api/v1/inpaint")
async def inpaint_scene(
    file: UploadFile = File(...),
    interventions: str = Form("[]"),
    polygons: str | None = Form(None),
):
    try:
        content = await file.read()
        if len(content) > 15 * 1024 * 1024:
            raise HTTPException(413, "Image file too large (max 15 MB)")
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid or unreadable image file: {e}")

    # Parse interventions JSON
    try:
        interventions_list = json.loads(interventions) if interventions else []
    except Exception:
        interventions_list = []

    # Parse or compute polygons
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
                lbl = m.get("label")
                if lbl:
                    polygons_by_class.setdefault(lbl, []).extend(m.get("polygons", []))
        except Exception as e:
            warnings.warn(f"On-the-fly segmentation failed for inpainting: {e}")

    try:
        inpainted_img = inpainter.inpaint(img, polygons_by_class, interventions_list)
        data_url = encode_image_to_base64(inpainted_img)
        return {
            "ok": True,
            "imageUrl": data_url,
            "interventions_count": len(interventions_list),
        }
    except Exception as e:
        raise HTTPException(500, f"Inpainting generation error: {e}")


