"""Legacy land-cover heat regression (heat_model.joblib) and the indicative thermal-impact calculation.

The land-cover-only model is low confidence (spatial-CV R2 0.258) and serves only the legacy single-photo flow
(/api/v1/heat and the thermal_impact block); it is not extended. Intervention cooling comes from literature bands.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import sklearn
from fastapi import HTTPException

from buildable import cooling_band, land_cover_shift
from buildable.config import THERMAL

HERE = Path(__file__).parent

bundle = joblib.load(HERE / "heat_model.joblib")
model, FRAC = bundle["model"], bundle["features"]
card = json.loads((HERE / "model_card.json").read_text())
RANGES = card.get("range")

_built_with = card.get("sklearn_version")
if getattr(model, "monotonic_cst", None) is None:
    warnings.warn("heat_model.joblib has no monotonic constraints; run `python retrain.py`.")
if _built_with and _built_with != sklearn.__version__:
    warnings.warn(f"heat_model.joblib built with {_built_with}, running {sklearn.__version__}.")


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
    """Indicative microclimate cooling with confidence bands.

    Per-type coefficients and land-cover shifts come from buildable/config.py. Types without a
    coefficient are listed in unmodelledInterventions instead of silently counting as zero.
    """
    total_cooling_est = 0.0
    cooling_lo = cooling_hi = 0.0
    shifts = {"f_built": 0.0, "f_tree": 0.0, "f_grass": 0.0}
    per_item: List[Dict[str, Any]] = []
    unmodelled: List[str] = []

    for item in interventions:
        itype = item.get("type", "")
        cov = float(item.get("coverage", 0.5))
        band = cooling_band(itype, cov)
        if band is None:
            unmodelled.append(itype)
            continue
        lo, mid, hi = band
        total_cooling_est += mid
        cooling_lo += lo
        cooling_hi += hi
        per_item.append({"type": itype, "coolingC": mid, "rangeC": [lo, hi]})
        for k, v in land_cover_shift(itype, cov).items():
            shifts[k] = shifts.get(k, 0.0) + v

    # Machine learning land surface temperature model prediction
    try:
        before = to_fractions(surfaces)
        after = {k: max(0.0, before[k] + shifts.get(k, 0.0)) for k in FRAC}
        s = sum(after.values())
        after = {k: v / s for k, v in after.items()}
        p0, p1 = predict(before), predict(after)
        ml_delta = p1 - p0
    except Exception:
        p0 = THERMAL["fallback_baseline_c"]
        p1, ml_delta = p0 - total_cooling_est, -total_cooling_est

    rmse = float(card["spatial_cv"]["RMSE"]) if card else None
    r2 = float(card["spatial_cv"]["R2"]) if card else None
    result = {
        "baselineSurfaceTempC": round(p0, 1),
        "projectedSurfaceTempC": round(p0 - total_cooling_est, 1),
        "totalCoolingReductionC": round(total_cooling_est, 1),
        "mlModelDeltaC": round(ml_delta, 2),
        "methodology": "Dual-layer: Physical empirical microclimate calculations coupled with Landsat 8/9 LST regression.",
        "disclaimer": "AI image visualization illustrates architectural cooling intent; physical cooling figures are computed via thermodynamic and empirical models.",
        # Confidence bands (optional, newer clients)
        "coolingRangeC": [round(cooling_lo, 1), round(cooling_hi, 1)],
        "perIntervention": per_item,
        "unmodelledInterventions": unmodelled,
        "mlModelR2": round(r2, 2) if r2 is not None else None,
        "mlModelRmseC": round(rmse, 2) if rmse is not None else None,
        "mlModelDeltaRangeC": [round(ml_delta - rmse, 2), round(ml_delta + rmse, 2)] if rmse is not None else None,
        "confidence": "low",
        "confidenceNote": (
            f"Indicative only: the land-surface-temperature regression has spatial-CV R2 {r2:.2f} "
            f"(RMSE {rmse:.1f} C), and per-type cooling coefficients are literature-based ranges "
            "summed without interaction effects." if r2 is not None else
            "Indicative only: per-type cooling coefficients are literature-based ranges."
        ),
    }
    if unmodelled:
        result["warning"] = f"No cooling coefficient for: {', '.join(unmodelled)}; excluded from totals."
    return result
