"""Train the site heat model v2 and write site_model.joblib + site_model_card.json.

Run from backend/ with the project venv:
    python -m site_model.train                      # uses resilicity_samples.csv
    python -m site_model.train --csv ../data/chennai_grid.csv

Evaluation is spatial GroupKFold on 0.05 deg blocks. Random KFold is reported only as a leakage comparison.
Prediction intervals are split-conformal on spatial out-of-fold residuals; their coverage is measured with nested
folds (the interval width for a fold is fitted without that fold's residuals).
"""

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from .features import BACKEND, FRAC, IDX, MONO, SAMPLES_CSV, TARGET, XY, load_training

N_FOLDS = 5
LEVELS = (0.8, 0.9)
MODEL_PATH = BACKEND / "site_model.joblib"
CARD_PATH = BACKEND / "site_model_card.json"


def make_model(cols: List[str]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=30,
        monotonic_cst=[MONO.get(c, 0) for c in cols], random_state=0,
    )


def within_block_r2(y: np.ndarray, pred: np.ndarray, blocks: pd.Series) -> float:
    """R2 of the variation inside blocks: both series demeaned by their block means."""
    frame = pd.DataFrame({"y": y, "p": pred, "b": blocks.values})
    yw = frame.y - frame.groupby("b").y.transform("mean")
    pw = frame.p - frame.groupby("b").p.transform("mean")
    return float(1.0 - np.sum((yw - pw) ** 2) / np.sum(yw ** 2))


def oof_predictions(df: pd.DataFrame, cols: List[str], spatial: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions and the fold id of each row."""
    X, y = df[cols].values, df[TARGET].values
    splitter = GroupKFold(n_splits=N_FOLDS) if spatial else KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    pred, fold = np.zeros(len(df)), np.zeros(len(df), dtype=int)
    for k, (tr, te) in enumerate(splitter.split(X, y, groups=df.block if spatial else None)):
        pred[te] = make_model(cols).fit(X[tr], y[tr]).predict(X[te])
        fold[te] = k
    return pred, fold


def metrics(df: pd.DataFrame, pred: np.ndarray) -> Dict[str, float]:
    y = df[TARGET].values
    return {
        "R2": float(r2_score(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "within_block_R2": within_block_r2(y, pred, df.block),
    }


def conformal_quantile(abs_resid: np.ndarray, level: float) -> float:
    """Split-conformal quantile with the finite-sample correction."""
    n = len(abs_resid)
    rank = min(n, int(np.ceil((n + 1) * level)))
    return float(np.sort(abs_resid)[rank - 1])


def nested_coverage(abs_resid: np.ndarray, fold: np.ndarray, level: float) -> float:
    hits = []
    for k in np.unique(fold):
        q = conformal_quantile(abs_resid[fold != k], level)
        hits.append(abs_resid[fold == k] <= q)
    return float(np.mean(np.concatenate(hits)))


def main(csv: Path) -> None:
    df = load_training(csv)
    print(f"{len(df)} samples, {df.block.nunique()} spatial blocks (~5 km), {N_FOLDS} folds")

    base_cols = FRAC + IDX
    candidates = {"land_cover+indices": base_cols, "land_cover+indices+latlon": base_cols + XY}
    results, oof = {}, {}
    for name, cols in candidates.items():
        pred, fold = oof_predictions(df, cols)
        results[name], oof[name] = metrics(df, pred), (pred, fold)
        print(f"  spatial CV  {name:28s} " + "  ".join(f"{k} {v:.3f}" for k, v in results[name].items()))

    use_xy = results["land_cover+indices+latlon"]["R2"] > results["land_cover+indices"]["R2"]
    chosen = "land_cover+indices+latlon" if use_xy else "land_cover+indices"
    cols = candidates[chosen]
    print(f"  chosen feature set: {chosen} (lat/lon {'kept' if use_xy else 'dropped'}: spatial-CV R2 "
          f"{results['land_cover+indices+latlon']['R2']:.3f} vs {results['land_cover+indices']['R2']:.3f})")

    rand_pred, _ = oof_predictions(df, cols, spatial=False)
    leakage = metrics(df, rand_pred)
    print(f"  random KFold (leakage comparison only): R2 {leakage['R2']:.3f}  RMSE {leakage['RMSE']:.3f}")

    pred, fold = oof[chosen]
    abs_resid = np.abs(df[TARGET].values - pred)
    intervals = {}
    for level in LEVELS:
        q = conformal_quantile(abs_resid, level)
        cov = nested_coverage(abs_resid, fold, level)
        intervals[f"{int(level * 100)}"] = {"half_width_c": q, "nominal": level, "empirical_coverage_nested_spatial": cov}
        print(f"  {int(level * 100)}% interval: +/-{q:.2f} C, nested spatial coverage {cov:.3f}")

    model = make_model(cols).fit(df[cols].values, df[TARGET].values)
    sample_ids = df["system:index"].astype(str).tolist() if "system:index" in df else [str(i) for i in range(len(df))]
    joblib.dump(
        {
            "model": model,
            "features": cols,
            "intervals": {k: v["half_width_c"] for k, v in intervals.items()},
            # Spatial out-of-fold predictions for the training points, used by demo-mode hotspots so the
            # map never shows in-sample fitted values.
            "oof": {"sample_id": sample_ids, "pred": pred.round(3).tolist()},
        },
        MODEL_PATH,
    )

    card = {
        "name": "ResiliCity site heat model v2",
        "purpose": "Where is it hot: predicts pre-monsoon Landsat land surface temperature for a site. "
                   "Not an intervention what-if model; it must not be used to predict the cooling of a redesign.",
        "target": "Landsat 8/9 LST, Mar-May 2021-2025 median, deg C (surface temperature, not air temperature)",
        "data_source": str(csv.name),
        "n_samples": int(len(df)),
        "n_blocks": int(df.block.nunique()),
        "features": cols,
        "monotonic_constraints": {c: MONO[c] for c in cols if c in MONO},
        "model": "HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=30)",
        "cv": {"scheme": f"GroupKFold({N_FOLDS}) on {0.05} deg blocks", "spatial": results[chosen]},
        "feature_set_comparison": results,
        "leakage_comparison_random_kfold": leakage,
        "intervals": {"method": "split-conformal on spatial out-of-fold absolute residuals", **intervals},
        "training_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in cols},
        "lst_range_c": [float(df[TARGET].min()), float(df[TARGET].max())],
        "sklearn_version": sklearn.__version__,
        "trained_on": dt.date.today().isoformat(),
    }
    CARD_PATH.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"saved {MODEL_PATH.name} and {CARD_PATH.name} (sklearn {sklearn.__version__})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=SAMPLES_CSV)
    main(ap.parse_args().csv)
