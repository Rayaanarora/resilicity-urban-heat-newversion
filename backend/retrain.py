"""Retrain the heat model locally from the CSV exported by ResiliCity_ML.ipynb (cell 2).

    python retrain.py path/to/resilicity_samples.csv

Mirrors the notebook (land-cover-only HistGradientBoosting with monotonic constraints, spatial
GroupKFold CV) and writes heat_model.joblib + model_card.json next to this file. It also refuses to
save a model whose greening scenarios warm the site, so a bad retrain cannot silently ship.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).parent
FRAC = ["f_tree", "f_shrub", "f_grass", "f_crop", "f_built", "f_bare", "f_water"]
# More built/bare never cools; more vegetation/water never warms.
MONO = {"f_tree": -1, "f_shrub": -1, "f_grass": -1, "f_built": 1, "f_water": -1}


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=30,
        monotonic_cst=[MONO.get(c, 0) for c in FRAC], random_state=0)


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv).dropna(subset=["LST_C"] + FRAC)
    df = df[(df.f_water < 0.5) & df.LST_C.between(20, 65)].reset_index(drop=True)
    df["block"] = (np.floor(df.lat / 0.05).astype(int).astype(str) + "_" + np.floor(df.lon / 0.05).astype(int).astype(str))
    return df


def main(csv: Path) -> None:
    df = load(csv)
    X, y = df[FRAC].values, df.LST_C.values
    pred = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=df.block):
        pred[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
    cv = {"R2": r2_score(y, pred), "RMSE": float(np.sqrt(mean_squared_error(y, pred))), "MAE": mean_absolute_error(y, pred)}
    print(f"{len(df)} samples, {df.block.nunique()} spatial blocks; spatial CV: " + ", ".join(f"{k} {v:.3f}" for k, v in cv.items()))

    model = make_model().fit(X, y)

    # Sanity gate: moving 5-20% of area from built to tree/grass must never predict warming.
    base = pd.DataFrame([{**{c: 0.0 for c in FRAC}, "f_tree": .05, "f_grass": .05, "f_built": .85, "f_water": .05}])
    for col in ("f_tree", "f_grass"):
        for s in (0.05, 0.10, 0.20):
            t = base.copy(); t["f_built"] -= s; t[col] += s
            d = float(model.predict(t[FRAC])[0] - model.predict(base[FRAC])[0])
            if d > 1e-9:
                sys.exit(f"REFUSING TO SAVE: built->{col[2:]} {s:.0%} predicts {d:+.2f} C (warmer).")
            print(f"built->{col[2:]:5s} {s:>4.0%}: {d:+.2f} C")

    joblib.dump({"model": model, "features": FRAC}, HERE / "heat_model.joblib")
    (HERE / "model_card.json").write_text(json.dumps({
        "spatial_cv": {k: float(v) for k, v in cv.items()},
        "n_samples": int(len(df)),
        "range": {c: [float(df[c].min()), float(df[c].max())] for c in FRAC},
        "sklearn_version": sklearn.__version__,
        "target": "Landsat 8/9 LST, Mar-May 2021-2025 median, deg C",
    }, indent=2))
    print(f"saved heat_model.joblib + model_card.json (scikit-learn {sklearn.__version__}); "
          f"set scikit-learn=={sklearn.__version__} in requirements.txt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
