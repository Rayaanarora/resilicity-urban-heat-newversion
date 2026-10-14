"""Shared feature definitions and data loading (mirrors ResiliCity_ML.ipynb)."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

FRAC: List[str] = ["f_tree", "f_shrub", "f_grass", "f_crop", "f_built", "f_bare", "f_water"]
IDX: List[str] = ["NDVI", "NDBI", "MNDWI"]
XY: List[str] = ["lat", "lon"]

# Physics prior from the notebook: more built area never cools, more vegetation/water never warms.
MONO: Dict[str, int] = {
    "f_tree": -1, "f_shrub": -1, "f_grass": -1, "f_built": 1, "f_water": -1,
    "NDVI": -1, "NDBI": 1, "MNDWI": -1,
}

BLOCK_DEG = 0.05  # ~5 km spatial blocks for GroupKFold
TARGET = "LST_C"

BACKEND = Path(__file__).resolve().parent.parent
SAMPLES_CSV = BACKEND / "resilicity_samples.csv"


def add_blocks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["block"] = (
        np.floor(df.lat / BLOCK_DEG).astype(int).astype(str) + "_" + np.floor(df.lon / BLOCK_DEG).astype(int).astype(str)
    )
    return df


def load_training(csv: Path = SAMPLES_CSV) -> pd.DataFrame:
    """Same filter as the notebook: non-null features, f_water < 0.5, 20 <= LST <= 65 C."""
    df = pd.read_csv(csv)
    df = df.dropna(subset=[TARGET] + FRAC + IDX)
    df = df[(df.f_water < 0.5) & df[TARGET].between(20, 65)].reset_index(drop=True)
    return add_blocks(df)
