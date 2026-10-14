"""Exposure weighting for hotspot priority (pluggable).

Default: population per cell, normalised to [0, 1] by the city maximum. Optionally, a CSV of points of interest
(data/pois.csv with columns lat, lon, type, weight; e.g. schools, bus stops, hospitals) multiplies a cell's exposure
by (1 + sum of weights within POI_RADIUS_M).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .grid import REPO

POI_CSV = REPO / "data" / "pois.csv"
POI_RADIUS_M = 250.0
EARTH_R_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(a))


def population_exposure(cells: pd.DataFrame) -> np.ndarray:
    pop = cells["population"].to_numpy(dtype=float)
    top = pop.max() if len(pop) else 0.0
    return pop / top if top > 0 else np.zeros_like(pop)


def poi_multiplier(cells: pd.DataFrame, pois: pd.DataFrame) -> np.ndarray:
    mult = np.ones(len(cells))
    for _, p in pois.iterrows():
        near = haversine_m(cells["lat"].to_numpy(), cells["lon"].to_numpy(), p["lat"], p["lon"]) <= POI_RADIUS_M
        mult[near] += float(p.get("weight", 1.0))
    return mult


def compute_exposure(cells: pd.DataFrame) -> tuple:
    """(exposure array, description of what it includes)."""
    exp = population_exposure(cells)
    parts = ["population (normalised)"]
    if POI_CSV.exists():
        pois = pd.read_csv(POI_CSV)
        if {"lat", "lon"} <= set(pois.columns) and len(pois):
            exp = exp * poi_multiplier(cells, pois)
            parts.append(f"{len(pois)} points of interest within {POI_RADIUS_M:.0f} m")
    return exp, " x ".join(parts)
