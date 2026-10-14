"""Hotspot ranking: predicted LST with intervals, heat excess over the city median, exposure-weighted priority."""

import json
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from site_model import load_site_model

from .exposure import compute_exposure, haversine_m
from .grid import CityGrid, load_city_grid

_LOCK = threading.Lock()
_CACHE: Optional["HotspotIndex"] = None


class HotspotIndex:
    """All cells scored once and kept in memory."""

    def __init__(self, grid: CityGrid):
        model = load_site_model()
        if model is None:
            raise RuntimeError("site_model.joblib missing: run `python -m site_model.train` in backend/")
        cells = grid.cells.copy()
        pred = model.predict(cells)

        # Demo cells are the model's own training points: show spatial out-of-fold predictions instead of
        # in-sample fits so the ranking is not flattered by memorisation.
        oof = model.oof_for(cells["sample_id"]) if grid.data_mode == "demo" and "sample_id" in cells else None
        prediction_kind = "spatial out-of-fold" if oof is not None else "model"
        if oof is not None:
            shift = oof - pred["lst_pred_c"].to_numpy()
            for col in pred.columns:
                if col.startswith("lst_"):
                    pred[col] = pred[col] + shift

        cells = pd.concat([cells.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        self.city_median = float(np.median(cells["lst_pred_c"]))
        cells["heat_excess_c"] = cells["lst_pred_c"] - self.city_median
        exposure, exposure_desc = compute_exposure(cells)
        cells["exposure"] = exposure
        cells["priority"] = np.clip(cells["heat_excess_c"], 0, None) * cells["exposure"]
        cells = cells.sort_values(["priority", "heat_excess_c"], ascending=False).reset_index(drop=True)
        cells["rank"] = np.arange(1, len(cells) + 1)

        self.cells = cells
        self.grid = grid
        self.model = model
        self.prediction_kind = prediction_kind
        self.exposure_desc = exposure_desc
        self._by_id = {cid: i for i, cid in enumerate(cells["cell_id"])}
        self._rows = [self._row(r) for r in cells.to_dict("records")]
        self._full_json: Optional[bytes] = None

    def full_json(self) -> bytes:
        """Encoded response for the unfiltered list, built once."""
        if self._full_json is None:
            self._full_json = json.dumps(self.query(), separators=(",", ":")).encode("utf-8")
        return self._full_json

    # -- serialisation -----------------------------------------------------------------------------------------
    def _row(self, r: Dict[str, Any]) -> Dict[str, Any]:
        observed = r.get("LST_C")
        return {
            "cell_id": r["cell_id"],
            "lat": round(float(r["lat"]), 5),
            "lon": round(float(r["lon"]), 5),
            "lst_pred_c": round(float(r["lst_pred_c"]), 2),
            "lst_interval_80_c": [round(float(r["lst_lo80_c"]), 2), round(float(r["lst_hi80_c"]), 2)],
            "lst_interval_90_c": [round(float(r["lst_lo90_c"]), 2), round(float(r["lst_hi90_c"]), 2)],
            "lst_observed_c": round(float(observed), 2) if observed is not None and pd.notna(observed) else None,
            "heat_excess_c": round(float(r["heat_excess_c"]), 2),
            "population": round(float(r["population"]), 1),
            "exposure": round(float(r["exposure"]), 4),
            "priority": round(float(r["priority"]), 4),
            "rank": int(r["rank"]),
            "outside_training_range": bool(r["outside_training_range"]),
        }

    def _detail_row(self, i: int) -> Dict[str, Any]:
        r = self.cells.iloc[i]
        return {
            **self._rows[i],
            "land_cover": {c: round(float(r[c]), 3) for c in ("f_tree", "f_shrub", "f_grass", "f_crop", "f_built", "f_bare", "f_water")},
            "indices": {c: round(float(r[c]), 3) for c in ("NDVI", "NDBI", "MNDWI")},
        }

    def meta(self) -> Dict[str, Any]:
        return {
            "data_mode": self.grid.data_mode,
            "data_source": self.grid.source,
            "population_source": self.grid.population_source,
            "n_cells": int(len(self.cells)),
            "city_median_lst_c": round(self.city_median, 2),
            "prediction": self.prediction_kind,
            "exposure": self.exposure_desc,
            "model": {
                "name": self.model.card.get("name"),
                "spatial_cv": self.model.metrics,
                "interval_half_width_c": self.model.intervals,
            },
            "notes": self._notes(),
            # Source tag for every per-cell field (identical for all cells, so sent once).
            "sources": {
                "lst_pred_c": "model",
                "lst_interval_80_c": "model",
                "lst_interval_90_c": "model",
                "lst_observed_c": "measured",
                "heat_excess_c": "model",
                "population": "measured" if self.grid.data_mode == "grid" else "assumed",
                "exposure": "measured" if self.grid.data_mode == "grid" else "assumed",
                "priority": "model",
            },
        }

    def _notes(self) -> List[str]:
        notes = ["Predicted LST is pre-monsoon land surface temperature, not air temperature."]
        if self.grid.data_mode == "demo":
            notes.append("Demo data: sparse sample points, not the 250 m grid. Run data/export_city_grid.py for the city grid.")
            notes.append("Population is a constant in demo mode, so priority ranks by heat excess only.")
        return notes

    # -- queries ---------------------------------------------------------------------------------------------
    def query(self, bbox: Optional[List[float]] = None, top: Optional[int] = None, min_pop: float = 0.0) -> Dict[str, Any]:
        rows = self._rows
        if bbox:
            west, south, east, north = bbox
            rows = [r for r in rows if west <= r["lon"] <= east and south <= r["lat"] <= north]
        if min_pop:
            rows = [r for r in rows if r["population"] >= min_pop]
        if top:
            rows = rows[:top]
        return {**self.meta(), "cells": rows}

    def cell(self, cell_id: str) -> Optional[Dict[str, Any]]:
        i = self._by_id.get(cell_id)
        return None if i is None else self._rows[i]

    def detail(self, cell_id: str, n_nearest: int = 5) -> Optional[Dict[str, Any]]:
        i = self._by_id.get(cell_id)
        if i is None:
            return None
        lat, lon = self.cells.at[i, "lat"], self.cells.at[i, "lon"]
        d = haversine_m(self.cells["lat"].to_numpy(), self.cells["lon"].to_numpy(), lat, lon)
        order = [j for j in np.argsort(d) if j != i][:n_nearest]
        nearest = [{**self._rows[j], "distance_m": round(float(d[j]), 1)} for j in order]
        return {**self.meta(), "cell": self._detail_row(i), "nearest": nearest}

    def nearest_cell(self, lat: float, lon: float) -> Dict[str, Any]:
        d = haversine_m(self.cells["lat"].to_numpy(), self.cells["lon"].to_numpy(), lat, lon)
        j = int(np.argmin(d))
        return {**self._rows[j], "distance_m": round(float(d[j]), 1)}


def get_index() -> HotspotIndex:
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            _CACHE = HotspotIndex(load_city_grid())
        return _CACHE


def reset_index_cache() -> None:
    global _CACHE
    with _LOCK:
        _CACHE = None
