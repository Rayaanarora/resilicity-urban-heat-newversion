"""City grid loading: the Earth Engine 250 m grid if exported, else the sample points as labelled demo data."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from site_model.features import FRAC, IDX, load_training

REPO = Path(__file__).resolve().parent.parent.parent
GRID_PARQUET = REPO / "data" / "chennai_grid.parquet"
GRID_CSV = REPO / "data" / "chennai_grid.csv"

# Demo mode has no population data; every point gets this constant so priority reduces to heat excess.
DEMO_POPULATION = 1000.0


@dataclass
class CityGrid:
    cells: pd.DataFrame        # cell_id, lat, lon, features, LST_C (observed), population
    data_mode: str             # "grid" | "demo"
    source: str
    population_source: str     # "JRC GHSL GHS_POP 2020" | "demo constant"


def load_city_grid() -> CityGrid:
    if GRID_PARQUET.exists() or GRID_CSV.exists():
        try:
            df = pd.read_parquet(GRID_PARQUET) if GRID_PARQUET.exists() else pd.read_csv(GRID_CSV)
        except ImportError:
            df = pd.read_csv(GRID_CSV)
        df = df.dropna(subset=["lat", "lon"] + FRAC + IDX).reset_index(drop=True)
        df["cell_id"] = df["cell_id"].astype(str)
        df["population"] = df["population"].fillna(0.0)
        return CityGrid(df, "grid", GRID_PARQUET.name if GRID_PARQUET.exists() else GRID_CSV.name, "JRC GHSL GHS_POP 2020")

    df = load_training()  # 3,117 filtered sample points
    df = df.rename(columns={"system:index": "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str)
    df["cell_id"] = "s" + df["sample_id"]
    df["population"] = DEMO_POPULATION
    return CityGrid(df, "demo", "backend/resilicity_samples.csv (sample points)", "demo constant")
