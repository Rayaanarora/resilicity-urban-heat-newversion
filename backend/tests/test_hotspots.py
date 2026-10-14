"""Hotspot ranking and endpoints (demo mode on the sample points unless a city grid was exported)."""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hotspots import get_index
from main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_cells_are_ranked_by_priority():
    idx = get_index()
    pri = idx.cells["priority"].to_numpy()
    assert (pri[:-1] >= pri[1:]).all()
    assert list(idx.cells["rank"][:3]) == [1, 2, 3]
    top = idx.cells.iloc[0]
    assert top["heat_excess_c"] > 0 and top["lst_pred_c"] > idx.city_median


def test_demo_mode_uses_out_of_fold_predictions_and_says_so():
    idx = get_index()
    if idx.grid.data_mode != "demo":
        pytest.skip("city grid present")
    meta = idx.meta()
    assert meta["prediction"] == "spatial out-of-fold"
    assert meta["population_source"] == "demo constant"
    assert any("Demo data" in n for n in meta["notes"])
    oof = idx.model.oof_for(idx.cells["sample_id"].head(10))
    assert oof == pytest.approx(idx.cells["lst_pred_c"].head(10).to_numpy(), abs=1e-6)


def test_hotspots_endpoint_shape_and_filters(client):
    res = client.get("/api/v1/hotspots", params={"top": 10})
    assert res.status_code == 200
    data = res.json()
    assert data["data_mode"] in ("grid", "demo") and len(data["cells"]) == 10
    cell = data["cells"][0]
    for key in ("cell_id", "lat", "lon", "lst_pred_c", "lst_interval_90_c", "population", "priority", "rank"):
        assert key in cell
    lo, hi = cell["lst_interval_90_c"]
    assert lo < cell["lst_pred_c"] < hi
    assert data["sources"]["lst_pred_c"] == "model" and data["sources"]["population"] in ("measured", "assumed")

    bbox = "80.20,13.00,80.30,13.10"
    boxed = client.get("/api/v1/hotspots", params={"bbox": bbox}).json()["cells"]
    assert all(80.20 <= c["lon"] <= 80.30 and 13.00 <= c["lat"] <= 13.10 for c in boxed)
    assert client.get("/api/v1/hotspots", params={"bbox": "1,2,3"}).status_code == 422


def test_hotspot_detail_returns_five_nearest(client):
    first = client.get("/api/v1/hotspots", params={"top": 1}).json()["cells"][0]
    detail = client.get(f"/api/v1/hotspots/{first['cell_id']}").json()
    assert detail["cell"]["cell_id"] == first["cell_id"]
    assert "land_cover" in detail["cell"] and "indices" in detail["cell"]
    assert len(detail["nearest"]) == 5
    dists = [n["distance_m"] for n in detail["nearest"]]
    assert dists == sorted(dists) and dists[0] > 0
    assert client.get("/api/v1/hotspots/does-not-exist").status_code == 404


def test_full_grid_responds_quickly(client):
    client.get("/api/v1/hotspots")  # warm cache
    t = time.perf_counter()
    res = client.get("/api/v1/hotspots")
    assert res.status_code == 200
    assert time.perf_counter() - t < 0.3
