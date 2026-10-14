"""Work-order exports: JSON, CSV and PDF for a project and a portfolio."""

import csv
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLES = Path(__file__).resolve().parent.parent.parent / "gen-ai" / "public" / "samples"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECTS_DB", str(tmp_path / "p.sqlite3"))
    from main import app
    return TestClient(app)


def _project(client, name="sample_5_commercial_campus.jpg", width="9"):
    with open(SAMPLES / name, "rb") as f:
        r = client.post("/api/v1/projects", files={"photo": (name, f, "image/jpeg")},
                        data={"lat": "12.95", "lon": "80.20", "road_width_m": width})
    assert r.status_code == 201
    return r.json()


def test_project_json_export_has_required_sections(client):
    p = _project(client)
    order = client.get(f"/api/v1/projects/{p['id']}/export", params={"format": "json"}).json()
    assert order["kind"] == "project_work_order" and order["data_mode"] in ("grid", "demo")
    assert order["hotspot"]["rank"] >= 1
    line = order["interventions"][0]
    assert line["cost_inr_low"] <= line["cost_inr_central"] <= line["cost_inr_high"]
    assert line["cooling_source"] == "literature" and line["cost_source"] == "assumed"
    assert all(r["rule"] and r["reason"] for r in order["rejected"])
    assert order["model_cards"]["site_model"]["spatial_cv"]["R2"] > 0
    assert order["model_cards"]["legacy_land_cover_model"]["spatial_cv"]["R2"] > 0
    assert "not the air temperature" in order["disclaimer"] and "literature" in order["disclaimer"]


def test_project_csv_has_one_row_per_accepted_item(client):
    p = _project(client)
    res = client.get(f"/api/v1/projects/{p['id']}/export", params={"format": "csv"})
    assert res.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(res.text)))
    assert len(rows) == len(p["accepted"])
    assert {"quantity", "unit", "cost_inr_low", "cost_inr_central", "cost_inr_high", "permits", "phase"} <= set(rows[0])


def test_project_pdf_is_a_pdf(client):
    p = _project(client)
    res = client.get(f"/api/v1/projects/{p['id']}/export", params={"format": "pdf"})
    assert res.status_code == 200 and res.headers["content-type"] == "application/pdf"
    assert res.content.startswith(b"%PDF") and len(res.content) > 2000
    assert "attachment" in res.headers["content-disposition"]


def test_portfolio_exports(client):
    ids = [_project(client)["id"], _project(client, "sample_2_sidewalk_road.jpg", "14")["id"]]
    body = {"project_ids": ids, "budget_lakh": 30}
    order = client.post("/api/v1/portfolio/export", params={"format": "json"}, json=body).json()
    assert order["kind"] == "portfolio_work_order" and order["projects"]
    assert order["totals"]["cost_inr"]["central"] <= 30 * 100_000
    assert "disclaimer" in order and order["model_cards"]
    pdf = client.post("/api/v1/portfolio/export", params={"format": "pdf"}, json=body)
    assert pdf.content.startswith(b"%PDF")
    rows = list(csv.DictReader(io.StringIO(client.post("/api/v1/portfolio/export", params={"format": "csv"}, json=body).text)))
    assert len(rows) == sum(len(p["interventions"]) for p in order["projects"])
    assert client.get("/api/v1/projects/nope/export").status_code == 404
