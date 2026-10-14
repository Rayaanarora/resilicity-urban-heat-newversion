"""Project persistence: create from a photo (no image generation), read back, on-demand image, delete."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import projects.service as project_service
from image_generation import HuggingFaceSpaceKontextProvider, RedesignEngine
from main import app
from projects import db

SAMPLE = Path(__file__).resolve().parent.parent.parent / "gen-ai" / "public" / "samples" / "sample_5_commercial_campus.jpg"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECTS_DB", str(tmp_path / "projects.sqlite3"))
    monkeypatch.delenv("REDESIGN_OFFLINE", raising=False)
    monkeypatch.setattr(project_service, "_engine", RedesignEngine(["hf_space"], cache_dir=str(tmp_path / "cache")))
    return TestClient(app)


def _create(client, **form):
    data = {"lat": "12.95", "lon": "80.20", "road_width_m": "9", **form}
    with open(SAMPLE, "rb") as f:
        return client.post("/api/v1/projects", files={"photo": ("campus.jpg", f, "image/jpeg")}, data=data)


def test_create_project_runs_gated_plan_without_image(client):
    res = _create(client)
    assert res.status_code == 201, res.text
    p = res.json()
    assert p["redesign"]["status"] == "none" and p["redesign"]["image_url"] is None
    assert p["accepted"], "expected at least one accepted intervention"
    item = p["accepted"][0]
    assert item["cost_inr"]["low"] <= item["cost_inr"]["central"] <= item["cost_inr"]["high"]
    assert item["cooling_c"]["source"] == "literature" and item["cost_inr"]["source"] == "assumed"
    assert all(r["rule"] and r["reason"] for r in p["rejected"])
    assert p["street_metrics"]["road_width_m"] == 9.0 and p["street_metrics"]["road_width_source"] == "measured"
    assert p["hotspot"] is not None and p["hotspot"]["link"] == "nearest"


def test_project_persists_and_lists(client):
    pid = _create(client, name="Campus street").json()["id"]
    assert db.get_project(pid)["name"] == "Campus street"   # read straight from SQLite
    listed = client.get("/api/v1/projects").json()["projects"]
    assert [p["id"] for p in listed] == [pid]
    detail = client.get(f"/api/v1/projects/{pid}").json()
    assert detail["runs"][0]["kind"] == "analysis" and detail["runs"][0]["status"] == "ok"
    assert client.get(f"/api/v1/projects/{pid}/photo").status_code == 200


def test_redesign_image_is_generated_on_demand(client):
    pid = _create(client).json()["id"]
    assert client.get(f"/api/v1/projects/{pid}/redesign-image").status_code == 404
    fake = Image.fromarray(np.random.default_rng(0).integers(30, 220, (300, 400, 3), dtype=np.uint8))
    with patch.object(HuggingFaceSpaceKontextProvider, "edit", new=AsyncMock(return_value=(fake, None))):
        res = client.post(f"/api/v1/projects/{pid}/redesign-image")   # background task runs before return
    assert res.status_code == 202
    p = client.get(f"/api/v1/projects/{pid}").json()
    assert p["redesign"]["status"] == "ready" and p["redesign"]["illustrated_interventions"]
    assert client.get(f"/api/v1/projects/{pid}/redesign-image").status_code == 200


def test_redesign_honours_offline_mode(client, tmp_path, monkeypatch):
    pid = _create(client).json()["id"]
    monkeypatch.setenv("REDESIGN_OFFLINE", "1")
    monkeypatch.setattr(project_service, "_engine", RedesignEngine(["hf_space"], cache_dir=str(tmp_path / "empty")))
    client.post(f"/api/v1/projects/{pid}/redesign-image")
    p = client.get(f"/api/v1/projects/{pid}").json()
    assert p["redesign"]["status"] == "failed" and "Offline mode" in p["redesign"]["error"]


def test_create_requires_location_and_valid_cell(client):
    with open(SAMPLE, "rb") as f:
        res = client.post("/api/v1/projects", files={"photo": ("x.jpg", f, "image/jpeg")})
    assert res.status_code == 422
    assert _create(client, cell_id="nope").status_code == 422


def test_delete_project_removes_row_and_files(client):
    p = _create(client).json()
    folder = Path(db.get_project(p["id"])["photo_path"]).parent
    assert client.delete(f"/api/v1/projects/{p['id']}").status_code == 200
    assert db.get_project(p["id"]) is None and not folder.exists()
    assert client.get(f"/api/v1/projects/{p['id']}").status_code == 404
