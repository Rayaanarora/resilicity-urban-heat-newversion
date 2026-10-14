"""Verify loop: difference-in-differences against control cells, and the measurements endpoint."""

import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from projects.monitor import difference_in_differences, parse_rows

SAMPLE = Path(__file__).resolve().parent.parent.parent / "gen-ai" / "public" / "samples" / "sample_5_commercial_campus.jpg"


def _rows(project_change, control_changes, pre=42.0):
    rows = [{"role": "project", "pre_lst_c": pre, "post_lst_c": pre + project_change}]
    rows += [{"role": "control", "pre_lst_c": pre, "post_lst_c": pre + d} for d in control_changes]
    return rows


def test_did_detects_clear_cooling():
    res = difference_in_differences(_rows(-2.0, [0.4, 0.5, 0.6, 0.5, 0.45]))
    assert res["did_c"] == pytest.approx(-2.0 - 0.49, abs=1e-3)
    lo, hi = res["interval_95_c"]
    assert hi < 0 and res["supports_claim"] and "cooled" in res["statement"]


def test_did_refuses_claim_when_interval_spans_zero():
    res = difference_in_differences(_rows(-0.3, [1.5, -1.2, 0.8, -0.9, 0.2]))
    assert not res["supports_claim"] and "cannot support a claim" in res["statement"]


def test_did_needs_three_controls():
    res = difference_in_differences(_rows(-3.0, [0.1, 0.2]))
    assert res["did_c"] is None and not res["supports_claim"] and "at least 3" in res["statement"]


def test_parse_csv_and_validation():
    csv_text = "role,pre_lst_c,post_lst_c\nproject,41,39\ncontrol,41,41.2\ncontrol,40.5,40.9\ncontrol,41.2,41.1\n"
    parsed = parse_rows(csv_text.encode(), "text/csv")
    assert len(parsed["rows"]) == 4 and parsed["rows"][0]["role"] == "project"
    with pytest.raises(ValueError):
        parse_rows(b"role,pre_lst_c\nproject,41\n", "text/csv")
    with pytest.raises(ValueError):
        difference_in_differences([{"role": "control", "pre_lst_c": 1, "post_lst_c": 2}])


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECTS_DB", str(tmp_path / "p.sqlite3"))
    from main import app
    return TestClient(app)


def test_measurements_endpoint_json_and_file(client):
    with open(SAMPLE, "rb") as f:
        pid = client.post("/api/v1/projects", files={"photo": ("c.jpg", f, "image/jpeg")},
                          data={"lat": "12.95", "lon": "80.20", "road_width_m": "9"}).json()["id"]
    body = {"pre_period": "2023-03/2023-05", "post_period": "2027-03/2027-05", "rows": _rows(-1.8, [0.3, 0.2, 0.4, 0.35])}
    res = client.post(f"/api/v1/projects/{pid}/measurements", json=body)
    assert res.status_code == 201, res.text
    data = res.json()
    assert data["supports_claim"] and data["predicted_cooling_c"]["source"] == "literature"
    assert data["pre_period"] == "2023-03/2023-05" and "different quantities" in data["comparison_note"]

    csv_bytes = b"role,pre_lst_c,post_lst_c\nproject,41,41.1\ncontrol,41,41.0\n"
    res2 = client.post(f"/api/v1/projects/{pid}/measurements", files={"file": ("m.csv", io.BytesIO(csv_bytes), "text/csv")})
    assert res2.status_code == 201 and not res2.json()["supports_claim"]

    listed = client.get(f"/api/v1/projects/{pid}/measurements").json()["measurements"]
    assert len(listed) == 2
    assert client.post(f"/api/v1/projects/{pid}/measurements", content=b"not json or csv").status_code == 422
    assert client.post("/api/v1/projects/nope/measurements", json=body).status_code == 404
