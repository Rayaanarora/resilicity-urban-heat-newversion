import io
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import planner  # noqa: E402

SURF = {"road": 34, "wall": 26, "roof": 18, "vegetation": 16, "pavement": 6}


def reply(items, summary="Hot asphalt."):
    return "Sure!\n" + json.dumps({"site_summary": summary, "interventions": items}) + "\nDone"


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "gray").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    app = FastAPI()
    app.include_router(planner.router)
    return TestClient(app)


def post(client, surfaces=SURF, file=None):
    return client.post("/api/v1/plan", files={"file": ("a.png", file or png_bytes(), "image/png")},
                       data={"surfaces": json.dumps(surfaces)})


def test_numbers_computed_from_surface_mix_not_model_text():
    plan = planner.parse_plan(reply([{"type": "tree_canopy", "coverage": 1.0, "priority": 1,
                                      "cooling": "5C", "landCoverShift": {"f_tree": 0.9}}]), SURF)
    iv = plan["interventions"][0]
    assert iv["landCoverShift"] == {"f_built": -0.1, "f_tree": 0.1}  # coverage 1.0 -> 0.10, capped by 0.5*40% hard
    assert iv["coolingImpact"] == 2.4


def test_tree_shift_capped_by_available_hard_surface():
    iv = planner.parse_plan(reply([{"type": "tree_canopy", "coverage": 1.0}]), {"road": 4, "wall": 10})["interventions"][0]
    assert iv["landCoverShift"]["f_tree"] == 0.02


def test_inapplicable_and_unknown_types_dropped():
    no_roof = {"road": 50, "pavement": 10, "vegetation": 5}
    plan = planner.parse_plan(reply([
        {"type": "cool_roof", "priority": 1}, {"type": "flying_cars", "priority": 2}, {"type": "cool_pavement", "priority": 3},
    ]), no_roof)
    assert [i["type"] for i in plan["interventions"]] == ["cool_pavement"]


def test_duplicates_removed_ranked_and_default_enabled_top_two():
    plan = planner.parse_plan(reply([
        {"type": "shade_structure", "priority": 3}, {"type": "tree_canopy", "priority": 1},
        {"type": "tree_canopy", "priority": 2}, {"type": "cool_roof", "priority": 2},
    ]), SURF)
    ivs = plan["interventions"]
    assert [i["type"] for i in ivs] == ["tree_canopy", "cool_roof", "shade_structure"]
    assert [i["priority"] for i in ivs] == [1, 2, 3]
    assert [i["defaultEnabled"] for i in ivs] == [True, True, False]


def test_bad_values_are_sanitised():
    iv = planner.parse_plan(reply([{"type": "cool_roof", "coverage": "lots", "priority": "x", "target_region": "moon"}]),
                            SURF)["interventions"][0]
    assert iv["targetRegion"] == "roof" and iv["priority"] >= 1 and iv["literatureCoolingC"] == 0.9


@pytest.mark.parametrize("text", ["no json here", '{"interventions": "nope"}', reply([]), reply([{"type": "bogus"}])])
def test_unusable_replies_raise(text):
    with pytest.raises(ValueError):
        planner.parse_plan(text, SURF)


def test_endpoint_happy_path(client, monkeypatch):
    monkeypatch.setattr(planner, "_call_model", lambda b64, s: reply([{"type": "cool_roof", "coverage": 1}]))
    r = post(client)
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "vlm" and body["interventions"][0]["type"] == "cool_roof"


def test_endpoint_without_api_key_is_503(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert post(client).status_code == 503


def test_endpoint_bad_inputs(client, monkeypatch):
    monkeypatch.setattr(planner, "_call_model", lambda b64, s: reply([{"type": "cool_roof"}]))
    assert post(client, file=b"not an image").status_code == 400
    r = client.post("/api/v1/plan", files={"file": ("a.png", png_bytes(), "image/png")}, data={"surfaces": "[1,2]"})
    assert r.status_code == 422


def test_endpoint_model_failure_is_502(client, monkeypatch):
    def boom(b64, s):
        raise RuntimeError("upstream down")
    monkeypatch.setattr(planner, "_call_model", boom)
    assert post(client).status_code == 502
    monkeypatch.setattr(planner, "_call_model", lambda b64, s: "garbage")
    assert post(client).status_code == 502
