"""Site heat model v2: card fields come from a real run and the model loads and predicts with intervals."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from site_model import load_site_model
from site_model.features import FRAC, IDX, load_training

CARD = Path(__file__).resolve().parent.parent / "site_model_card.json"


@pytest.fixture(scope="module")
def card():
    assert CARD.exists(), "run `python -m site_model.train` in backend/ first"
    return json.loads(CARD.read_text(encoding="utf-8"))


def test_card_reports_spatial_metrics_and_coverage(card):
    spatial = card["cv"]["spatial"]
    for key in ("R2", "RMSE", "MAE", "within_block_R2"):
        assert isinstance(spatial[key], float)
    assert 0 < spatial["R2"] < 1 and spatial["RMSE"] > 0
    assert "GroupKFold" in card["cv"]["scheme"]
    for level in ("80", "90"):
        iv = card["intervals"][level]
        assert iv["half_width_c"] > 0 and 0 < iv["empirical_coverage_nested_spatial"] <= 1
    assert card["intervals"]["90"]["half_width_c"] > card["intervals"]["80"]["half_width_c"]
    assert card["sklearn_version"] and card["trained_on"] and card["data_source"]
    assert set(FRAC + IDX) <= set(card["features"])


def test_random_cv_is_only_a_leakage_comparison(card):
    # Random folds leak spatial autocorrelation, so they must look better than the spatial headline.
    assert card["leakage_comparison_random_kfold"]["R2"] >= card["cv"]["spatial"]["R2"]


def test_card_states_it_is_not_an_intervention_model(card):
    assert "not an intervention" in card["purpose"].lower() or "must not be used" in card["purpose"]


def test_site_model_predicts_with_intervals():
    model = load_site_model()
    assert model is not None
    df = load_training().head(25)
    out = model.predict(df)
    assert {"lst_pred_c", "lst_lo80_c", "lst_hi80_c", "lst_lo90_c", "lst_hi90_c"} <= set(out.columns)
    assert (out.lst_lo90_c < out.lst_pred_c).all() and (out.lst_pred_c < out.lst_hi90_c).all()
    assert not out.outside_training_range.any()
    oof = model.oof_for(df["system:index"])
    assert oof is not None and len(oof) == 25


def test_out_of_range_inputs_are_flagged():
    model = load_site_model()
    df = load_training().head(1).copy()
    df["NDVI"] = 5.0
    assert model.predict(df).outside_training_range.iloc[0]
