"""Load the trained site model and predict LST with conformal intervals."""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .features import BACKEND

MODEL_PATH = BACKEND / "site_model.joblib"
CARD_PATH = BACKEND / "site_model_card.json"


@dataclass
class SiteModel:
    model: object
    features: List[str]
    intervals: Dict[str, float]  # level ("80", "90") -> half width in deg C
    card: Dict
    oof: Dict[str, float] = field(default_factory=dict)  # sample_id -> spatial out-of-fold prediction

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prediction plus 80% / 90% intervals and a training-range flag for each row."""
        pred = self.model.predict(df[self.features].values)
        out = pd.DataFrame({"lst_pred_c": pred}, index=df.index)
        for level, half in self.intervals.items():
            out[f"lst_lo{level}_c"] = pred - half
            out[f"lst_hi{level}_c"] = pred + half
        ranges = self.card.get("training_ranges", {})
        outside = np.zeros(len(df), dtype=bool)
        for col, (lo, hi) in ranges.items():
            if col in df:
                outside |= (df[col] < lo - 1e-9).values | (df[col] > hi + 1e-9).values
        out["outside_training_range"] = outside
        return out

    def oof_for(self, sample_ids: pd.Series) -> Optional[np.ndarray]:
        """Spatial out-of-fold predictions for training points, or None if any id is unknown."""
        if not self.oof:
            return None
        vals = [self.oof.get(str(s)) for s in sample_ids]
        return None if any(v is None for v in vals) else np.array(vals, dtype=float)

    @property
    def metrics(self) -> Dict[str, float]:
        return self.card.get("cv", {}).get("spatial", {})


@lru_cache(maxsize=1)
def load_site_model() -> Optional[SiteModel]:
    """The trained model, or None when site_model.joblib has not been produced yet."""
    if not MODEL_PATH.exists() or not CARD_PATH.exists():
        return None
    bundle = joblib.load(MODEL_PATH)
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    oof = dict(zip(bundle.get("oof", {}).get("sample_id", []), bundle.get("oof", {}).get("pred", [])))
    return SiteModel(model=bundle["model"], features=bundle["features"], intervals=bundle["intervals"], card=card, oof=oof)
