"""Site observations used by gating: obstacles along each kerb and possible junctions.

Obstacle classes are not part of the sidewalk mask (segmentation labels them separately), so they
are measured in a kerb strip: the sidewalk mask dilated by a small margin on its own side.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

from segmentation.mapper import ADE20K_PROTECTED_OBJECTS

from .config import METRICS

# ADE20K ids for the obstacle groups gating cares about.
POLE_IDS = [87, 93, 136]  # streetlight, utility pole, traffic signal
VEHICLE_IDS = [i for i, c in ADE20K_PROTECTED_OBJECTS.items() if c == "vehicle"]
PERSON_IDS = [i for i, c in ADE20K_PROTECTED_OBJECTS.items() if c == "person"]

ZONES = ("left_sidewalk", "right_sidewalk")


@dataclass
class SiteObservations:
    # zone -> {"pole": frac, "vehicle": frac, "person": frac} of the kerb strip
    obstacle_fractions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # zone -> True when a side-street opening is visible on that side
    junction_suspected: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"obstacle_fractions": self.obstacle_fractions, "junction_suspected": self.junction_suspected}


def _kerb_strip(side_mask: np.ndarray, left_half: bool, margin_px: int) -> np.ndarray:
    kernel = np.ones((3, 2 * margin_px + 1), np.uint8)
    strip = cv2.dilate(side_mask.astype(np.uint8), kernel).astype(bool)
    w = side_mask.shape[1]
    cols = np.arange(w)
    half = cols < w // 2 if left_half else cols >= w // 2
    return strip & half[None, :]


def _junction_rows(road_mask: np.ndarray, side_mask: np.ndarray, left: bool) -> bool:
    h, w = road_mask.shape
    y0, y1 = (int(h * f) for f in METRICS["junction_band"])
    rows_with_road, openings = 0, 0
    for y in range(y0, y1):
        xs = np.flatnonzero(road_mask[y])
        if xs.size == 0:
            continue
        rows_with_road += 1
        touches = xs[0] == 0 if left else xs[-1] == w - 1
        if touches and not side_mask[y].any():
            openings += 1
    return rows_with_road > 0 and openings / rows_with_road >= METRICS["junction_min_row_frac"]


def observe_site(
    raw_preds: Optional[np.ndarray],
    road_mask: Optional[np.ndarray],
    left_sidewalk_mask: Optional[np.ndarray],
    right_sidewalk_mask: Optional[np.ndarray],
) -> SiteObservations:
    obs = SiteObservations()
    if raw_preds is None or road_mask is None or left_sidewalk_mask is None or right_sidewalk_mask is None:
        return obs

    margin = max(2, int(METRICS["conflict_strip_frac"] * raw_preds.shape[1]))
    groups = {
        "pole": np.isin(raw_preds, POLE_IDS),
        "vehicle": np.isin(raw_preds, VEHICLE_IDS),
        "person": np.isin(raw_preds, PERSON_IDS),
    }
    for zone, mask, left in (("left_sidewalk", left_sidewalk_mask, True), ("right_sidewalk", right_sidewalk_mask, False)):
        strip = _kerb_strip(mask, left, margin)
        n = int(strip.sum())
        obs.obstacle_fractions[zone] = {
            g: round(float((m & strip).sum()) / n, 3) if n else 0.0 for g, m in groups.items()
        }
        obs.junction_suspected[zone] = _junction_rows(road_mask, mask, left)
    return obs
