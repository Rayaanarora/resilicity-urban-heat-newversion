"""Metric street estimation: sidewalk and road widths in metres from segmentation masks.

Scale reference: the carriageway width in metres (user-supplied road_width_m, default from
config). For a camera looking along the street, ground points on the same image row are at
roughly the same depth, so within a row the ratio sidewalk_px / road_px equals the ratio of
real widths. Widths are measured row by row over the near-field ground band and the median
is reported with an uncertainty band.
"""

from typing import List, Optional, Tuple

import numpy as np

from .config import METRICS
from .models import StreetMetrics


def _row_spans(
    road_mask: np.ndarray,
    side_masks: Tuple[np.ndarray, np.ndarray],
    road_width_m: float,
) -> Tuple[List[Tuple[float, int, int]], List[Tuple[float, int, int]]]:
    """Return (clean_rows, truncated_rows); each row is (metres_per_px, left_px, right_px)."""
    h, w = road_mask.shape
    y0, y1 = (int(h * f) for f in METRICS["row_band"])
    min_span = METRICS["min_road_span_frac"] * w
    left_mask, right_mask = side_masks

    clean, truncated = [], []
    for y in range(y0, min(y1, h)):
        xs = np.flatnonzero(road_mask[y])
        if xs.size < min_span:
            continue
        x0, x1 = int(xs[0]), int(xs[-1])
        touches_left, touches_right = x0 == 0, x1 == w - 1
        if touches_left and touches_right:
            continue  # road wider than the frame: no usable scale on this row
        row = (road_width_m / float(x1 - x0 + 1), int(left_mask[y].sum()), int(right_mask[y].sum()))
        (truncated if (touches_left or touches_right) else clean).append(row)
    return clean, truncated


def _side_width(rows: List[Tuple[float, int, int]], side: int) -> Optional[float]:
    vals = [row[0] * row[side] for row in rows if row[side] >= METRICS["min_side_px"]]
    if len(vals) < METRICS["min_valid_rows"]:
        return None
    width = float(np.median(vals))
    return width if width >= METRICS["min_plausible_sidewalk_m"] else None


def estimate_street_metrics(
    road_mask: Optional[np.ndarray],
    left_sidewalk_mask: Optional[np.ndarray],
    right_sidewalk_mask: Optional[np.ndarray],
    road_width_m: Optional[float] = None,
) -> StreetMetrics:
    """Estimate left/right sidewalk widths in metres using the carriageway as scale reference."""
    scale_source = "user" if road_width_m else "default"
    road_width_m = float(road_width_m or METRICS["default_road_width_m"])
    notes: List[str] = []

    if road_mask is None or left_sidewalk_mask is None or right_sidewalk_mask is None or not road_mask.any():
        return StreetMetrics(
            road_width_m=road_width_m, scale_source=scale_source, uncertainty_pct=METRICS["uncertainty_pct"],
            reliable=False, notes=["No roadway detected; metric scale unavailable."],
        )

    clean, truncated = _row_spans(road_mask, (left_sidewalk_mask, right_sidewalk_mask), road_width_m)
    road_truncated = len(clean) < METRICS["min_valid_rows"]
    rows = truncated if road_truncated else clean
    uncertainty = METRICS["truncated_uncertainty_pct"] if road_truncated else METRICS["uncertainty_pct"]

    if len(rows) < METRICS["min_valid_rows"]:
        return StreetMetrics(
            road_width_m=road_width_m, scale_source=scale_source, uncertainty_pct=uncertainty,
            road_truncated=road_truncated, rows_used=len(rows), reliable=False,
            notes=["Too few ground rows with a measurable carriageway; metric scale unavailable."],
        )
    if road_truncated:
        notes.append("Carriageway runs off the frame edge; widths are biased high and the band is widened.")

    left_m, right_m = _side_width(rows, 1), _side_width(rows, 2)
    for name, val in (("Left", left_m), ("Right", right_m)):
        if val is None:
            notes.append(f"{name} sidewalk not measurable in the photo (absent or occluded).")
    if scale_source == "default":
        notes.append(f"Scale assumes a {road_width_m:.1f} m carriageway; pass road_width_m for a measured value.")

    def band(v: Optional[float]) -> Optional[List[float]]:
        if v is None:
            return None
        return [round(v * (1 - uncertainty / 100.0), 2), round(v * (1 + uncertainty / 100.0), 2)]

    return StreetMetrics(
        road_width_m=road_width_m,
        scale_source=scale_source,
        left_sidewalk_m=round(left_m, 2) if left_m is not None else None,
        right_sidewalk_m=round(right_m, 2) if right_m is not None else None,
        left_range_m=band(left_m),
        right_range_m=band(right_m),
        uncertainty_pct=uncertainty,
        road_truncated=road_truncated,
        rows_used=len(rows),
        reliable=not road_truncated and (left_m is not None or right_m is not None),
        notes=notes,
    )
