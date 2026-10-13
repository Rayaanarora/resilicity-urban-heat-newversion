"""Buildable-design configuration: every threshold, dimension, cost and coefficient lives here.

INDICATIVE VALUES ONLY: replace with the local Schedule of Rates (e.g. CPWD DSR / state SoR)
and the applicable street design guidelines (e.g. IRC:103 pedestrian facilities, IRC SP:117,
the city's street design manual) before using any output for estimates, tenders or approvals.

Nothing outside this file may hardcode a cost, clearance, spacing or cooling number.
"""

CONFIG_LABEL = "indicative, replace with local Schedule of Rates / street design guidelines"

# ---------------------------------------------------------------------------
# 1. Metric estimation (image -> metres)
# ---------------------------------------------------------------------------
METRICS = {
    "default_road_width_m": 7.0,        # carriageway width used as the scale reference when none is given
    "uncertainty_pct": 30,              # +/- band reported on every metric estimate
    "truncated_uncertainty_pct": 50,    # band when the road runs off the frame (scale is biased)
    "row_band": (0.50, 0.97),           # fraction of image height sampled (near-field ground rows)
    "min_road_span_frac": 0.05,         # ignore rows where the road spans < 5% of image width
    "min_side_px": 2,                   # minimum sidewalk pixels in a row to count it
    "min_valid_rows": 5,                # fewer valid rows -> estimate unavailable
    "min_plausible_sidewalk_m": 0.5,    # narrower medians are segmentation specks, reported as not measurable
    "conflict_strip_frac": 0.02,        # kerb-strip dilation around each sidewalk (fraction of width)
    "junction_band": (0.45, 0.80),      # rows inspected for side-street openings
    "junction_min_row_frac": 0.25,      # share of those rows that must show an opening
}
