"""Verify loop: observed post-construction change against untreated control cells (difference-in-differences).

Input rows (from data/export_monitor.py or entered by hand): role ("project" | "control"), pre_lst_c, post_lst_c and
optionally cell_id, lat, lon. Landsat LST is surface temperature; one median per period per cell.
"""

import csv
import io
import json
import math
from typing import Any, Dict, List, Optional

# Two-sided 95% Student-t critical values by degrees of freedom (1-30); normal beyond.
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]
MIN_CONTROLS = 3


def t95(df: int) -> float:
    return _T95[df - 1] if 1 <= df <= len(_T95) else 1.96


def parse_rows(raw: bytes, content_type: str) -> Dict[str, Any]:
    """Accept the export_monitor JSON ({rows: [...], pre_period, post_period}) or its CSV."""
    text = raw.decode("utf-8-sig")
    if "json" in content_type or text.lstrip().startswith(("{", "[")):
        data = json.loads(text)
        if isinstance(data, list):
            data = {"rows": data}
    else:
        data = {"rows": list(csv.DictReader(io.StringIO(text)))}
    rows = []
    for r in data.get("rows", []):
        try:
            rows.append({
                "role": str(r["role"]).strip().lower(),
                "pre_lst_c": float(r["pre_lst_c"]),
                "post_lst_c": float(r["post_lst_c"]),
                "cell_id": r.get("cell_id"),
                "lat": float(r["lat"]) if r.get("lat") not in (None, "") else None,
                "lon": float(r["lon"]) if r.get("lon") not in (None, "") else None,
            })
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Bad measurement row {r}: needs role, pre_lst_c, post_lst_c ({e})")
    return {"rows": rows, "pre_period": data.get("pre_period"), "post_period": data.get("post_period")}


def difference_in_differences(rows: List[Dict[str, Any]], predicted_cooling_c: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    project = [r for r in rows if r["role"] == "project"]
    controls = [r for r in rows if r["role"] == "control"]
    if len(project) != 1:
        raise ValueError("Exactly one row with role 'project' is required")
    p = project[0]
    d_project = p["post_lst_c"] - p["pre_lst_c"]
    d_controls = [c["post_lst_c"] - c["pre_lst_c"] for c in controls]
    n = len(d_controls)

    out: Dict[str, Any] = {
        "project_change_c": round(d_project, 3),
        "n_controls": n,
        "control_changes_c": [round(d, 3) for d in d_controls],
        "predicted_cooling_c": predicted_cooling_c,
        "source": "measured (Landsat surface temperature)",
        "comparison_note": (
            "Predicted values are indicative literature cooling for the interventions; observed values are "
            "Landsat land surface temperature at 30-100 m resolution. They are different quantities, so the "
            "comparison is a sanity check, not a validation."
        ),
    }
    if n < MIN_CONTROLS:
        out.update(did_c=None, interval_95_c=None, supports_claim=False,
                   statement=f"Only {n} control cell(s); at least {MIN_CONTROLS} are needed. The data cannot support a claim.")
        return out

    mean_c = sum(d_controls) / n
    sd = math.sqrt(sum((d - mean_c) ** 2 for d in d_controls) / (n - 1))
    did = d_project - mean_c
    se = sd * math.sqrt(1 + 1 / n)  # one treated cell against the control mean
    half = t95(n - 1) * se
    lo, hi = did - half, did + half
    supports = lo > 0 or hi < 0
    if not supports:
        statement = (f"Observed change relative to controls is {did:+.2f} °C, but the 95% interval ({lo:+.2f} to {hi:+.2f} °C) "
                     "includes zero. The data cannot support a claim of cooling or warming.")
    elif did < 0:
        statement = (f"The project cell cooled by {abs(did):.2f} °C more than comparable untreated cells "
                     f"(95% interval {lo:+.2f} to {hi:+.2f} °C, surface temperature, {n} controls).")
    else:
        statement = (f"The project cell warmed by {did:.2f} °C relative to comparable untreated cells "
                     f"(95% interval {lo:+.2f} to {hi:+.2f} °C, surface temperature, {n} controls).")
    out.update(
        control_mean_change_c=round(mean_c, 3),
        control_sd_c=round(sd, 3),
        did_c=round(did, 3),
        interval_95_c=[round(lo, 3), round(hi, 3)],
        supports_claim=supports,
        statement=statement,
    )
    return out
