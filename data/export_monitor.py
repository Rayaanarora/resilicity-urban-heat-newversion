"""Export post-construction Landsat LST for a project cell and untreated control cells (run this yourself).

For the project location it computes the Mar-May Landsat 8/9 LST median for a "pre" and a "post" period at 250 m,
then picks control cells within a ring around the project that had a similar pre-period LST and a similar built-up
fraction (ESA WorldCover), excluding the project's own neighbourhood. Output feeds
POST /api/v1/projects/{id}/measurements.

    python data/export_monitor.py --project YOUR_GCP_PROJECT --lat 13.0500 --lon 80.2200 \
        --pre 2023 --post 2027 --out data/monitor_myproject.json

Periods are years; each uses March to May of that year. Use the same months for both so the comparison is fair.
"""

import argparse
import json
from pathlib import Path

SCALE_M = 250
CRS = "EPSG:32644"


def build(ee, lat, lon, pre_year, post_year, inner_m, outer_m, max_lst_gap, max_built_gap, n_controls):
    point = ee.Geometry.Point([lon, lat])
    ring = point.buffer(outer_m).difference(point.buffer(inner_m))
    target = ee.Projection(CRS).atScale(SCALE_M)

    def mask(img):
        return img.updateMask(img.select("QA_PIXEL").bitwiseAnd(0b11110).eq(0))

    def lst_for(year):
        col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
               .filterBounds(point.buffer(outer_m)).filterDate(f"{year}-03-01", f"{year}-06-01").map(mask))
        lst = col.select("ST_B10").median().multiply(0.00341802).add(149.0).subtract(273.15)
        return lst.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(target)

    built = (ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(50)
             .reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(target).rename("f_built"))
    stack = ee.Image.cat([lst_for(pre_year).rename("pre_lst_c"), lst_for(post_year).rename("post_lst_c"), built,
                          ee.Image.pixelLonLat().reproject(target).rename(["lon", "lat"])])

    project = stack.reduceRegion(ee.Reducer.first(), point, scale=SCALE_M, crs=CRS).getInfo()
    if project.get("pre_lst_c") is None or project.get("post_lst_c") is None:
        raise SystemExit("No cloud-free Landsat data for the project cell in one of the periods; try other years.")

    candidates = stack.sample(region=ring, projection=target, geometries=False).getInfo()["features"]
    pool = []
    for f in candidates:
        p = f["properties"]
        if None in (p.get("pre_lst_c"), p.get("post_lst_c"), p.get("f_built")):
            continue
        lst_gap = abs(p["pre_lst_c"] - project["pre_lst_c"])
        built_gap = abs(p["f_built"] - project["f_built"])
        if lst_gap <= max_lst_gap and built_gap <= max_built_gap:
            pool.append((lst_gap + built_gap, p))
    pool.sort(key=lambda x: x[0])
    controls = [p for _, p in pool[:n_controls]]
    return project, controls


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--pre", type=int, required=True, help="year before construction (Mar-May)")
    ap.add_argument("--post", type=int, required=True, help="year after construction (Mar-May)")
    ap.add_argument("--inner-m", type=float, default=750, help="exclude cells closer than this (spill-over)")
    ap.add_argument("--outer-m", type=float, default=3000)
    ap.add_argument("--max-lst-gap", type=float, default=1.0, help="max pre-period LST difference, deg C")
    ap.add_argument("--max-built-gap", type=float, default=0.15, help="max difference in built-up fraction")
    ap.add_argument("--controls", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("data/monitor.json"))
    a = ap.parse_args()

    import ee  # pip install earthengine-api

    ee.Initialize(project=a.project)
    project, controls = build(ee, a.lat, a.lon, a.pre, a.post, a.inner_m, a.outer_m, a.max_lst_gap, a.max_built_gap,
                              a.controls)

    def row(role, p):
        return {"role": role, "lat": p["lat"], "lon": p["lon"], "pre_lst_c": round(p["pre_lst_c"], 3),
                "post_lst_c": round(p["post_lst_c"], 3), "f_built": round(p["f_built"], 3)}

    out = {
        "pre_period": f"{a.pre}-03/{a.pre}-05",
        "post_period": f"{a.post}-03/{a.post}-05",
        "selection": {"inner_m": a.inner_m, "outer_m": a.outer_m, "max_lst_gap_c": a.max_lst_gap,
                      "max_built_gap": a.max_built_gap},
        "rows": [row("project", project)] + [row("control", c) for c in controls],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {a.out} with {len(controls)} control cells. Upload it to the project's Monitor tab or POST it to "
          f"/api/v1/projects/<id>/measurements.")
    if len(controls) < 3:
        print("Warning: fewer than 3 controls; widen --outer-m or the gaps, otherwise no claim can be made.")


if __name__ == "__main__":
    main()
