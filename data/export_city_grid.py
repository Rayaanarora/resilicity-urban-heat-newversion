"""Export the Chennai 250 m city grid from Google Earth Engine (run this yourself; see data/README.md).

For every 250 m cell in the ROI it exports: lat, lon, the 7 ESA WorldCover fractions, Sentinel-2 NDVI/NDBI/MNDWI
(Mar-May 2021-2025 median), Landsat 8/9 LST (Mar-May 2021-2025 median, deg C; the model target) and population
(JRC GHSL GHS_POP, people per cell). Definitions match ResiliCity_ML.ipynb.

Two steps:
    python data/export_city_grid.py export --project YOUR_GCP_PROJECT     # starts a Drive export task
    python data/export_city_grid.py convert path/to/chennai_grid_raw.csv  # writes data/chennai_grid.parquet + .csv
"""

import argparse
from pathlib import Path

ROI = [79.90, 12.60, 80.35, 13.20]   # same as the notebook: Chengalpattu to Chennai
START, END = "2021-01-01", "2026-01-01"
SCALE_M = 250
CRS = "EPSG:32644"                   # UTM 44N, metres
CLASSES = {"tree": 10, "shrub": 20, "grass": 30, "crop": 40, "built": 50, "bare": 60, "water": 80}
OUT_DIR = Path(__file__).resolve().parent


def build_stack(ee):
    roi = ee.Geometry.Rectangle(ROI)
    pre_monsoon = ee.Filter.calendarRange(3, 5, "month")
    target = ee.Projection(CRS).atScale(SCALE_M)

    def mask_landsat(img):
        bad = img.select("QA_PIXEL").bitwiseAnd(0b11110).neq(0)   # dilated cloud, cirrus, cloud, shadow
        return img.updateMask(bad.Not())

    def to_celsius(img):
        return img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("LST_C")

    landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
    lst = landsat.filterBounds(roi).filterDate(START, END).filter(pre_monsoon).map(mask_landsat).map(to_celsius).median()

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate(START, END)
          .filter(pre_monsoon).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)).median())
    idx = ee.Image.cat([
        s2.normalizedDifference(["B8", "B4"]).rename("NDVI"),
        s2.normalizedDifference(["B11", "B8"]).rename("NDBI"),
        s2.normalizedDifference(["B3", "B11"]).rename("MNDWI"),
    ])

    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    frac = ee.Image.cat([wc.eq(v).rename("f_" + k) for k, v in CLASSES.items()])

    pop = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP").filter(ee.Filter.eq("system:index", "2020")).first().rename("population")

    def mean_to_cell(img):
        return img.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(target)

    stack = ee.Image.cat([
        mean_to_cell(lst),
        mean_to_cell(idx),
        mean_to_cell(frac),
        pop.reduceResolution(ee.Reducer.sum(), maxPixels=1024).reproject(target),   # people per 250 m cell
        ee.Image.pixelLonLat().reproject(target).rename(["lon", "lat"]),
    ])
    return stack, roi, target


def export(project: str, folder: str) -> None:
    import ee  # pip install earthengine-api

    ee.Initialize(project=project)
    stack, roi, target = build_stack(ee)
    samples = stack.sample(region=roi, projection=target, geometries=False, dropNulls=False, tileScale=4)
    task = ee.batch.Export.table.toDrive(
        collection=samples, description="chennai_grid_raw", folder=folder, fileNamePrefix="chennai_grid_raw",
        fileFormat="CSV",
    )
    task.start()
    print(f"Started Earth Engine task {task.id}: Drive folder '{folder}', file chennai_grid_raw.csv")
    print("Watch it at https://code.earthengine.google.com/tasks, then run the 'convert' step on the downloaded CSV.")


def convert(raw_csv: Path) -> None:
    import pandas as pd

    df = pd.read_csv(raw_csv)
    keep = ["lat", "lon", *("f_" + k for k in CLASSES), "NDVI", "NDBI", "MNDWI", "LST_C", "population"]
    missing = [c for c in keep if c not in df]
    if missing:
        raise SystemExit(f"Missing columns in {raw_csv}: {missing}")
    df = df[keep].dropna(subset=["lat", "lon", "NDVI", "NDBI", "MNDWI"])
    df.insert(0, "cell_id", [f"{la:.4f}_{lo:.4f}" for la, lo in zip(df.lat, df.lon)])
    df.to_csv(OUT_DIR / "chennai_grid.csv", index=False)
    try:
        df.to_parquet(OUT_DIR / "chennai_grid.parquet", index=False)
        print(f"Wrote {len(df)} cells to data/chennai_grid.parquet and data/chennai_grid.csv")
    except ImportError:
        print(f"Wrote {len(df)} cells to data/chennai_grid.csv (install pyarrow for the parquet copy)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="start the Earth Engine export to Google Drive")
    e.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    e.add_argument("--folder", default="resilicity", help="Google Drive folder for the CSV")
    c = sub.add_parser("convert", help="turn the downloaded CSV into data/chennai_grid.parquet")
    c.add_argument("raw_csv", type=Path)
    args = ap.parse_args()
    export(args.project, args.folder) if args.cmd == "export" else convert(args.raw_csv)
