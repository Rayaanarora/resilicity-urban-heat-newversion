# data/: Earth Engine exports (you run these)

The backend works without anything in this folder: it falls back to `backend/resilicity_samples.csv`
(3,117 sample points after filtering) and labels everything as **demo data**. The full 250 m city grid adds
population and a dense grid of cells.

Nothing here authenticates on your behalf; the scripts use your own Earth Engine account.

## One-time setup

```powershell
python -m pip install earthengine-api pandas pyarrow
earthengine authenticate          # opens a browser and stores your credentials locally
```

You need a Google Cloud project registered for Earth Engine (https://code.earthengine.google.com/register).

## 1. City grid (hotspot map)

```powershell
python data/export_city_grid.py export --project YOUR_GCP_PROJECT
```

This starts a Drive export task (`chennai_grid_raw.csv` in the Drive folder `resilicity`). It covers the ROI
`[79.90, 12.60, 80.35, 13.20]` at 250 m, about 50,000 cells, and takes a while. When the task finishes, download the
CSV, then:

```powershell
python data/export_city_grid.py convert C:\path\to\chennai_grid_raw.csv
```

This writes `data/chennai_grid.parquet` and `data/chennai_grid.csv`. Restart the backend; `/api/v1/hotspots` then
reports `"data_mode": "grid"`.

Columns: `cell_id, lat, lon, f_tree, f_shrub, f_grass, f_crop, f_built, f_bare, f_water, NDVI, NDBI, MNDWI, LST_C,
population` (people per 250 m cell, JRC GHSL GHS_POP 2020).

Optionally retrain the site model on the grid:

```powershell
cd backend
python -m site_model.train --csv ../data/chennai_grid.csv
```

## 2. Post-construction monitoring (verify loop)

After an intervention is built, compare the project cell with untreated control cells:

```powershell
python data/export_monitor.py --project YOUR_GCP_PROJECT --lat 13.0500 --lon 80.2200 --pre 2023 --post 2027 --out data/monitor_myproject.json
```

Each period is March to May of that year. Controls are 250 m cells 0.75 to 3 km away with a pre-period LST within
1 °C and a built-up fraction within 0.15 of the project cell (all adjustable). Upload the JSON in the project's
Monitor tab, or:

```powershell
curl -X POST -H "Content-Type: application/json" --data-binary @data/monitor_myproject.json http://localhost:8000/api/v1/projects/<id>/measurements
```

The API reports a difference-in-differences estimate with a 95% interval and states plainly when the data cannot
support a claim (fewer than 3 controls, or an interval that includes zero).
