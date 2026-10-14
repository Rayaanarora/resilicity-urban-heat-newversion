# ResiliCity: Heat Action Planner

ResiliCity helps a city engineer answer: *"I have ₹X lakh and a ward full of streets. Where do I act first, what exactly
can be built there, and how much cooling do I get per rupee?"*

**Find** hotspots on a Chennai map → **Plan** a street from a photo with buildability rules → **Decide** across
streets under one budget → **Export** a work order → **Verify** after construction with satellite data.

Machine learning answers *where it is hot*. Rules and published literature answer *what can be built and how much
it cools*. Every number in the app carries a source tag (`model`, `measured`, `rule`, `literature`, `assumed`) and a
band. See `docs/ARCHITECTURE.md`.

## Models and their real metrics

| Model | Use | Spatial-CV R² | RMSE °C | MAE °C | Within-block R² |
|---|---|---|---|---|---|
| Site heat model (`site_model.joblib`): land cover + Sentinel-2 NDVI/NDBI/MNDWI + lat/lon | hotspots | 0.722 | 1.514 | 1.165 | 0.707 |
| Legacy land-cover model (`heat_model.joblib`) | single-photo `/heat` only, low confidence | 0.258 | 2.475 | 1.872 | — |

Both trained on 3,117 Landsat 8/9 points (Mar–May 2021–2025 median LST) around Chennai/Chengalpattu, evaluated with
GroupKFold on 0.05° (~5 km) blocks. The site model's split-conformal intervals are ±1.84 °C (80%, measured coverage
80.1%) and ±2.45 °C (90%, coverage 90.1%). Random cross-validation (R² 0.748) leaks spatial information and is reported
only as a comparison. Numbers come from `site_model_card.json` / `model_card.json`; retrain with
`python -m site_model.train` in `backend/`.

The site model describes sites; it is **not** an intervention what-if model. LST is surface temperature, not air
temperature.

## Features

- **Map** (`GET /api/v1/hotspots`): cells ranked by predicted LST above the city median × population; intervals,
  observed LST, land cover, nearest cells.
- **Projects** (`POST /api/v1/projects`): one street per project, from a photo and a hotspot cell or lat/lon.
  Segmentation, metric widths (±30%), gating with reasons, quantities per 100 m, INR bands, phases, permits. Re-plan
  with new inputs; the illustrative FLUX.1 Kontext image is generated on demand.
- **Portfolio** (`POST /api/v1/portfolio/optimize`): exact multiple-choice knapsack maximising person-°C within the
  budget, worst/best cases, efficient frontier, phase schedule.
- **Work orders**: JSON, CSV and PDF per project (`GET /api/v1/projects/{id}/export`) and per portfolio
  (`POST /api/v1/portfolio/export`).
- **Verify** (`POST /api/v1/projects/{id}/measurements`): difference-in-differences against control cells with a 95%
  interval; it says so when the data cannot support a claim.
- **Single photo** (`POST /api/v1/analyze-and-redesign`): the original one-street flow, unchanged.

## Data modes

Without exported data the app runs in **demo mode** on the sample points (spatial out-of-fold predictions, constant
population) and labels it everywhere. To use the full 250 m grid with population, run the Earth Engine export in
`data/README.md` (`data/export_city_grid.py`); the backend switches to grid mode automatically.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Status |
| GET | `/api/v1/model-cards` | Both model cards with real metrics |
| GET | `/api/v1/hotspots?bbox=&top=&min_pop=` | Ranked cells |
| GET | `/api/v1/hotspots/{cell_id}` | Cell detail and 5 nearest |
| POST / GET | `/api/v1/projects` | Create (photo, lat/lon or cell_id, road_width_m, climate_zone, budget_lakh) / list |
| GET / DELETE | `/api/v1/projects/{id}` | Detail (with run log and measurements) / delete |
| POST | `/api/v1/projects/{id}/replan` | Re-run the gated plan with new inputs |
| POST / GET | `/api/v1/projects/{id}/redesign-image` | Queue / fetch the illustrative image |
| GET | `/api/v1/projects/{id}/export?format=json\|csv\|pdf` | Project work order |
| POST / GET | `/api/v1/projects/{id}/measurements` | Upload / list post-construction measurements |
| POST | `/api/v1/portfolio/optimize` | Budget optimisation |
| POST | `/api/v1/portfolio/export?format=json\|csv\|pdf` | Portfolio work order |
| POST | `/api/v1/analyze-and-redesign` | Single-photo flow |
| POST | `/api/v1/refine-design`, `/api/v1/plan`, `/api/v1/segment` | Existing endpoints |
| POST | `/api/v1/heat` | Legacy land-cover regression (its `reliable` flag is false only for a wrong sign or \|Δ\| < 0.05 °C; RMSE is a band) |
| POST | `/api/v1/inpaint` | **Deprecated** |

## Image generation

FLUX.1 Kontext via the free Hugging Face Space (`HF_TOKEN` optional) with fal.ai as a paid fallback (`FAL_KEY`),
disk-cached; `REDESIGN_OFFLINE=1` serves cached images only. The plan never depends on the image.

## Project structure

```text
backend/
  main.py               FastAPI app and routes
  site_model/           site heat model v2: training, prediction, intervals
  hotspots/             grid loading (grid or demo), exposure, ranking
  projects/             SQLite persistence, project service, monitoring (DiD)
  optimizer/            candidates, exact knapsack, portfolio service
  exports/              work-order content and PDF
  analysis_service.py   photo → segmentation → scene → gated plan (shared)
  planner.py, buildable/  rule-based planner, config (all numbers), gating, quantities, cooling bands
  legacy_heat.py        legacy land-cover model and thermal impact (single-photo flow)
  image_generation/, segmentation/, scene_understanding/
  tests/
data/                   Earth Engine export scripts you run yourself (grid, monitoring)
gen-ai/                 React 19 + TypeScript + Vite + Zustand + Tailwind + react-leaflet frontend
docs/ARCHITECTURE.md    the loop and the source of every number
```

## Quickstart

See `setup.md`. Tests: activate `backend/.venv`, then `python -m pytest backend/tests`.
