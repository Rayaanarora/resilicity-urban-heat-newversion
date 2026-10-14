# ResiliCity v2: Heat Action Planner (plan)

Branch: `feat/heat-action-planner` (from `main` at `c40ab51`).

## Baseline (recorded before any change)

- `python -m pytest backend/tests`: **111 passed** (2026-10-14, global Python 3.12.10).
- Backend runtime is `backend/.venv` (Python 3.12.10, **scikit-learn 1.9.1**). The shipped `model_card.json`
  says sklearn 1.9.1, R² 0.258, RMSE 2.475, MAE 1.872. `requirements.txt` has `scikit-learn>=1.8.0` (unpinned).
- `backend/resilicity_samples.csv`: 4,000 rows with `LST_C, MNDWI, NDBI, NDVI`, 7 WorldCover fractions, `lat, lon`.
  The notebook/retrain filter (`f_water < 0.5`, `20 <= LST <= 65`, non-null) leaves the 3,117 training rows.
- Not installed in the venv: `pyarrow`, `reportlab`, `SQLAlchemy`. Frontend has `jspdf` + `html2canvas`
  (client-side PDF of a rendered div), no router (tabs live in the Zustand store), no map library.

## Principles carried through every phase

- ML answers **where it is hot** (site model with Sentinel-2 indices). Rules + literature answer **what can be built
  and how much it cools** (existing gating + `cooling_band`, tagged `literature`).
- The land-cover-only model stays only behind the legacy `/heat` flow, labelled low confidence. It is not extended.
- No site-aware what-if model (it predicts warming for built-to-grass in most sites). No claim anywhere that ML
  predicts intervention cooling.
- Every number shown carries a source tag (`model | literature | rule | assumed | measured`) and a band.
- Existing endpoints keep their contracts; new endpoints are added. `REDESIGN_OFFLINE=1` and the cache keep working.

## Phase 0: housekeeping

- Branch + baseline (done, above). This file.
- Rewrite `README.md` and `setup.md` to match reality: rule-based planner, FLUX.1 Kontext (HF Space / fal),
  actual endpoints, `/inpaint` deprecated, offline mode, and an accurate description of `/heat`
  (`reliable` is false only when the sign is wrong or |Δ| < 0.05 °C; RMSE is reported as a band, not a gate).
- Commit: `docs: align README and setup with the current system`.

## Phase 1: site heat model v2 and the city grid

Files: `data/export_city_grid.py`, `data/README.md`, `backend/site_model/{__init__,features,train,predict}.py`,
`backend/site_model.joblib`, `backend/site_model_card.json`, `backend/tests/test_site_model.py`.

1. `data/export_city_grid.py` (Earth Engine, user-run): ROI `[79.90, 12.60, 80.35, 13.20]`, Mar to May 2021-2025
   medians, 250 m cells: lat, lon, 7 WorldCover fractions, NDVI/NDBI/MNDWI, Landsat LST, JRC GHSL population.
   Writes `data/chennai_grid.parquet` (and `.csv`). I will not run or authenticate it; `data/README.md` documents
   `earthengine authenticate`, the project id flag and the exact command.
2. Grid loader: `data/chennai_grid.parquet` if present (needs `pyarrow`), else `data/chennai_grid.csv`, else
   **demo mode** = filtered `resilicity_samples.csv` points with population set to a labelled constant.
3. `site_model/train.py` (run with the venv):
   - Features: 7 fractions + NDVI + NDBI + MNDWI; lat/lon evaluated as an option and kept only if spatial-CV R²
     improves on the real run (decision recorded in the card).
   - HistGradientBoosting (notebook settings) with `MONO` incl. indices (NDVI -1, NDBI +1, MNDWI -1).
   - Spatial GroupKFold on 0.05° blocks: R², RMSE, MAE, within-block R². Random KFold reported only as a
     leakage comparison.
   - Prediction intervals: split-conformal on spatial out-of-fold residuals (80% and 90%). Coverage is
     measured honestly with nested folds (quantile fitted without the held-out fold). Reported in the card.
   - Writes `site_model.joblib` + `site_model_card.json` (metrics, features, training ranges, sklearn version,
     data source, date, OOF predictions summary). Pin `scikit-learn==1.9.1` in `requirements.txt`.
   - Metrics are printed from the run and copied from the card; nothing hard-coded.
4. `retrain.py`, `heat_model.joblib`, `model_card.json` untouched.

## Phase 2: hotspot API

Files: `backend/hotspots/{__init__,grid,exposure,service}.py`, routes in a new router module, tests.

- Predict LST + interval per cell. **Demo mode uses the spatial out-of-fold predictions** so the hotspot map is not
  shown in-sample-fitted values (labelled). `heat_excess = predicted − city median`.
- `priority = heat_excess × normalised exposure`; `exposure.py` is pluggable (population default, optional POI CSV
  adds weight). In demo mode population is constant, so priority reduces to heat excess (stated in the response).
- `GET /api/v1/hotspots?bbox=&top=&min_pop=` and `GET /api/v1/hotspots/{cell_id}` (detail + 5 nearest).
  Predictions computed once and cached in memory; responses well under 300 ms.

## Phase 3: projects with persistence

Files: `backend/projects/{__init__,db,service,routes}.py`, `backend/analysis_service.py`, tests.

- **stdlib `sqlite3`** (no SQLAlchemy dependency). DB at `backend/data/projects.sqlite3` (gitignored; path
  overridable with `PROJECTS_DB`). Tables `projects`, `project_runs`, `measurements`. Photos under
  `backend/data/projects/<id>/`.
- Extract the photo → segmentation → scene → gated plan → thermal path out of `main.py` into
  `analysis_service.py` (and `compute_thermal_impact` into its own module, re-exported from `main` so existing
  imports and tests keep working).
- `POST /api/v1/projects` (multipart photo, lat, lon, optional cell_id, road_width_m, climate_zone, budget):
  runs analysis **without** image generation. `GET` list/detail, `DELETE`,
  `POST /api/v1/projects/{id}/redesign-image` as a FastAPI background task through the existing engine (cache and
  offline mode honoured), with a status field to poll.

## Phase 4: budget optimiser

Files: `backend/optimizer/{__init__,candidates,knapsack,service}.py`, route, tests.

- Candidates = gated-accepted line items of each project: cost (low/central/high), cooling band (`literature`),
  exposure weight from the project's hotspot cell (population; constant in demo), phase.
- Exclusivity groups (pick at most one): street trees `{tree_canopy, parking_lane_tree}` per zone; roof
  `{cool_roof, green_roof}`; overhead shade `{shade_structure, solar_canopy, facade_awning}` per zone. Others are
  singleton groups.
- Exact multiple-choice knapsack, DP over cost in ₹0.1 lakh units; greedy ratio solver as cross-check.
  Objectives: `person_c` (Σ exposure × cooling_central), `max_cooling`, `min_cost_for_target` (needs
  `target_person_c`; DP over cost returning the cheapest set reaching it). Optional `phase_limit`.
- Robustness: the chosen set evaluated at worst case (cooling_low, cost_high) and best case (cooling_high,
  cost_low), plus a re-optimisation for each scenario.
- Output: selected items per project, totals, person-°C, °C per ₹ lakh, phase schedule, 10-point budget sweep.
- `POST /api/v1/portfolio/optimize`. Tests: budget never exceeded, exclusivity respected, objective monotonic in
  budget, DP ≥ greedy, empty input handled.

## Phase 5: work-order export

- Per project and per portfolio: JSON, CSV (bill of quantities) and PDF with street, hotspot rank, items, INR
  low/central/high, permits, phase, conditions, rejected options with rule and measured values, source tags,
  bands, data mode, both model cards and the disclaimer (literature cooling; surface, not air, temperature).
- The existing PDF is client-side (html2canvas screenshot). Per the brief, a backend generator is added and the
  client one kept. **Needs a dependency decision (question below).**

## Phase 6: verify loop

- `data/export_monitor.py` (Earth Engine, user-run): post-construction Landsat LST median for the project cell and
  nearby untreated control cells with a similar baseline.
- `POST /api/v1/projects/{id}/measurements` accepts that CSV/JSON. Difference-in-differences:
  `(post − pre)_project − mean(post − pre)_controls`, interval from the spread of control changes. If fewer than 3
  controls or the interval spans zero, the response says the data cannot support a claim.
- UI shows predicted (literature band) vs observed.

## Phase 7: frontend

- Pages via the existing store tab state (no router): **Map**, **Projects** (list + the existing analysis flow as
  the project page, plus a Monitor tab), **Portfolio**, **Method**.
- Map: heat-ranked cells coloured by predicted LST, population as size/opacity, cell panel, top-N list, demo badge.
- Portfolio: project picker, budget slider, objective toggle, results table, efficient-frontier chart (plain SVG,
  no chart library), phase timeline, export buttons.
- A shared `<Figure value band source>` component so every number shows its source tag and band. Empty states
  instead of dummy numbers; the current sample "saved runs" mock data is removed.

## Phase 8: honesty and polish

- Copy review across README, UI and Method; Method shows both model cards (loaded from the backend, not typed in).
- `docs/ARCHITECTURE.md`: loop diagram and a table of what is model, rule, literature or assumption.

## Process per phase

`python -m pytest backend/tests` + new tests, `npm run build` + `oxlint` in `gen-ai/`, start the API on a spare
port and curl the new endpoints, then one commit per phase.

## Decisions taken (change any of these if you disagree)

- `data/` at the repo root for Earth Engine scripts and grid files (as in the brief); runtime data in `backend/data/`.
- stdlib `sqlite3` instead of SQLAlchemy.
- Demo hotspots use spatial out-of-fold predictions, not in-sample fits.
- lat/lon kept in the site model only if the real spatial-CV run improves on indices-only.
- Frontier and timeline charts as plain SVG; no chart dependency.
- Training runs in `backend/.venv` (sklearn 1.9.1), and that version is pinned.

## Risks

- Demo mode is a sparse point set (3,117 points over the ROI), not a 250 m grid; the map will look like dots.
- Without the GHSL export, population is a constant, so "person-°C" equals cooling × constant (labelled).
- Within-block R² and conformal coverage on 3,117 points may differ from the brief's figures; I report what the run
  gives.
- New dependencies install into `backend/.venv` and `gen-ai/node_modules`, both on the C: drive (see question).

## Dependency decision (answered)

Approved: install `reportlab` and `pyarrow` into `backend/.venv` and `leaflet` + `react-leaflet` into `gen-ai`,
with pip and npm **caches on D:** (`PIP_CACHE_DIR=D:\Resillicity\pip_cache`, `npm_config_cache=D:\Resillicity\npm_cache`).
The packages themselves live in the project's venv/node_modules on C:.

Original question: dependencies (all would install on C:, inside the project): `reportlab` (backend PDF), `pyarrow` (read the parquet
grid) in the venv, and `leaflet` + `react-leaflet` (map with an OpenStreetMap basemap, which loads tiles from the
internet) in the frontend. The no-dependency alternative is a text PDF with the existing `jspdf`, a CSV grid, and an
SVG cell map without a basemap.

## Outcome (2026-10-14)

All phases implemented on `feat/heat-action-planner`, one commit per phase.

- Tests: 111 at baseline, **153 passing** at the end (`backend/.venv`, `python -m pytest backend/tests`).
- Site model (real run, `site_model_card.json`): spatial-CV R² 0.722, RMSE 1.514, MAE 1.165, within-block R² 0.707
  with lat/lon (indices only: 0.707 / 1.556 / 1.194 / 0.701, so lat/lon was kept). Random KFold R² 0.748 (leakage
  comparison only). Conformal intervals 80% ±1.84 °C (nested coverage 0.801), 90% ±2.45 °C (0.901).
- Legacy model unchanged: R² 0.258, RMSE 2.475, MAE 1.872.
- Demo mode: no `data/chennai_grid.parquet` yet, so hotspots run on the 3,117 sample points with spatial out-of-fold
  predictions and a constant population (1,000 per point), labelled in the API and UI.

Additional decisions taken during implementation:

- Added `POST /api/v1/projects/{id}/replan` and `GET /api/v1/model-cards` (needed by the project and method pages).
- Gating treats "no carriageway scale in the photo" as unknown (allowed with a survey condition) and "road measured,
  no footpath found" as evidence (rejected).
- Optimiser costs are rounded up to Rs 0.1 lakh units so the rupee budget is never exceeded; exclusivity groups are
  street trees per zone, roof, and overhead shade per zone.
- Verify: difference-in-differences needs at least 3 controls; the interval uses Student t with n-1 df on the spread
  of control changes.
- Frontend routing stays in the store, with URL hashes (#map, #project/<id>) and `?motion=off` for projectors.

Known limitations:

- The road-material detector sometimes flags ordinary asphalt as historic cobblestone, which rejects cool pavement
  (pre-existing; seen on 3 of 5 samples).
- The illustrative image depends on the free Hugging Face quota (about 5 GPU minutes per day per account).
- The OSM basemap tiles were not confirmed in headless screenshots; the cell layer was.
