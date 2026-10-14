# ResiliCity architecture: the Heat Action Planner loop

```text
            ┌──────────────────────────────────────────────────────────────────────────────┐
            │                                                                              │
            ▼                                                                              │
   1. FIND  hotspots/  ── site_model (land cover + NDVI/NDBI/MNDWI + lat/lon)               │
            GET /api/v1/hotspots        predicted LST ± conformal interval per cell        │
            rank = heat excess over city median × population exposure                     │
            │                                                                              │
            ▼                                                                              │
   2. PLAN  projects/ ── analysis_service: SegFormer → scene → metres (buildable/metrics)  │
            POST /api/v1/projects       gating (buildable/gating) → accepted + rejected    │
                                        implementation plan per 100 m (quantities, INR,    │
                                        phase, permits); illustrative FLUX.1 Kontext image │
                                        on demand (POST /projects/{id}/redesign-image)     │
            │                                                                              │
            ▼                                                                              │
   3. DECIDE optimizer/ ── exact multiple-choice knapsack over accepted items               │
            POST /api/v1/portfolio/optimize   max person-°C within budget; worst/best      │
                                              cases; 10-point efficient frontier           │
            │                                                                              │
            ▼                                                                              │
   4. EXPORT exports/ ── work orders (JSON, CSV bill of quantities, PDF via reportlab)      │
            GET /projects/{id}/export, POST /portfolio/export                              │
            │                                                                              │
            ▼                                                                              │
   5. VERIFY data/export_monitor.py (Earth Engine, user-run) → POST /projects/{id}/measurements
            difference-in-differences vs control cells, 95% interval, "no claim" when the  │
            data cannot support one ──────────────────────────────────────────────────────┘
```

The single-photo flow (`POST /api/v1/analyze-and-redesign`) is the same Plan step without persistence.

## What is model, rule, literature, assumption or measurement

| Quantity | Source | Where it comes from |
|---|---|---|
| Predicted LST per cell and its 80/90% interval | model | `backend/site_model` (spatial-CV metrics in `site_model_card.json`) |
| Hotspot rank and priority | model × measured/assumed | heat excess (model) × population (GHSL on the grid; constant in demo mode) |
| Observed LST | measured | Landsat 8/9, Mar–May 2021–2025 median |
| Footpath widths | measured (±30% band) | photo rows, scaled by the carriageway width |
| Carriageway width | measured if entered, else assumed (7.0 m) | user input / `buildable/config.py` |
| Accept / reject decisions | rule | `buildable/gating.py`, thresholds in `buildable/config.py` |
| Quantities | rule (assumed when a default width was used) | `buildable/implementation.py` |
| Unit costs and cost bands | assumed | `buildable/config.py` (indicative; replace with the local Schedule of Rates) |
| Cooling per intervention (low / central / high) | literature | `buildable/config.py`, via `buildable/thermal.cooling_band` |
| person-°C | literature × measured/assumed | cooling central × population in the project's hotspot cell |
| Observed change after construction | measured | difference-in-differences against control cells |
| Single-photo heat index | assumed | surface weights in `gen-ai/src/services/heatCalculator.ts` (not calibrated) |
| Legacy land-cover regression change | model, low confidence | `heat_model.joblib`; reference only, never used for intervention cooling |

## Why the ML does not predict intervention cooling

- The land-cover-only regression (spatial-CV R² 0.258) predicts small effects from land-cover changes and is not a
  credible what-if model.
- The Sentinel-2 indices that lift the site model to R² ≈ 0.72 describe the site as it is; a redesign does not edit
  them, and shifting them with land cover was shown to predict warming for built-to-grass in most sites.
- There is no ground truth for the effect of an intervention, so no model could be validated for it. Cooling therefore
  comes from literature bands, and the verify step measures what actually happened.

## Data modes

- **grid**: `data/chennai_grid.parquet` (or `.csv`) exported with `data/export_city_grid.py`; includes population.
- **demo**: no grid present. The 3,117 filtered sample points from `backend/resilicity_samples.csv` stand in, with
  spatial out-of-fold predictions and a constant population. Every response and page says so.
