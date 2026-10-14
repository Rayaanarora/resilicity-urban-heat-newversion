# ResiliCity

ResiliCity turns a street photo into a **buildable cooling plan**: it segments the street, measures it in metres,
keeps only interventions that physically fit, and prices them per 100 m with permits and phasing. An AI image
illustrates the top priorities; the specification is the deliverable.

Work on the Heat Action Planner (city map, projects, budget optimiser, work orders, verification) is in progress on
`feat/heat-action-planner`; see `PLAN.md`. This README describes what is in the code today.

## How it works

```text
Street photo
   │
   ▼
1. Segmentation        SegFormer-B0 (ADE20K) → road, footpath, walls, vegetation, sky, protected objects
   │
   ▼
2. Measurement         footpath widths in metres, using the carriageway width as scale (±30% band)
   │
   ▼
3. Buildable planning  rule-based planner (planner.py + buildable/): gating by clear walkway, junctions,
   │                   obstacles, climate and heritage; ranking by utility or by cooling per lakh under a budget
   │
   ├──► Implementation plan: quantities per 100 m, indicative INR range, phase, permits, rejected items with reasons
   │
   └──► 4. Illustration  FLUX.1 Kontext edits the photo to show the top two intervention types (illustrative only)
```

There is no LLM in the planning path. Planning is deterministic rules plus literature cooling ranges; every threshold,
dimension and unit cost lives in `backend/buildable/config.py` and is labelled indicative.

## Heat numbers and what they mean

- **Cooling per intervention** comes from literature ranges in `buildable/config.py` (low / central / high). It is an
  indicative localized estimate, summed without interaction effects.
- **Land-surface regression** (`heat_model.joblib`): HistGradientBoosting on 7 WorldCover land-cover fractions,
  trained on 3,117 Landsat points around Chennai/Chengalpattu. Spatial cross-validation (5 km blocks):
  **R² 0.258, RMSE 2.475 °C, MAE 1.872 °C** (`model_card.json`). It explains about a quarter of the variation and
  predicts small effects from land-cover changes, so it is shown as a low-confidence reference, not as the source of
  intervention cooling.
- `POST /api/v1/heat` returns the regression's before/after prediction. Its `reliable` flag is `false` only when the
  change has the wrong sign or `|Δ| < 0.05 °C`; the RMSE is reported alongside as a band, it is not used as a gate.
- Landsat LST is surface (skin) temperature, not air temperature, from March to May 2021–2025 medians.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Status of segmentation, heat model, planner and image providers |
| POST | `/api/v1/analyze-and-redesign` | Photo → segmentation, metrics, gated plan, implementation plan, thermal impact, illustrative image. Optional form fields: `design_profile`, `quality_tier`, `road_width_m`, `budget_lakh`, `climate_zone` |
| POST | `/api/v1/refine-design` | Free-text edit of a generated image |
| POST | `/api/v1/plan` | Planner only (legacy shape) |
| POST | `/api/v1/heat` | Land-cover regression before/after (legacy, low confidence) |
| POST | `/api/v1/segment` | Segmentation only |
| POST | `/api/v1/inpaint` | **Deprecated** procedural inpainter, kept for old clients |

Interactive docs: `http://localhost:8000/docs`.

## Image generation

FLUX.1 Kontext through providers tried in order (`IMAGE_PROVIDERS`, default `hf_space,fal`): the free Hugging Face
Space (optional `HF_TOKEN` for more quota) and fal.ai (`FAL_KEY`, paid fallback). Results are cached on disk
(`REDESIGN_CACHE_DIR`), and `REDESIGN_OFFLINE=1` serves cached results only. See `backend/.env.example`.

## Project structure

```text
backend/
  main.py                 FastAPI app and routes
  planner.py              rule-based spatial planner (candidates, interactions, ranking)
  buildable/              config (all numbers), catalog, metrics, observations, gating, implementation, thermal
  scene_understanding/    street geometry, solar/shade proxies, heat priority map
  segmentation/           SegFormer engine and ADE20K → ResiliCity mapping
  image_generation/       FLUX.1 Kontext providers, prompts, redesign engine with disk cache, validation
  inpainting/             deprecated procedural inpainter behind /api/v1/inpaint
  heat_model.joblib, model_card.json, retrain.py, resilicity_samples.csv
  tests/                  pytest suite
gen-ai/                   React 19 + TypeScript + Vite + Zustand + Tailwind frontend
ResiliCity_ML.ipynb       Earth Engine extraction and model training notebook
```

## Quickstart

See `setup.md`. Tests: `python -m pytest backend/tests`.
