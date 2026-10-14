# ResiliCity: setup

Needs Python 3.10+ (3.12 used in development) and Node.js 20+.

## Backend (terminal 1)

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
copy .env.example .env            # then edit: optional HF_TOKEN, FAL_KEY, cache paths
python -m uvicorn main:app --reload --port 8000
```

Check `http://localhost:8000/api/v1/health`: `"ok": true`, `"segmenter_available": true`.

`.env` is read at startup only; restart the server after editing it.

## Frontend (terminal 2)

```powershell
cd gen-ai
npm install
npm run dev
```

Open `http://localhost:5173` (starts on the Map). The dev server proxies `/api` to `http://localhost:8000`.
Add `?motion=off` to the URL to turn animations off, e.g. on a projector.

## Image generation

- Default: the free FLUX.1 Kontext Hugging Face Space, about 30 s per image. A free `HF_TOKEN`
  (huggingface.co/settings/tokens) raises the daily GPU quota; free accounts get about 5 minutes per day, resetting
  24 hours after first use.
- `FAL_KEY` enables fal.ai as a paid fallback when the Space fails or is out of quota.
- Every result is cached in `REDESIGN_CACHE_DIR`; the same photo and plan never cost a second call.
- `REDESIGN_OFFLINE=1` serves cached results only (use it while working on the UI).

The planner, measurements and implementation plan work without any key; only the illustrative image needs a provider.

## Data and storage

- Projects are stored in `backend/data/projects.sqlite3` with photos under `backend/data/projects/` (override with
  `PROJECTS_DB`).
- Without `data/chennai_grid.parquet` the map, projects and optimiser run in **demo mode** on the sample points; the UI
  says so. To export the real 250 m grid with population, follow `data/README.md` (Earth Engine, your own account).

## Tests

```powershell
backend\.venv\Scripts\activate
python -m pytest backend/tests
cd gen-ai; npm run build; npm run lint
```

## Heat models

- Site heat model (hotspots): `python -m site_model.train` in `backend/` retrains it from `resilicity_samples.csv`
  (or `--csv ../data/chennai_grid.csv`) and rewrites `site_model.joblib` and `site_model_card.json`.
- Legacy land-cover regression (`heat_model.joblib`, spatial-CV R² 0.258, RMSE 2.475 °C): retrain with
  `python retrain.py resilicity_samples.csv` in `backend/` (it refuses to save a model in which greening warms the
  site). `scikit-learn` is pinned in `requirements.txt` to the version the models were trained with.
