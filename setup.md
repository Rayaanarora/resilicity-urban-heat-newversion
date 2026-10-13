# ResiliCity: setup

Needs: Node.js 20+ (nodejs.org) and Python 3.10+ (python.org).

Terminal 1, backend:
  cd backend
  python -m venv .venv
  .venv\Scripts\activate          (Mac/Linux: source .venv/bin/activate)
  python -m pip install -r requirements.txt
  python -m uvicorn main:app --reload --port 8000

Terminal 2, front end:
  cd gen-ai
  npm install
  npm run dev
Open the http://localhost:5173 link.

Check: http://localhost:8000/api/v1/health should show "ok": true and "segmenter_available": true.

Image generation (street redesign): copy backend/.env.example to backend/.env. By default the backend
redesigns photos with FLUX.1 Kontext on the free Hugging Face Space (about 30 s per image, no key needed).
- HF_TOKEN (free, huggingface.co/settings/tokens) raises the daily free GPU quota.
- FAL_KEY (fal.ai, a few cents per image) is used only as a fallback when the free Space fails.
- Every result is cached in REDESIGN_CACHE_DIR, so the same photo never costs a second call.
- REDESIGN_OFFLINE=1 serves cached results only: use it while working on the UI.
Click "Export notes" in the app to see the model-based temperature line and export PDF/JSON.
Note: uploaded photos now run live SegFormer semantic segmentation with real polygon overlays.

Model status: the shipped heat model has spatial-CV R2 0.27 (RMSE 2.46 C). The app only uses its
scenario change when it is larger than that error and cools in the expected direction; otherwise it uses
literature values. Re-run ResiliCity_ML.ipynb (now with monotonic constraints) and replace
backend/heat_model.joblib and backend/model_card.json to improve it. Or, with the exported CSV, run
`python retrain.py resilicity_samples.csv` in backend/ (it refuses to save a model where greening warms the site),
then set scikit-learn== in requirements.txt to the version it prints.

