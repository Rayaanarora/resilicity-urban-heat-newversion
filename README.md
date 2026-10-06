# ResiliCity: Urban Heat Island AI Resilience Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React_19-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

An AI-powered urban resilience platform that turns street-level imagery into an explainable, data-backed cooling plan with interactive before/after visual simulations.

---

## Architecture Overview

```text
Street-Level Photo
        │
        ▼
STAGE 1 — Scene Understanding (LIVE)
SegFormer Semantic Segmentation (nvidia/segformer-b0-finetuned-ade-512-512)
        │
        ├─────────────────────────────────────────┐
        │                                         │
        ▼                                         ▼
Surface Mix Percentages                      Visual Context
(Road, Pavement, Veg, Wall, Sky)                  │
        │                                         │
        ▼                                         ▼
STAGE 2 — Heat Risk Regressor (LIVE)         STAGE 3 — VLM Planner (LIVE, needs ANTHROPIC_API_KEY)
Landsat 8/9 LST + Sentinel-2 Gradient Boost       Structured JSON Interventions
        │                                         │
        │                                         ▼
        │                                    STAGE 4 — SDXL + ControlNet (Future)
        │                                    Depth-Bounded Street Inpainting
        │                                         │
        └───────────────────┬─────────────────────┘
                            ▼
                  ResiliCity Dashboard
            Interactive Slider · SVG Overlays · PDF Report
```

---

## Features

- **Stage 1 (Live Scene Understanding):** Pretrained SegFormer Vision Transformer performs real-time pixel-level segmentation on uploaded street photos.
- **ADE20K to ResiliCity Taxonomy:** 150 ADE20K classes deterministically mapped into 8 core resilience classes (`road`, `pavement`, `vegetation`, `wall`, `roof`, `water`, `sky`, `other`).
- **Dynamic Mask Overlays:** OpenCV multi-region contour extraction rendered as crisp SVG overlays with per-class visibility toggles.
- **Stage 3 (Live VLM Planner):** `POST /api/v1/plan` sends the photo and measured surface mix to a vision model, which picks interventions from a fixed catalog. Land-cover shifts and cooling estimates are computed server-side from the measured surfaces, not taken from model text. Without an API key the app falls back to rule-based interventions and says so in the UI.
- **Stage 1 → Stage 2 Integration:** Segmented surface percentages automatically feed into the satellite heat regressor.
- **Stage 2 (Live Empirical Heat Regressor):** Gradient Boosting model trained on 3,117 Google Earth Engine satellite points across Chennai/Chengalpattu with spatial cross-validation and monotonic physics constraints.
- **Hybrid Temperature Calculations:** Combines satellite LST delta predictions (for land-cover shifts) with literature-calibrated cooling factors (for albedo treatments like cool roofs and cool pavements).
- **Interactive Before/After Preview:** Drag-slider comparison with synchronized SVG mask clip paths.
- **Automated Reporting:** Client-side JSON export and full PDF report generation.

---

## Project Structure

```text
gen-ai_project_fixed/
├── backend/                        # FastAPI Python ML backend
│   ├── segmentation/               # SegFormer engine, taxonomy mapper, contour extractor
│   │   ├── __init__.py
│   │   ├── model.py                # Singleton SegFormer inference engine
│   │   ├── mapper.py               # ADE20K -> ResiliCity label mapping
│   │   └── contours.py             # OpenCV polygon extraction
│   ├── heat_model.joblib           # Trained satellite LST regression model
│   ├── model_card.json             # Spatial CV metrics (R²=0.27, RMSE=2.46°C)
│   ├── planner.py                  # Stage 3 VLM planner (/plan) + output validation
│   ├── tests/                      # pytest (run: python -m pytest backend/tests)
│   ├── main.py                     # FastAPI routes (/health, /segment, /heat)
│   └── requirements.txt            # Python dependencies
├── gen-ai/                         # React 19 + TypeScript + Vite frontend
│   ├── src/
│   │   ├── components/             # BeforeAfterPreview, SurfaceMasksCard, HeatRiskScoreCard, etc.
│   │   ├── services/               # API clients and heat calculation algorithms
│   │   ├── store/                  # Zustand global application state
│   │   └── types/                  # TypeScript interfaces
│   └── package.json
├── ResiliCity_ML.ipynb             # Google Earth Engine data extraction & model training notebook
└── setup.md                        # Quickstart instructions
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- Node.js 20+

### 1. Backend Setup (Terminal 1)
```bash
cd backend
python -m pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # optional: enables the Stage 3 planner (PLANNER_MODEL overrides the model)
python -m uvicorn main:app --reload --port 8000
```
- API Health Check: `http://localhost:8000/api/v1/health`
- Interactive OpenAPI Docs: `http://localhost:8000/docs`

### 2. Frontend Setup (Terminal 2)
```bash
cd gen-ai
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## Implemented ML & Inpainting Modules

1. **Part B Local SegFormer Fine-Tuning (`fine_tune_segformer.py`):** Fully implemented pipeline fine-tuning `nvidia/mit-b0` to the native 8-class ResiliCity taxonomy (`road`, `pavement`, `vegetation`, `wall`, `roof`, `water`, `sky`, `other`). Weights and processor are saved at `backend/segformer-local-final`, which the inference engine automatically detects and loads as `local_fine_tuned`.
2. **Stage 4 Resilient Inpainting Engine (`backend/inpainting/`):** Mask-bounded procedural inpainter applying photo-realistic cooling transformations (high-albedo reflective cool roofs, extensive green roofs, reflective cool pavement, permeable interlocking pavers, lush street tree canopies with ground shadows, and tensile fabric shade structures) directly through `POST /api/v1/inpaint`.
