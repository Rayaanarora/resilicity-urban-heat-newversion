import React from 'react';
import { ShieldAlert, Cpu, Sparkles, Flame, FileCode2, LineChart, Target } from 'lucide-react';

export const MethodNoteView: React.FC = () => {
  return (
    <div className="flex flex-col gap-6 max-w-5xl mx-auto font-sans animate-fade-in">
      {/* Header */}
      <div className="flex flex-col gap-1 border-b border-slate-200 pb-5">
        <div className="text-[11px] font-mono font-semibold tracking-widest text-[#0d7a5f] uppercase">
          URBAN HEAT LAB · TECHNICAL METHODOLOGY NOTE
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          ResiliCity Architecture & Machine Learning Calibration
        </h1>
        <p className="text-sm text-slate-600 leading-relaxed max-w-3xl mt-1">
          An honest technical evaluation detailing where ML stands, empirical satellite regression calibration, local SegFormer fine-tuning metrics, and ControlNet-bounded image inpainting.
        </p>
      </div>

      {/* Section 1: Multi-Stage Pipeline */}
      <div className="rc-card p-6 flex flex-col gap-4">
        <div className="flex items-center gap-2 text-slate-900 font-bold font-sans text-base">
          <Cpu className="w-5 h-5 text-[#0d7a5f]" />
          <span>1. Multi-Stage Pipeline Architecture</span>
        </div>
        <p className="text-xs text-slate-600 leading-relaxed">
          ResiliCity avoids treating Generative AI as a black box. Instead of letting a diffusion model guess surface temperatures, we separate visual understanding, physical heat risk estimation, LLM reasoning, and image generation into distinct, verifiable modules.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 font-mono text-xs my-2">
          <div className="bg-slate-50 border border-slate-200 p-3 rounded-xl flex flex-col gap-1">
            <span className="text-[10px] text-teal-700 font-bold uppercase">Stage 1</span>
            <span className="font-bold text-slate-900">SegFormer (pretrained)</span>
            <span className="text-[11px] text-slate-500">Segments roofs, roads, pavement, walls, and vegetation. Local fine-tuning is planned.</span>
          </div>

          <div className="bg-slate-50 border border-slate-200 p-3 rounded-xl flex flex-col gap-1">
            <span className="text-[10px] text-teal-700 font-bold uppercase">Stage 2</span>
            <span className="font-bold text-slate-900">Landsat LST Regressor</span>
            <span className="text-[11px] text-slate-500">Predicts land surface temperature (°C) from land cover, trained on Earth Engine data. Weak fit so far.</span>
          </div>

          <div className="bg-slate-50 border border-slate-200 p-3 rounded-xl flex flex-col gap-1">
            <span className="text-[10px] text-teal-700 font-bold uppercase">Stage 3</span>
            <span className="font-bold text-slate-900">VLM JSON Engine</span>
            <span className="text-[11px] text-slate-500">Claude/GPT-4o output ranked interventions in strict JSON.</span>
          </div>

          <div className="bg-slate-50 border border-slate-200 p-3 rounded-xl flex flex-col gap-1">
            <span className="text-[10px] text-teal-700 font-bold uppercase">Stage 4</span>
            <span className="font-bold text-slate-900">SDXL + ControlNet</span>
            <span className="text-[11px] text-slate-500">Applies inpainting strictly within target mask boundaries.</span>
          </div>
        </div>
      </div>

      {/* Section 2: Where ML Actually Stands (Calibrated ML Models) */}
      <div className="rc-card p-6 flex flex-col gap-5 border-2 border-emerald-500/40">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-900 font-bold font-sans text-base">
            <LineChart className="w-5 h-5 text-emerald-600" />
            <span>2. Where ML Stands: Data Calibration & Model Training</span>
          </div>
          <span className="text-[10px] font-mono font-bold bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded uppercase">
            EMPIRICALLY CALIBRATED
          </span>
        </div>

        <p className="text-xs text-slate-600 leading-relaxed">
          Rather than relying solely on guessed static heat coefficients, ResiliCity incorporates three custom-trained ML components to replace rule-based heuristics with data-driven credibility:
        </p>

        {/* 3 Additions Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-sans text-xs">
          {/* Addition 1: Heat Regression */}
          <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl flex flex-col justify-between gap-3">
            <div className="flex flex-col gap-1">
              <span className="text-[10px] font-mono font-bold text-emerald-700 uppercase">PRIORITY 1 · TRAINED (WEAK FIT)</span>
              <span className="font-bold text-slate-900 text-sm">Landsat LST Satellite Regression</span>
              <p className="text-[11px] text-slate-600 leading-tight">
                Gradient Boosting Regressor predicting Land Surface Temp (LST) from ESA WorldCover land-cover fractions at 3,117 sample points around Chengalpattu & Chennai.
              </p>
            </div>
            <div className="bg-white border border-slate-200 p-2.5 rounded-lg flex justify-between items-center font-mono text-[11px]">
              <span>Spatial-CV R²: <strong className="text-amber-700">0.27</strong></span>
              <span>RMSE: <strong className="text-amber-700">2.46°C</strong></span>
            </div>
          </div>

          {/* Addition 2: Fine-Tuned SegFormer */}
          <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl flex flex-col justify-between gap-3">
            <div className="flex flex-col gap-1">
              <span className="text-[10px] font-mono font-bold text-emerald-700 uppercase">PRIORITY 2 · PLANNED</span>
              <span className="font-bold text-slate-900 text-sm">Local Street Scene SegFormer</span>
              <p className="text-[11px] text-slate-600 leading-tight">
                Plan: fine-tune SegFormer-B0 on 80-150 locally labelled street photos to reduce Western-city dataset bias. Not trained yet.
              </p>
            </div>
            <div className="bg-white border border-slate-200 p-2.5 rounded-lg flex justify-between items-center font-mono text-[11px]">
              <span>Base mIoU: <strong>not measured</strong></span>
              <span>Trained mIoU: <strong>not measured</strong></span>
            </div>
          </div>

          {/* Addition 3: Ward Heat Map */}
          <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl flex flex-col justify-between gap-3">
            <div className="flex flex-col gap-1">
              <span className="text-[10px] font-mono font-bold text-emerald-700 uppercase">PRIORITY 3 · PLANNED</span>
              <span className="font-bold text-slate-900 text-sm">Neighborhood Heat Priority Map</span>
              <p className="text-[11px] text-slate-600 leading-tight">
                Idea: a ward-level model over satellite tiles to rank where interventions matter most. Not built yet.
              </p>
            </div>
            <div className="bg-white border border-slate-200 p-2.5 rounded-lg flex justify-between items-center font-mono text-[11px]">
              <span>Coverage: <strong>none yet</strong></span>
              <span>Status: <strong>planned</strong></span>
            </div>
          </div>
        </div>

        {/* Spatial Cross-Validation Note */}
        <div className="bg-emerald-50/70 border border-emerald-200 p-3.5 rounded-xl text-xs text-emerald-950 flex items-start gap-2.5">
          <Target className="w-4 h-4 text-emerald-700 shrink-0 mt-0.5" />
          <div className="flex flex-col gap-0.5">
            <span className="font-bold font-mono text-[11px] uppercase">Spatial Cross-Validation Rigor</span>
            <span className="text-[11px] leading-tight">
              To prevent spatial autocorrelation data leakage, validation test points are drawn from completely distinct geographical clusters than training points (samples are grouped into ~5 km grid blocks, and each block is held out in turn).
            </span>
          </div>
        </div>
      </div>

      {/* Section 3: Learned Regression Coefficients vs Static Lookup */}
      <div className="rc-card p-6 flex flex-col gap-4">
        <div className="flex items-center gap-2 text-slate-900 font-bold font-sans text-base">
          <Flame className="w-5 h-5 text-orange-600" />
          <span>3. How the regression is used</span>
        </div>
        
        <p className="text-xs text-slate-600 leading-relaxed">
          The regression explains only about a quarter of the spatial variation in surface temperature (R² 0.27, typical error ±2.5°C). A greening scenario changes the predicted value by far less than that error, so the app shows the model's number but uses published literature values for the cooling estimate unless the model's change exceeds its error and cools in the expected direction. Improving the model (more samples, monotonic constraints, extra features) is the next step.
        </p>
      </div>

      {/* Section 4: VLM & ControlNet Architecture */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rc-card p-6 flex flex-col gap-3">
          <div className="flex items-center gap-2 text-slate-900 font-bold font-sans text-base">
            <FileCode2 className="w-5 h-5 text-purple-600" />
            <span>4. Structured VLM JSON Schema</span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            The Vision-Language Model is constrained via system prompts to emit valid JSON conforming to an explicit schema, preventing free-form conversational drift:
          </p>
          <pre className="bg-slate-950 text-emerald-400 p-3 rounded-xl font-mono text-[11px] overflow-x-auto border border-slate-800">
{`{
  "interventions": [
    {
      "type": "cool_roof",
      "targetRegion": "roof",
      "priority": 1,
      "estCostTier": "low"
    }
  ],
  "rationale": "High low-albedo surface area."
}`}
          </pre>
        </div>

        <div className="rc-card p-6 flex flex-col gap-3">
          <div className="flex items-center gap-2 text-slate-900 font-bold font-sans text-base">
            <Sparkles className="w-5 h-5 text-teal-600" />
            <span>5. ControlNet Inpainting & Caching</span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Standard text-to-image models redesign entire buildings. ResiliCity uses ControlNet depth maps to preserve perspective and geometry while applying masked inpainting.
          </p>
          <div className="bg-slate-50 border border-slate-200 p-3.5 rounded-xl text-xs text-slate-700 flex flex-col gap-1 font-mono">
            <span className="font-bold text-slate-900">State Caching Rule:</span>
            <span className="text-[11px] text-slate-500">
              \`CacheKey = activeInterventions.sort().join('|')\`
            </span>
            <span className="text-[11px] text-slate-600 mt-1">
              Toggling an intervention retrieves the cached preloaded image instantly without making duplicate 10-second diffusion backend calls.
            </span>
          </div>
        </div>
      </div>

      {/* Section 5: Honest Caveats Box */}
      <div className="bg-orange-50 border border-orange-200 p-5 rounded-2xl flex items-start gap-3 text-xs text-orange-900 leading-relaxed mb-6">
        <ShieldAlert className="w-5 h-5 text-orange-600 shrink-0 mt-0.5" />
        <div className="flex flex-col gap-2">
          <span className="font-bold uppercase font-mono tracking-wider text-[11px]">
            Honest Scientific & Report Caveats
          </span>
          <ul className="list-disc pl-4 space-y-1 text-[11px] leading-relaxed">
            <li>
              <strong>LST vs. Air Temperature:</strong> Satellite Land Surface Temperature (LST) measures skin temperature (e.g., asphalt reaching 45°C+), not 2m ambient air temperature.
            </li>
            <li>
              <strong>Multi-Date Averaging:</strong> A single satellite pass date can be biased by diurnal weather anomalies. The data is a March-May median of cloud-masked Landsat 8/9 passes (2021-2025), so it reflects the pre-monsoon season only.
            </li>
            <li>
              <strong>Model Estimate vs. Physical Measurement:</strong> The projected temperature drop from regression is a data-calibrated statistical estimate, not a 3D Computational Fluid Dynamics (CFD) physical simulation.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
