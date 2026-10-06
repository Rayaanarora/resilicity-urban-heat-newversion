import type { Intervention, SegmentationResponse } from '../types/resilicity';

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? '/api/v1';

export interface HeatResponse {
  lstBeforeC: number;
  lstAfterC: number;
  deltaC: number; // negative = cooler
  rmseC: number;
  outOfRange: string[];
  reliable?: boolean; // false when the change is smaller than the model error or has the wrong sign
  withinNoise?: boolean;
  wrongSign?: boolean;
}

export interface HealthResponse {
  ok: boolean;
  heat_model_available?: boolean;
  segmenter_available?: boolean;
  segmenter_model?: string;
  segmenter_device?: string;
  segmenter_source?: string;
  model?: string;
  n_samples?: number;
  spatial_cv?: {
    R2: number;
    RMSE: number;
    MAE: number;
  };
  planner_available?: boolean;
  segmenter?: boolean;
  generative_provider?: string;
  gemini_configured?: boolean;
  gemini_image_model?: string;
  gemini_final_model?: string;
}

export async function fetchHeat(
  surfaces: Record<string, number>,
  shifts: Record<string, number>,
): Promise<HeatResponse> {
  const r = await fetch(`${BASE}/heat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ surfaces, shifts }),
  });
  if (!r.ok) throw new Error(`Heat API error ${r.status}`);
  return r.json();
}

export async function fetchSegmentation(file: File): Promise<SegmentationResponse> {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch(`${BASE}/segment`, { method: 'POST', body: form });
  if (!r.ok) {
    let errMsg = `Segmentation API error ${r.status}`;
    try {
      const errData = await r.json();
      if (errData?.detail) errMsg = errData.detail;
    } catch {
      // ignore
    }
    throw new Error(errMsg);
  }
  const data = (await r.json()) as SegmentationResponse;
  return data;
}

export async function fetchHealth(): Promise<HealthResponse | null> {
  try {
    const r = await fetch(`${BASE}/health`);
    if (!r.ok) return null;
    return r.json();
  } catch {
    return null;
  }
}

export async function pingBackend(): Promise<boolean> {
  try {
    const r = await fetch(`${BASE}/health`);
    return r.ok;
  } catch {
    return false;
  }
}


export interface PlanResponse {
  source: 'vlm';
  model: string;
  siteSummary: string;
  interventions: Intervention[];
}

export async function fetchPlan(file: File, surfaces: Record<string, number>): Promise<PlanResponse> {
  const form = new FormData();
  form.append('file', file);
  form.append('surfaces', JSON.stringify(surfaces));
  const r = await fetch(`${BASE}/plan`, { method: 'POST', body: form });
  if (!r.ok) {
    let msg = `Plan API error ${r.status}`;
    try {
      const d = await r.json();
      if (d?.detail) msg = d.detail;
    } catch {
      // ignore
    }
    throw new Error(msg);
  }
  return r.json();
}

export interface InpaintResponse {
  ok: boolean;
  imageUrl: string;
  interventions_count: number;
}

export async function fetchInpaint(
  file: File | Blob,
  interventions: any[],
  polygons?: Record<string, any>,
): Promise<InpaintResponse> {
  const form = new FormData();
  form.append('file', file);
  form.append('interventions', JSON.stringify(interventions));
  if (polygons) {
    form.append('polygons', JSON.stringify(polygons));
  }
  const r = await fetch(`${BASE}/inpaint`, { method: 'POST', body: form });
  if (!r.ok) {
    let msg = `Inpaint API error ${r.status}`;
    try {
      const d = await r.json();
      if (d?.detail) msg = d.detail;
    } catch {
      // ignore
    }
    throw new Error(msg);
  }
  return r.json();
}

export async function fetchAnalyzeAndRedesign(
  file: File | Blob,
  designProfile: string = 'balanced',
  requestedInterventions?: string[],
  qualityTier: 'fast' | 'final' = 'fast',
  refinementPrompt?: string
): Promise<import('../types/resilicity').UnifiedRedesignResponse> {
  const form = new FormData();
  form.append('image', file);
  form.append('design_profile', designProfile);
  form.append('quality_tier', qualityTier);
  if (requestedInterventions && requestedInterventions.length > 0) {
    form.append('requested_interventions', JSON.stringify(requestedInterventions));
  }
  if (refinementPrompt) {
    form.append('refinement_prompt', refinementPrompt);
  }

  const r = await fetch(`${BASE}/analyze-and-redesign`, { method: 'POST', body: form });
  if (!r.ok) {
    let msg = `Redesign API error ${r.status}`;
    try {
      const d = await r.json();
      if (d?.detail) msg = d.detail;
    } catch {
      // ignore
    }
    throw new Error(msg);
  }
  return r.json();
}

export async function fetchRefineDesign(
  file: File | Blob,
  refinementPrompt: string,
  qualityTier: 'fast' | 'final' = 'fast'
): Promise<{ status: string; image_url: string; width: number; height: number; provider: string; model: string }> {
  const form = new FormData();
  form.append('image', file);
  form.append('refinement_prompt', refinementPrompt);
  form.append('quality_tier', qualityTier);

  const r = await fetch(`${BASE}/refine-design`, { method: 'POST', body: form });
  if (!r.ok) {
    let msg = `Refine API error ${r.status}`;
    try {
      const d = await r.json();
      if (d?.detail) msg = d.detail;
    } catch {
      // ignore
    }
    throw new Error(msg);
  }
  return r.json();
}

