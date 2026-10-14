import type {
  HotspotDetail,
  HotspotsResponse,
  MeasurementResult,
  ModelCards,
  OptimizeRequest,
  PortfolioResult,
  Project,
  ProjectSummary,
} from '../types/planner';

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? '/api/v1';

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `Request failed (${r.status})`;
    try {
      const d = await r.json();
      if (d?.detail) msg = typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail);
    } catch {
      // keep the status message
    }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

/** Resolve an API path returned by the backend (e.g. /api/v1/projects/x/photo) against BASE. */
export const apiUrl = (path: string) => (path.startsWith('/api/v1') ? BASE + path.slice('/api/v1'.length) : path);

export const fetchHotspots = (top?: number) =>
  fetch(`${BASE}/hotspots${top ? `?top=${top}` : ''}`).then((r) => json<HotspotsResponse>(r));

export const fetchHotspotDetail = (cellId: string) =>
  fetch(`${BASE}/hotspots/${encodeURIComponent(cellId)}`).then((r) => json<HotspotDetail>(r));

export interface NewProjectInput {
  photo: File;
  name?: string;
  cellId?: string;
  lat?: number;
  lon?: number;
  roadWidthM?: number;
  climateZone?: string;
  budgetLakh?: number;
}

export function createProject(input: NewProjectInput): Promise<Project> {
  const form = new FormData();
  form.append('photo', input.photo);
  if (input.name) form.append('name', input.name);
  if (input.cellId) form.append('cell_id', input.cellId);
  if (input.lat !== undefined) form.append('lat', String(input.lat));
  if (input.lon !== undefined) form.append('lon', String(input.lon));
  if (input.roadWidthM) form.append('road_width_m', String(input.roadWidthM));
  if (input.climateZone) form.append('climate_zone', input.climateZone);
  if (input.budgetLakh) form.append('budget_lakh', String(input.budgetLakh));
  return fetch(`${BASE}/projects`, { method: 'POST', body: form }).then((r) => json<Project>(r));
}

export const listProjects = () => fetch(`${BASE}/projects`).then((r) => json<{ projects: ProjectSummary[] }>(r));

export const getProject = (id: string) => fetch(`${BASE}/projects/${id}`).then((r) => json<Project>(r));

export const deleteProject = (id: string) => fetch(`${BASE}/projects/${id}`, { method: 'DELETE' }).then((r) => json(r));

export const requestRedesign = (id: string, tier: 'fast' | 'final' = 'fast') =>
  fetch(`${BASE}/projects/${id}/redesign-image?quality_tier=${tier}`, { method: 'POST' }).then((r) => json<{ status: string }>(r));

export function replanProject(id: string, roadWidthM?: number | null, climateZone?: string | null, budgetLakh?: number | null) {
  const form = new FormData();
  if (roadWidthM) form.append('road_width_m', String(roadWidthM));
  if (climateZone) form.append('climate_zone', climateZone);
  if (budgetLakh) form.append('budget_lakh', String(budgetLakh));
  return fetch(`${BASE}/projects/${id}/replan`, { method: 'POST', body: form }).then((r) => json<Project>(r));
}

export function uploadMeasurement(id: string, file: File) {
  const form = new FormData();
  form.append('file', file);
  return fetch(`${BASE}/projects/${id}/measurements`, { method: 'POST', body: form }).then((r) =>
    json<MeasurementResult & { id: number }>(r)
  );
}

export const optimizePortfolio = (req: OptimizeRequest) =>
  fetch(`${BASE}/portfolio/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  }).then((r) => json<PortfolioResult>(r));

export const projectExportUrl = (id: string, format: 'json' | 'csv' | 'pdf') => `${BASE}/projects/${id}/export?format=${format}`;

/** POST the portfolio export and trigger a browser download. */
export async function downloadPortfolioExport(req: OptimizeRequest, format: 'json' | 'csv' | 'pdf') {
  const r = await fetch(`${BASE}/portfolio/export?format=${format}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`Export failed (${r.status})`);
  const blob = await r.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `resilicity-portfolio-work-order.${format}`;
  a.click();
  URL.revokeObjectURL(a.href);
}

export const fetchModelCards = () => fetch(`${BASE}/model-cards`).then((r) => json<ModelCards>(r));
