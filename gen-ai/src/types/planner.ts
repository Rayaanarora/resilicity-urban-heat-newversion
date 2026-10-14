import type { ImplementationPlan, SpatialDesignPlan, StreetMetrics } from './resilicity';

/** Where a number comes from. Every figure in the planner UI carries one. */
export type SourceTag = 'model' | 'literature' | 'rule' | 'assumed' | 'measured';

export interface Band {
  low: number;
  central: number;
  high: number;
  source?: SourceTag | string;
}

export interface HotspotCell {
  cell_id: string;
  lat: number;
  lon: number;
  lst_pred_c: number;
  lst_interval_80_c: [number, number];
  lst_interval_90_c: [number, number];
  lst_observed_c: number | null;
  heat_excess_c: number;
  population: number;
  exposure: number;
  priority: number;
  rank: number;
  outside_training_range: boolean;
  land_cover?: Record<string, number>;
  indices?: Record<string, number>;
  distance_m?: number;
}

export interface HotspotMeta {
  data_mode: 'grid' | 'demo';
  data_source: string;
  population_source: string;
  n_cells: number;
  city_median_lst_c: number;
  prediction: string;
  exposure: string;
  model: { name: string; spatial_cv: Record<string, number>; interval_half_width_c: Record<string, number> };
  notes: string[];
  sources: Record<string, SourceTag>;
}

export interface HotspotsResponse extends HotspotMeta {
  cells: HotspotCell[];
}

export interface HotspotDetail extends HotspotMeta {
  cell: HotspotCell;
  nearest: HotspotCell[];
}

export interface AcceptedItem {
  type: string;
  title: string;
  target_zone: string | null;
  quantity: { value: number; unit: string; basis: string; source: SourceTag };
  cost_inr: Band;
  cooling_c: Band | null;
  phase: string;
  phase_label: string;
  lifespan_years: number;
  permits: string[];
  conditions: string[];
  maintenance: string;
}

export interface RejectedItem {
  type: string;
  target_zone: string | null;
  rule: string;
  reason: string;
  measured: Record<string, unknown>;
}

export interface ProjectHotspot extends HotspotCell {
  link: 'selected' | 'nearest';
  data_mode: 'grid' | 'demo';
}

export interface Measurement {
  id: number;
  created_at: number;
  payload: { pre_period?: string; post_period?: string; rows: unknown[] };
  result: MeasurementResult;
}

export interface MeasurementResult {
  project_change_c: number;
  n_controls: number;
  control_changes_c: number[];
  control_mean_change_c?: number;
  did_c: number | null;
  interval_95_c: [number, number] | null;
  supports_claim: boolean;
  statement: string;
  predicted_cooling_c: Band | null;
  comparison_note: string;
  pre_period?: string | null;
  post_period?: string | null;
}

export interface Project {
  id: string;
  name: string;
  created_at: number;
  updated_at: number;
  lat: number | null;
  lon: number | null;
  cell_id: string | null;
  inputs: { road_width_m: number | null; climate_zone: string | null; budget_lakh: number | null };
  hotspot: ProjectHotspot | null;
  data_mode: 'grid' | 'demo' | null;
  photo_url: string;
  street_metrics: (StreetMetrics & { source: SourceTag; road_width_source: SourceTag }) | null;
  accepted: AcceptedItem[];
  rejected: RejectedItem[];
  implementation_plan: ImplementationPlan | null;
  thermal: Record<string, unknown> | null;
  plan: SpatialDesignPlan;
  redesign: {
    status: 'none' | 'queued' | 'running' | 'ready' | 'failed';
    image_url: string | null;
    error: string | null;
    illustrated_interventions?: string[];
    cached?: boolean;
  };
  runs?: { kind: string; status: string; started_at: number; finished_at: number }[];
  measurements?: Measurement[];
}

export interface ProjectSummary {
  id: string;
  name: string;
  created_at: number;
  lat: number | null;
  lon: number | null;
  cell_id: string | null;
  hotspot_rank: number | null;
  lst_pred_c: number | null;
  n_accepted: number;
  n_rejected: number;
  cost_inr_low: number | null;
  cost_inr_high: number | null;
  redesign_status: string;
  photo_url: string;
  data_mode: 'grid' | 'demo' | null;
}

export type Objective = 'person_c' | 'max_cooling' | 'min_cost_for_target';

export interface PortfolioItem {
  key: string;
  type: string;
  title: string;
  zone: string | null;
  phase: string;
  cost_inr: Band;
  cooling_c: Band;
  exposure: number;
  person_c: Band;
  sources: Record<string, string>;
}

export interface PortfolioTotals {
  cost_inr: Omit<Band, 'source'>;
  cooling_c_sum: Omit<Band, 'source'>;
  person_c: Omit<Band, 'source'>;
  c_per_lakh: number;
  person_c_per_lakh: number;
  objective_value: number;
  n_items: number;
}

export interface PortfolioResult {
  feasible: boolean;
  reason?: string;
  objective: Objective;
  budget_lakh: number;
  n_candidates: number;
  selected: { project_id: string; name: string; items: PortfolioItem[] }[];
  totals: PortfolioTotals;
  greedy_cross_check: PortfolioTotals & { dp_at_least_greedy: boolean };
  robustness: Record<'worst' | 'best', {
    assumption: string;
    same_selection: { cost_inr: number; person_c: number; cooling_c_sum: number; within_budget: boolean };
    reoptimised?: { items: string[]; cost_inr: number; person_c: number };
  }>;
  phase_schedule: { phase: string; label: string; items: string[]; cost_inr_central: number }[];
  frontier: { budget_lakh: number; objective_value: number; person_c: number; cost_inr: number; n_items: number }[];
  exposure_source: string[];
  notes: string[];
}

export interface OptimizeRequest {
  project_ids: string[];
  budget_lakh: number;
  objective: Objective;
  phase_limit?: 'quick_win' | 'medium' | 'long_term' | null;
  target_person_c?: number | null;
}

export interface ModelCards {
  site_model: { use: string; spatial_cv: Record<string, number> | null; intervals_c: Record<string, number> | null; card?: Record<string, unknown> };
  legacy_land_cover_model: { use: string; spatial_cv: Record<string, number> | null; card?: Record<string, unknown> };
}
