export type SurfaceClass = 'roof' | 'road' | 'vegetation' | 'pavement' | 'sidewalk' | 'water' | 'wall' | 'sky' | 'other';

export interface SurfaceMask {
  id: string;
  className: SurfaceClass;
  label: string;
  areaPercentage: number; // e.g., 34.5
  albedo: number; // 0.05 to 0.85
  emissivity: number; // 0.80 to 0.98
  color: string; // Hex for overlay mask display
  pixelCount?: number;
  polygonPoints?: [number, number][]; // Relative % coordinates [x, y] for primary polygon
  polygons?: [number, number][][]; // All simplified polygons in relative % coordinates
}

export interface SegmentationClass {
  id: SurfaceClass;
  label: string;
  percentage: number;
  pixel_count: number;
  polygons: [number, number][][];
}

export interface SegmentationModelMetadata {
  name: string;
  source: string;
  device: string;
}

export interface SegmentationResponse {
  model: SegmentationModelMetadata;
  image: {
    width: number;
    height: number;
  };
  classes: SegmentationClass[];
  source: 'segformer' | 'fallback';
  masks: SurfaceMask[];
}

export type InterventionType =
  | 'cool_roof'
  | 'tree_canopy'
  | 'cool_pavement'
  | 'shade_structure'
  | 'shade_canopy'
  | 'green_roof'
  | 'permeable_pave'
  | 'facade_awning'
  | 'bus_shelter'
  | 'solar_canopy'
  | 'kerb_bioswale'
  | 'parking_lane_tree'
  | 'cool_wall'
  | 'misting_system'
  | 'green_facade';

export type CostTier = 'low' | 'med' | 'high';

export interface Intervention {
  id: string;
  type: InterventionType;
  title: string;
  targetRegion: SurfaceClass | 'sidewalk' | 'courtyard' | 'pedestrian';
  priority: number;
  estCostTier: CostTier;
  estCostText: string;
  coolingImpact: number; // Temperature score reduction e.g. 1.4
  description: string;
  promptTemplate: string;
  defaultEnabled?: boolean;
  coverage?: number;
  // Land-cover change the ML model can see (fractions of the scene, e.g. {f_built: -0.05, f_tree: 0.05}).
  landCoverShift?: Partial<Record<'f_built' | 'f_tree' | 'f_grass', number>>;
  // Optional literature-based surface cooling in deg C for albedo measures the ML model cannot see.
  // If omitted, the old assumed score-to-temperature conversion is used and flagged as assumed.
  literatureCoolingC?: number;
}

export type ClimateZone = 'tropical' | 'arid' | 'temperate' | 'mediterranean';

export interface HeatScoreMetrics {
  baseScore: number; // 0.0 - 10.0
  projectedScore: number; // 0.0 - 10.0
  deltaScore: number;
  surfaceCooling: number; // score points removed by reflective / surface treatments
  shadeCooling: number; // score points removed by trees and shade structures
  estimatedTempReductionC: number; // e.g. 3.2°C
  modelCoolingC: number; // part of the estimate from the Landsat-calibrated regression
  assumedCoolingC: number; // part from literature / assumed conversion
  tempSource: 'model' | 'model+assumed' | 'assumed';
  modelRmseC?: number; // spatial-CV RMSE of the regression
  outOfRange?: string[]; // land-cover inputs outside the training range
  modelDeltaC?: number; // raw change predicted by the regression (negative = cooler), shown even when not used
  modelUnreliable?: boolean; // true when the regression change is within its error or has the wrong sign
  isReliable: boolean;
  fallbackReason?: string;
  surfaceComposition: Record<SurfaceClass, number>;
}

export interface SampleScene {
  id: string;
  title: string;
  location: string;
  description: string;
  rawImageUrl: string;
  afterImageUrl: string; // Default combined intervention after image
  inpaintedVariants?: Record<string, string>; // Map of activeInterventionKeys -> image URL
  masks: SurfaceMask[];
  interventions: Intervention[];
  baseScore: number;
  isReliable: boolean;
  fallbackReason?: string;
}

export interface BackendStatus {
  isConnected: boolean;
  endpointUrl: string;
  mode: 'hybrid_demo' | 'colab_live' | 'local_fastapi';
  latencyMs: number;
  segmenterAvailable?: boolean;
  segmenterModel?: string;
  segmenterDevice?: string;
  segmenterSource?: string;
  plannerAvailable?: boolean;
  generativeProvider?: string;
  geminiConfigured?: boolean;
  geminiImageModel?: string;
  geminiFinalModel?: string;
}

export type DesignProfile =
  | 'balanced'
  | 'pedestrian_first'
  | 'maximum_cooling'
  | 'green_infrastructure'
  | 'low_cost';

export interface SpatialInterventionSpec {
  type: string;
  target_region: string;
  priority: number;
  coverage: number;
  placement: string;
  visual_design: string;
  reason: string;
  feasibility: number;
  cooling_impact_c: number;
  confidence: number;
  title?: string;
  target_zone?: string | null;
  evidence?: string[];
  spatial_constraints?: string[];
}

/** Sidewalk/road widths in metres estimated from the photo (carriageway as scale reference). */
export interface StreetMetrics {
  road_width_m: number;
  scale_source: 'user' | 'default' | string;
  left_sidewalk_m?: number | null;
  right_sidewalk_m?: number | null;
  left_range_m?: [number, number] | null;
  right_range_m?: [number, number] | null;
  uncertainty_pct: number;
  road_truncated: boolean;
  rows_used: number;
  scale_available?: boolean;
  reliable: boolean;
  notes: string[];
}

export interface ImplementationLineItem {
  type: string;
  title: string;
  target_zone?: string | null;
  quantity: number;
  unit: string;
  basis: string;
  quantity_flagged: boolean;
  cost_inr_low: number;
  cost_inr_high: number;
  cooling_c: number;
  cooling_per_lakh_c: number;
  phase: 'quick_win' | 'medium' | 'long_term' | string;
  phase_label: string;
  lifespan_years: number;
  maintenance: string;
  permits: string[];
  conditions: string[];
  extra?: Record<string, number>;
  dimensions?: Record<string, number | number[]>;
}

export interface ImplementationPhase {
  phase: string;
  label: string;
  items: string[];
  cost_inr_low: number;
  cost_inr_high: number;
}

export interface RejectedCandidate {
  type: string;
  target_zone?: string | null;
  reason: string;
  conflict?: string;
  rule?: string;
  measured?: Record<string, unknown>;
}

/** Bill of quantities, indicative costs, phasing and permits per street_length_m of street. */
export interface ImplementationPlan {
  street_length_m: number;
  line_items: ImplementationLineItem[];
  phases: ImplementationPhase[];
  total_cost_inr_low: number;
  total_cost_inr_high: number;
  total_cooling_c: number;
  cooling_per_lakh_c: number;
  budget_lakh?: number | null;
  within_budget?: boolean | null;
  budget_excluded: RejectedCandidate[];
  prerequisites: string[];
  assumptions: string[];
  clearances?: Record<string, number>;
  source: string;
}

export interface SpatialDesignPlan {
  site_summary: string;
  heat_drivers: string[];
  constraints: string[];
  design_profile: DesignProfile;
  interventions: SpatialInterventionSpec[];
  overall_design_intent: string;
  rejected_candidates?: RejectedCandidate[];
  street_metrics?: StreetMetrics | null;
  implementation_plan?: ImplementationPlan | null;
}

export interface SceneAnalysis {
  width: number;
  height: number;
  aspect_ratio: string;
  surface_percentages: Record<string, number>;
  has_visible_roof: boolean;
  has_pedestrian_sidewalk: boolean;
  has_road_corridor: boolean;
  existing_vegetation_pct: number;
  detected_constraints: string[];
}

export interface VisualizationOutput {
  status: 'ready' | 'unavailable' | 'cached';
  image_url: string | null;
  width?: number;
  height?: number;
  provider: string;
  model: string;
  quality_tier: 'fast' | 'final';
  generation_time_ms: number;
  refinement_count: number;
  error_message?: string | null;
}

export interface ValidationReport {
  is_valid: boolean;
  aspect_ratio_preserved: boolean;
  dimensions_valid: boolean;
  non_blank_verified: boolean;
  checks_passed: string[];
  warnings: string[];
}

export interface RefinementRevision {
  revision: number;
  instruction: string;
  imageUrl: string;
  timestamp: number;
  model?: string;
}

export interface RefineDesignResponse {
  status: 'ready' | 'unavailable' | 'cached';
  image_url?: string | null;
  width?: number;
  height?: number;
  instruction: string;
  provider?: string;
  model?: string;
  quality_tier?: 'fast' | 'final';
  generation_time_ms?: number;
  error_message?: string | null;
  intent?: {
    goal: string;
    add: string[];
    remove: string[];
    modify: string[];
    preserve: string[];
    spatial_constraints: string[];
  };
}

export interface UnifiedRedesignResponse {
  scene_analysis: SceneAnalysis;
  design_plan: SpatialDesignPlan;
  visualization: VisualizationOutput;
  thermal_impact: {
    baselineSurfaceTempC: number;
    projectedSurfaceTempC: number;
    totalCoolingReductionC: number;
    mlModelDeltaC: number;
    methodology: string;
    disclaimer: string;
    coolingRangeC?: [number, number];
    perIntervention?: { type: string; coolingC: number; rangeC: [number, number] }[];
    unmodelledInterventions?: string[];
    mlModelR2?: number | null;
    mlModelRmseC?: number | null;
    mlModelDeltaRangeC?: [number, number] | null;
    confidence?: string;
    confidenceNote?: string;
    warning?: string;
  };
  validation: ValidationReport;
}

