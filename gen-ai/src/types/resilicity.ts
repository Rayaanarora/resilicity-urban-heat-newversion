export type SurfaceClass = 'roof' | 'road' | 'vegetation' | 'pavement' | 'water' | 'wall' | 'sky' | 'other';

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
  | 'green_roof' 
  | 'permeable_pave';

export type CostTier = 'low' | 'med' | 'high';

export interface Intervention {
  id: string;
  type: InterventionType;
  title: string;
  targetRegion: SurfaceClass | 'sidewalk' | 'courtyard';
  priority: number;
  estCostTier: CostTier;
  estCostText: string;
  coolingImpact: number; // Temperature score reduction e.g. 1.4
  description: string;
  promptTemplate: string;
  defaultEnabled?: boolean;
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
}
