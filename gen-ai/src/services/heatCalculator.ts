import type { ClimateZone, HeatScoreMetrics, Intervention, SurfaceClass, SurfaceMask } from '../types/resilicity';

// Standard surface heat impact factors (higher = hotter surface microclimate)
const SURFACE_HEAT_FACTORS: Record<SurfaceClass, { factor: number; defaultAlbedo: number }> = {
  road: { factor: 9.5, defaultAlbedo: 0.08 },
  roof: { factor: 8.8, defaultAlbedo: 0.12 },
  pavement: { factor: 6.8, defaultAlbedo: 0.25 },
  sidewalk: { factor: 6.5, defaultAlbedo: 0.25 },
  wall: { factor: 5.2, defaultAlbedo: 0.30 },
  water: { factor: 2.5, defaultAlbedo: 0.10 },
  vegetation: { factor: 1.0, defaultAlbedo: 0.25 },
  sky: { factor: 0.0, defaultAlbedo: 0.50 },
  other: { factor: 4.0, defaultAlbedo: 0.20 },
};

// ASSUMPTION (not measured): approximate surface-temperature change per score point.
// Used only as a fallback (backend offline) and for measures the ML model cannot see (cool roof / pavement).
export const TEMP_PER_SCORE_POINT = 0.85;

export function exposureLabel(score: number): string {
  if (score >= 8) return 'Very high exposure';
  if (score >= 6) return 'High exposure';
  if (score >= 4) return 'Moderate exposure';
  return 'Low exposure';
}

// Climate zone baseline multipliers
const CLIMATE_MULTIPLIERS: Record<ClimateZone, number> = {
  arid: 1.15,
  tropical: 1.05,
  mediterranean: 1.0,
  temperate: 0.88,
};

export function calculateHeatMetrics(
  masks: SurfaceMask[],
  activeInterventions: Intervention[],
  climateZone: ClimateZone = 'tropical'
): HeatScoreMetrics {
  const surfaceComposition: Record<SurfaceClass, number> = {
    road: 0,
    roof: 0,
    pavement: 0,
    sidewalk: 0,
    wall: 0,
    water: 0,
    vegetation: 0,
    sky: 0,
    other: 0,
  };

  let totalCoveredPercentage = 0;

  masks.forEach((m) => {
    surfaceComposition[m.className] = (surfaceComposition[m.className] || 0) + m.areaPercentage;
    totalCoveredPercentage += m.areaPercentage;
  });

  let isReliable = true;
  let fallbackReason: string | undefined = undefined;

  if (masks.length < 2 || totalCoveredPercentage < 50) {
    isReliable = false;
    fallbackReason = totalCoveredPercentage < 50 
      ? `Incomplete surface segmentation (Only ${Math.round(totalCoveredPercentage)}% area identified).`
      : 'Insufficient surface class diversity (<2 distinct classes detected).';
  }

  let rawWeightedSum = 0;
  Object.entries(surfaceComposition).forEach(([cls, pct]) => {
    const surface = cls as SurfaceClass;
    const factorInfo = SURFACE_HEAT_FACTORS[surface] || { factor: 5.0, defaultAlbedo: 0.2 };
    
    const matchingMask = masks.find((m) => m.className === surface);
    const albedo = matchingMask?.albedo ?? factorInfo.defaultAlbedo;
    
    const effectiveFactor = factorInfo.factor * (1 - (albedo - 0.1) * 0.5);
    rawWeightedSum += (pct / 100) * effectiveFactor;
  });

  const climateMult = CLIMATE_MULTIPLIERS[climateZone] || 1.0;
  const baseScore = Math.min(10, Math.max(0.5, Number((rawWeightedSum * climateMult).toFixed(1))));

  let surfaceCooling = 0;
  let shadeCooling = 0;
  activeInterventions.forEach((intervention) => {
    if (intervention.type === 'tree_canopy' || intervention.type === 'shade_structure') {
      shadeCooling += intervention.coolingImpact;
    } else {
      surfaceCooling += intervention.coolingImpact;
    }
  });
  const totalCoolingImpact = surfaceCooling + shadeCooling;

  const projectedScore = Math.min(
    baseScore,
    Math.max(0.5, Number((baseScore - totalCoolingImpact).toFixed(1)))
  );

  const deltaScore = Number((baseScore - projectedScore).toFixed(1));
  const estimatedTempReductionC = Number((deltaScore * TEMP_PER_SCORE_POINT).toFixed(1));

  return {
    baseScore,
    projectedScore,
    deltaScore,
    surfaceCooling: Number(surfaceCooling.toFixed(1)),
    shadeCooling: Number(shadeCooling.toFixed(1)),
    estimatedTempReductionC,
    modelCoolingC: 0,
    assumedCoolingC: estimatedTempReductionC,
    tempSource: 'assumed',
    isReliable,
    fallbackReason,
    surfaceComposition,
  };
}
