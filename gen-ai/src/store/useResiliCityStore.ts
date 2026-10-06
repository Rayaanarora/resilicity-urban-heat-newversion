import { create } from 'zustand';
import { fetchHeat, fetchPlan, fetchSegmentation, fetchHealth, fetchInpaint } from '../services/api';
import { TEMP_PER_SCORE_POINT, calculateHeatMetrics } from '../services/heatCalculator';
import { SAMPLE_SCENES } from '../services/sampleData';
import type { BackendStatus, ClimateZone, HeatScoreMetrics, Intervention, SurfaceMask } from '../types/resilicity';

export type NavTab = 'analysis' | 'saved_runs' | 'method_note';

interface ResiliCityStoreState {
  // Navigation
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;

  // Scene & Images
  currentSceneId: string;
  currentFile: File | null;
  rawImageUrl: string | null;
  generatedImageUrl: string | null;
  sliderPosition: number; // 0 to 100

  // Canvas Overlay Masks
  segmentationMasks: SurfaceMask[];
  visibleMaskIds: string[];
  isOverlayActive: boolean;
  toggleOverlayActive: () => void;
  segmentationSource: 'segformer' | 'sample' | 'fallback';
  segmentationModelInfo: { name: string; device: string; source: string } | null;
  planSource: 'vlm' | 'rules' | 'sample';
  planSummary: string | null;
  imageDimensions: { width: number; height: number } | null;

  // Interventions
  interventions: Intervention[];
  activeInterventionIds: string[];
  climateZone: ClimateZone;

  // Inference & Caching
  isUploadedPlaceholder: boolean; // true when masks/after-image are placeholders, not model output
  isSegmenting: boolean;
  isReasoning: boolean;
  isGenerating: boolean;
  apiError: string | null;
  generationCache: Record<string, string>; // cacheKey -> Blob URL or Image URL

  // Metrics
  heatMetrics: HeatScoreMetrics;
  backendStatus: BackendStatus;

  // Actions
  loadSampleScene: (sceneId: string) => void;
  uploadCustomImage: (file: File) => Promise<void>;
  toggleIntervention: (interventionId: string) => void;
  toggleMaskVisibility: (maskId: string) => void;
  setSliderPosition: (position: number) => void;
  setClimateZone: (zone: ClimateZone) => void;
  setBackendMode: (mode: BackendStatus['mode']) => void;
  refreshModelEstimate: () => Promise<void>;
  checkBackendHealth: () => Promise<void>;
  resetAll: () => void;
}

let estimateRequestId = 0; // ignore out-of-order responses when the user clicks quickly
const round1 = (n: number) => Number(n.toFixed(1));

export const useResiliCityStore = create<ResiliCityStoreState>((set, get) => {
  const initialScene = SAMPLE_SCENES[0];
  const initialDefaultActiveIds = initialScene.interventions
    .filter((i) => i.defaultEnabled)
    .map((i) => i.id);

  const initialMetrics = calculateHeatMetrics(
    initialScene.masks,
    initialScene.interventions.filter((i) => initialDefaultActiveIds.includes(i.id)),
    'tropical'
  );

  return {
    activeTab: 'analysis',
    setActiveTab: (tab: NavTab) => set({ activeTab: tab }),

    currentSceneId: initialScene.id,
    currentFile: null,
    rawImageUrl: initialScene.rawImageUrl,
    generatedImageUrl: initialScene.afterImageUrl,
    sliderPosition: 50,

    segmentationMasks: initialScene.masks,
    visibleMaskIds: initialScene.masks.map((m) => m.id),
    isOverlayActive: true,
    toggleOverlayActive: () => set((state) => ({ isOverlayActive: !state.isOverlayActive })),
    segmentationSource: 'sample',
    segmentationModelInfo: { name: 'ResiliCity sample scene', device: 'reference', source: 'calibrated' },
    planSource: 'sample',
    planSummary: null,
    imageDimensions: { width: 1024, height: 1024 },

    interventions: initialScene.interventions,
    activeInterventionIds: initialDefaultActiveIds,
    climateZone: 'tropical',

    isUploadedPlaceholder: false,
    isSegmenting: false,
    isReasoning: false,
    isGenerating: false,
    apiError: null,

    generationCache: {
      [[...initialDefaultActiveIds].sort().join('|')]: initialScene.afterImageUrl,
    },

    heatMetrics: initialMetrics,
    backendStatus: {
      isConnected: false, // set by checkBackendHealth()
      endpointUrl: 'http://localhost:8000/api/v1',
      mode: 'hybrid_demo',
      latencyMs: 120,
    },

    loadSampleScene: (sceneId: string) => {
      const scene = SAMPLE_SCENES.find((s) => s.id === sceneId) || SAMPLE_SCENES[0];
      const defaultActiveIds = scene.interventions
        .filter((i) => i.defaultEnabled)
        .map((i) => i.id);

      const metrics = calculateHeatMetrics(
        scene.masks,
        scene.interventions.filter((i) => defaultActiveIds.includes(i.id)),
        get().climateZone
      );

      set({
        currentSceneId: scene.id,
        rawImageUrl: scene.rawImageUrl,
        generatedImageUrl: scene.afterImageUrl,
        segmentationMasks: scene.masks,
        visibleMaskIds: scene.masks.map((m) => m.id),
        interventions: scene.interventions,
        activeInterventionIds: defaultActiveIds,
        heatMetrics: metrics,
        apiError: null,
        isUploadedPlaceholder: false,
        segmentationSource: 'sample',
        segmentationModelInfo: { name: 'ResiliCity sample scene', device: 'reference', source: 'calibrated' },
        planSource: 'sample',
        planSummary: null,
        imageDimensions: { width: 1024, height: 1024 },
        activeTab: 'analysis',
      });
      void get().refreshModelEstimate();
    },

    uploadCustomImage: async (file: File) => {
      set({ isSegmenting: true, apiError: null });

      try {
        const dataUrl = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });

        // Perform real SegFormer semantic segmentation via backend API
        const segResult = await fetchSegmentation(file);
        const realMasks = segResult.masks;

        // Adapt interventions based on detected surface composition
        const hasRoof = realMasks.some((m) => m.className === 'roof' && m.areaPercentage > 2);
        const hasRoadOrPave = realMasks.some((m) => (m.className === 'road' || m.className === 'pavement') && m.areaPercentage > 5);

        // Stage 3: ask the vision planner; fall back to the rule-based list if it is unavailable.
        const composition: Record<string, number> = {};
        realMasks.forEach((m) => {
          composition[m.className] = (composition[m.className] ?? 0) + m.areaPercentage;
        });
        let planned: Intervention[] | null = null;
        let planSummary: string | null = null;
        try {
          const plan = await fetchPlan(file, composition);
          planned = plan.interventions;
          planSummary = plan.siteSummary || null;
        } catch {
          planned = null; // planner offline / no API key / bad reply: use rules below
        }

        const ruleInterventions: Intervention[] = [
          {
            id: 'c-int-tree',
            type: 'tree_canopy',
            title: 'Tree canopy expansion',
            targetRegion: 'pavement',
            priority: 1,
            coverage: 0.75,
            estCostTier: 'med',
            estCostText: 'Medium cost',
            coolingImpact: 1.2,
            description: 'Native high-shade canopy along pedestrian corridors',
            promptTemplate: 'lush green broad native canopy trees with shaded ground',
            landCoverShift: { f_built: -0.05, f_tree: 0.05 },
            defaultEnabled: true,
          },
          ...(hasRoadOrPave ? [{
            id: 'c-int-pave',
            type: 'cool_pavement' as const,
            title: 'Permeable & reflective pavement',
            targetRegion: 'road' as const,
            priority: 2,
            coverage: 0.80,
            estCostTier: 'med' as const,
            estCostText: 'Medium cost',
            coolingImpact: 0.6,
            literatureCoolingC: 0.9,
            description: 'Cool surface sealant and permeable pavement on walkways',
            promptTemplate: 'light gray high-albedo permeable pavement surface',
            defaultEnabled: true,
          }] : []),
          ...(hasRoof ? [{
            id: 'c-int-roof',
            type: 'cool_roof' as const,
            title: 'High-albedo cool roof coating',
            targetRegion: 'roof' as const,
            priority: 3,
            coverage: 0.85,
            estCostTier: 'low' as const,
            estCostText: 'Low cost',
            coolingImpact: 0.8,
            literatureCoolingC: 1.5,
            description: 'Reflective solar-blocking coating on exposed roof surfaces',
            promptTemplate: 'reflective bright white cool roof coating',
            defaultEnabled: true,
          }] : []),
        ];

        const customInterventions = planned ?? ruleInterventions;
        const defaultActive = customInterventions.filter((i) => i.defaultEnabled).map((i) => i.id);
        const metrics = calculateHeatMetrics(
          realMasks,
          customInterventions.filter((i) => defaultActive.includes(i.id)),
          get().climateZone
        );

        set({
          currentFile: file,
          rawImageUrl: dataUrl,
          generatedImageUrl: dataUrl,
          segmentationMasks: realMasks,
          visibleMaskIds: realMasks.map((m) => m.id),
          interventions: customInterventions,
          activeInterventionIds: defaultActive,
          heatMetrics: metrics,
          isSegmenting: false,
          isUploadedPlaceholder: false,
          segmentationSource: 'segformer',
          segmentationModelInfo: segResult.model,
          planSource: planned ? 'vlm' : 'rules',
          planSummary,
          imageDimensions: segResult.image,
          apiError: null,
          activeTab: 'analysis',
        });
        void get().refreshModelEstimate();

        if (defaultActive.length > 0) {
          const activeList = customInterventions.filter((i) => defaultActive.includes(i.id));
          const polys: Record<string, any> = {};
          for (const m of realMasks) {
            if (m.polygons && m.polygons.length > 0) {
              const cls = m.className || m.id?.replace('seg-', '') || m.label.toLowerCase();
              polys[cls] = (polys[cls] || []).concat(m.polygons);
              polys[m.label] = (polys[m.label] || []).concat(m.polygons);
              if (m.id) polys[m.id] = (polys[m.id] || []).concat(m.polygons);
            }
          }
          fetchInpaint(file, activeList, polys)
            .then((res) => {
              if (res.ok && res.imageUrl) {
                const initKey = [...defaultActive].sort().join('|');
                set((state) => ({
                  generatedImageUrl: res.imageUrl,
                  generationCache: {
                    ...state.generationCache,
                    [initKey]: res.imageUrl,
                  },
                }));
              }
            })
            .catch(() => {});
        }
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Failed to process uploaded image file.';
        set({
          isSegmenting: false,
          apiError: `Segmentation failed: ${message}`,
        });
      }
    },


    toggleIntervention: async (interventionId: string) => {
      const {
        activeInterventionIds,
        interventions,
        segmentationMasks,
        climateZone,
        generationCache,
        rawImageUrl,
        currentFile,
      } = get();

      const nextActiveIds = activeInterventionIds.includes(interventionId)
        ? activeInterventionIds.filter((id) => id !== interventionId)
        : [...activeInterventionIds, interventionId];

      const cacheKey = [...nextActiveIds].sort().join('|');

      const activeInterventionsList = interventions.filter((i) => nextActiveIds.includes(i.id));
      const nextMetrics = calculateHeatMetrics(segmentationMasks, activeInterventionsList, climateZone);

      set({
        activeInterventionIds: nextActiveIds,
        heatMetrics: nextMetrics,
      });
      void get().refreshModelEstimate();

      if (generationCache[cacheKey]) {
        set({ generatedImageUrl: generationCache[cacheKey] });
        return;
      }

      set({ isGenerating: true });

      try {
        let targetUrl = rawImageUrl!;
        if (currentFile && nextActiveIds.length > 0) {
          const polys: Record<string, any> = {};
          for (const m of segmentationMasks) {
            if (m.polygons && m.polygons.length > 0) {
              const cls = m.className || m.id?.replace('seg-', '') || m.label.toLowerCase();
              polys[cls] = (polys[cls] || []).concat(m.polygons);
              polys[m.label] = (polys[m.label] || []).concat(m.polygons);
              if (m.id) polys[m.id] = (polys[m.id] || []).concat(m.polygons);
            }
          }
          const inpaintRes = await fetchInpaint(currentFile, activeInterventionsList, polys);
          if (inpaintRes.ok && inpaintRes.imageUrl) {
            targetUrl = inpaintRes.imageUrl;
          }
        } else {
          await new Promise((res) => setTimeout(res, 300));
          targetUrl = nextActiveIds.length > 0 ? get().generatedImageUrl || rawImageUrl! : rawImageUrl!;
        }

        set((state) => ({
          isGenerating: false,
          generatedImageUrl: targetUrl,
          generationCache: {
            ...state.generationCache,
            [cacheKey]: targetUrl,
          },
        }));
      } catch {
        set({ isGenerating: false });
      }
    },

    toggleMaskVisibility: (maskId: string) => {
      set((state) => ({
        visibleMaskIds: state.visibleMaskIds.includes(maskId)
          ? state.visibleMaskIds.filter((id) => id !== maskId)
          : [...state.visibleMaskIds, maskId],
      }));
    },

    setSliderPosition: (position: number) => {
      set({ sliderPosition: Math.max(0, Math.min(100, position)) });
    },

    setClimateZone: (zone: ClimateZone) => {
      const { segmentationMasks, interventions, activeInterventionIds } = get();
      const activeList = interventions.filter((i) => activeInterventionIds.includes(i.id));
      const metrics = calculateHeatMetrics(segmentationMasks, activeList, zone);
      set({ climateZone: zone, heatMetrics: metrics });
      void get().refreshModelEstimate();
    },

    refreshModelEstimate: async () => {
      const myId = ++estimateRequestId;
      const { interventions, activeInterventionIds, heatMetrics } = get();
      const active = interventions.filter((i) => activeInterventionIds.includes(i.id));
      const modelIvs = active.filter((i) => i.landCoverShift);
      const otherIvs = active.filter((i) => !i.landCoverShift);

      // Measures the regression cannot see (cool roof / pavement): literature value, else the old assumed conversion.
      const assumedCoolingC = round1(
        otherIvs.reduce((sum, i) => sum + (i.literatureCoolingC ?? i.coolingImpact * TEMP_PER_SCORE_POINT), 0),
      );

      if (modelIvs.length === 0) {
        set((st) => ({
          heatMetrics: {
            ...st.heatMetrics,
            modelCoolingC: 0,
            assumedCoolingC,
            estimatedTempReductionC: assumedCoolingC,
            tempSource: 'assumed',
          },
        }));
        return;
      }

      const shifts: Record<string, number> = {};
      modelIvs.forEach((i) =>
        Object.entries(i.landCoverShift ?? {}).forEach(([k, v]) => {
          shifts[k] = (shifts[k] ?? 0) + (v ?? 0);
        }),
      );

      try {
        const res = await fetchHeat(heatMetrics.surfaceComposition, shifts);
        if (myId !== estimateRequestId) return; // a newer request superseded this one
        // Use the regression only when its change is larger than its own error and cools in the right direction.
        const modelOk = res.reliable !== false;
        const fallbackC = round1(
          modelIvs.reduce((sum, i) => sum + (i.literatureCoolingC ?? i.coolingImpact * TEMP_PER_SCORE_POINT), 0),
        );
        const modelCoolingC = modelOk ? round1(Math.max(0, -res.deltaC)) : 0;
        const assumedTotal = modelOk ? assumedCoolingC : round1(assumedCoolingC + fallbackC);
        set((st) => ({
          heatMetrics: {
            ...st.heatMetrics,
            modelCoolingC,
            assumedCoolingC: assumedTotal,
            estimatedTempReductionC: round1(modelCoolingC + assumedTotal),
            tempSource: modelCoolingC > 0 ? (assumedTotal > 0 ? 'model+assumed' : 'model') : 'assumed',
            modelRmseC: res.rmseC,
            modelDeltaC: round1(res.deltaC),
            modelUnreliable: !modelOk,
            outOfRange: res.outOfRange,
          },
          backendStatus: { ...st.backendStatus, isConnected: true, mode: 'local_fastapi' },
        }));
      } catch {
        if (myId !== estimateRequestId) return;
        // Backend offline: keep the rule-based fallback already in heatMetrics.
        set((st) => ({ backendStatus: { ...st.backendStatus, isConnected: false, mode: 'hybrid_demo' } }));
      }
    },

    setBackendMode: (mode) => {
      set((state) => ({
        backendStatus: {
          ...state.backendStatus,
          mode,
        },
      }));
    },

    checkBackendHealth: async () => {
      const health = await fetchHealth();
      if (health) {
        set((st) => ({
          backendStatus: {
            ...st.backendStatus,
            isConnected: health.ok,
            mode: 'local_fastapi',
            segmenterAvailable: health.segmenter_available,
            segmenterModel: health.segmenter_model,
            segmenterDevice: health.segmenter_device,
            segmenterSource: health.segmenter_source,
            plannerAvailable: health.planner_available,
          },
        }));
      } else {
        set((st) => ({
          backendStatus: {
            ...st.backendStatus,
            isConnected: false,
          },
        }));
      }
    },

    resetAll: () => {
      get().loadSampleScene(SAMPLE_SCENES[0].id);
    },
  };
});

// Fetch backend health and initial model estimate once the store exists
void useResiliCityStore.getState().checkBackendHealth();
void useResiliCityStore.getState().refreshModelEstimate();

