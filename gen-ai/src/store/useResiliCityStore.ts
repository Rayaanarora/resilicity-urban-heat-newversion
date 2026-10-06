import { create } from 'zustand';
import {
  fetchHeat,
  fetchSegmentation,
  fetchHealth,
  fetchAnalyzeAndRedesign,
  fetchRefineDesign,
} from '../services/api';
import { TEMP_PER_SCORE_POINT, calculateHeatMetrics } from '../services/heatCalculator';
import { SAMPLE_SCENES } from '../services/sampleData';
import type {
  BackendStatus,
  ClimateZone,
  DesignProfile,
  HeatScoreMetrics,
  Intervention,
  SceneAnalysis,
  SpatialDesignPlan,
  SurfaceMask,
  VisualizationOutput,
} from '../types/resilicity';

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
  viewMode: 'slider' | 'side-by-side' | 'fullscreen';
  setViewMode: (mode: 'slider' | 'side-by-side' | 'fullscreen') => void;

  // Canvas Overlay Masks (MASKS OFF by default as per spec)
  segmentationMasks: SurfaceMask[];
  visibleMaskIds: string[];
  isOverlayActive: boolean;
  toggleOverlayActive: () => void;
  segmentationSource: 'segformer' | 'sample' | 'fallback';
  segmentationModelInfo: { name: string; device: string; source: string } | null;
  planSource: 'vlm' | 'rules' | 'sample';
  planSummary: string | null;
  imageDimensions: { width: number; height: number } | null;

  // Generative AI Spatial Design
  designProfile: DesignProfile;
  setDesignProfile: (profile: DesignProfile) => void;
  qualityTier: 'fast' | 'final';
  setQualityTier: (tier: 'fast' | 'final') => void;
  spatialDesignPlan: SpatialDesignPlan | null;
  sceneAnalysis: SceneAnalysis | null;
  visualizationOutput: VisualizationOutput | null;
  generationStage: string | null;
  refinementHistory: Array<{ prompt: string; imageUrl: string; timestamp: number }>;

  // Interventions
  interventions: Intervention[];
  activeInterventionIds: string[];
  climateZone: ClimateZone;

  // Inference & Caching
  isUploadedPlaceholder: boolean;
  isSegmenting: boolean;
  isReasoning: boolean;
  isGenerating: boolean;
  isRefining: boolean;
  apiError: string | null;
  generationCache: Record<string, string>;

  // Metrics
  heatMetrics: HeatScoreMetrics;
  backendStatus: BackendStatus;

  // Actions
  loadSampleScene: (sceneId: string) => void;
  uploadCustomImage: (file: File) => Promise<void>;
  generateResilientDesign: (tierOverride?: 'fast' | 'final') => Promise<void>;
  refineCurrentDesign: (instruction: string) => Promise<void>;
  toggleIntervention: (interventionId: string) => void;
  toggleMaskVisibility: (maskId: string) => void;
  setSliderPosition: (position: number) => void;
  setClimateZone: (zone: ClimateZone) => void;
  setBackendMode: (mode: BackendStatus['mode']) => void;
  refreshModelEstimate: () => Promise<void>;
  checkBackendHealth: () => Promise<void>;
  resetAll: () => void;
}

let estimateRequestId = 0;
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
    viewMode: 'slider',
    setViewMode: (mode) => set({ viewMode: mode }),

    segmentationMasks: initialScene.masks,
    visibleMaskIds: initialScene.masks.map((m) => m.id),
    // MASKS OFF BY DEFAULT: Hero content is Original ↔ AI Resilient Redesign
    isOverlayActive: false,
    toggleOverlayActive: () => set((state) => ({ isOverlayActive: !state.isOverlayActive })),
    segmentationSource: 'sample',
    segmentationModelInfo: { name: 'ResiliCity sample scene', device: 'reference', source: 'calibrated' },
    planSource: 'sample',
    planSummary: initialScene.description,
    imageDimensions: { width: 1024, height: 683 },

    designProfile: 'balanced',
    setDesignProfile: (profile) => set({ designProfile: profile }),
    qualityTier: 'fast',
    setQualityTier: (tier) => set({ qualityTier: tier }),
    spatialDesignPlan: null,
    sceneAnalysis: null,
    visualizationOutput: null,
    generationStage: null,
    refinementHistory: [],

    interventions: initialScene.interventions,
    activeInterventionIds: initialDefaultActiveIds,
    climateZone: 'tropical',

    isUploadedPlaceholder: false,
    isSegmenting: false,
    isReasoning: false,
    isGenerating: false,
    isRefining: false,
    apiError: null,

    generationCache: {
      [[...initialDefaultActiveIds].sort().join('|')]: initialScene.afterImageUrl,
    },

    heatMetrics: initialMetrics,
    backendStatus: {
      isConnected: false,
      endpointUrl: 'http://localhost:8000/api/v1',
      mode: 'local_fastapi',
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
        currentFile: null,
        rawImageUrl: scene.rawImageUrl,
        generatedImageUrl: scene.afterImageUrl,
        segmentationMasks: scene.masks,
        visibleMaskIds: scene.masks.map((m) => m.id),
        isOverlayActive: false,
        interventions: scene.interventions,
        activeInterventionIds: defaultActiveIds,
        heatMetrics: metrics,
        apiError: null,
        isUploadedPlaceholder: false,
        segmentationSource: 'sample',
        segmentationModelInfo: { name: 'ResiliCity reference scene', device: 'reference', source: 'calibrated' },
        planSource: 'sample',
        planSummary: scene.description,
        spatialDesignPlan: null,
        sceneAnalysis: null,
        visualizationOutput: null,
        generationStage: null,
        refinementHistory: [],
        imageDimensions: { width: 1024, height: 683 },
        activeTab: 'analysis',
      });
      void get().refreshModelEstimate();
    },

    uploadCustomImage: async (file: File) => {
      set({ isSegmenting: true, apiError: null, generationStage: 'Analyzing site' });

      try {
        const dataUrl = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });

        // 1. Semantic segmentation via SegFormer
        set({ generationStage: 'Mapping urban surfaces' });
        const segResult = await fetchSegmentation(file);
        const realMasks = segResult.masks;

        // 2. Generate initial spatial plan & rule adaptations
        set({ generationStage: 'Planning cooling interventions' });
        const hasRoof = realMasks.some((m) => m.className === 'roof' && m.areaPercentage > 2);
        const hasRoad = realMasks.some((m) => m.className === 'road' && m.areaPercentage > 5);
        const hasSidewalk = realMasks.some((m) => (m.className === 'pavement' || m.className === 'sidewalk') && m.areaPercentage > 3);

        const customInterventions: Intervention[] = [
          {
            id: 'gen-tree-canopy',
            type: 'tree_canopy',
            title: 'Native Shade Tree Canopy',
            targetRegion: hasSidewalk ? 'sidewalk' : 'pavement',
            priority: 1,
            coverage: 0.65,
            estCostTier: 'med',
            estCostText: 'Medium cost',
            coolingImpact: 1.8,
            description: 'Planted high-canopy native shade trees along pedestrian paths and road verges',
            promptTemplate: 'lush mature native canopy trees with architectural realism casting cooling shadows',
            landCoverShift: { f_built: -0.06, f_tree: 0.06 },
            defaultEnabled: true,
          },
          ...(hasRoad || hasSidewalk ? [{
            id: 'gen-cool-pavement',
            type: 'cool_pavement' as const,
            title: 'Solar-Reflective Cool Pavement',
            targetRegion: 'road' as const,
            priority: 2,
            coverage: 0.75,
            estCostTier: 'med' as const,
            estCostText: 'Medium cost',
            coolingImpact: 1.2,
            literatureCoolingC: 1.0,
            description: 'Light-colored high-albedo permeable pavers and solar reflective coating',
            promptTemplate: 'light gray solar-reflective pavement with preserved lane markings',
            defaultEnabled: true,
          }] : []),
          ...(hasRoof ? [{
            id: 'gen-cool-roof',
            type: 'cool_roof' as const,
            title: 'High-Albedo Cool Roof Membrane',
            targetRegion: 'roof' as const,
            priority: 3,
            coverage: 0.85,
            estCostTier: 'low' as const,
            estCostText: 'Low cost',
            coolingImpact: 1.5,
            literatureCoolingC: 1.6,
            description: 'Ultra-high SRI reflective coating applied across exposed roof surfaces',
            promptTemplate: 'reflective white cool roof coating preserving roof equipment',
            defaultEnabled: true,
          }] : []),
          {
            id: 'gen-shade-canopy',
            type: 'shade_structure',
            title: 'Architectural Tensile Shade Pergola',
            targetRegion: 'pavement',
            priority: 4,
            coverage: 0.40,
            estCostTier: 'low',
            estCostText: 'Low cost',
            coolingImpact: 1.3,
            description: 'Lightweight tensile fabric and timber pergola shade over walkways',
            promptTemplate: 'modern architectural tensile shade canopy with slender timber support posts',
            defaultEnabled: false,
          },
        ];

        const defaultActive = customInterventions.filter((i) => i.defaultEnabled).map((i) => i.id);
        const metrics = calculateHeatMetrics(
          realMasks,
          customInterventions.filter((i) => defaultActive.includes(i.id)),
          get().climateZone
        );

        set({
          currentFile: file,
          currentSceneId: `upload-${Date.now()}`,
          rawImageUrl: dataUrl,
          generatedImageUrl: dataUrl, // Initial display before generation request
          segmentationMasks: realMasks,
          visibleMaskIds: realMasks.map((m) => m.id),
          isOverlayActive: false, // MASKS OFF by default
          interventions: customInterventions,
          activeInterventionIds: defaultActive,
          heatMetrics: metrics,
          isSegmenting: false,
          isUploadedPlaceholder: false,
          segmentationSource: 'segformer',
          segmentationModelInfo: segResult.model,
          planSource: 'rules',
          planSummary: `Site analysis mapped ${realMasks.length} urban surface categories. Ready for generative redesign.`,
          imageDimensions: segResult.image,
          generationStage: null,
          apiError: null,
          activeTab: 'analysis',
        });

        void get().refreshModelEstimate();
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Failed to analyze uploaded photo.';
        set({
          isSegmenting: false,
          generationStage: null,
          apiError: `Site analysis failed: ${message}`,
        });
      }
    },

    generateResilientDesign: async (tierOverride?: 'fast' | 'final') => {
      const {
        currentFile,
        rawImageUrl,
        designProfile,
        qualityTier,
        interventions,
        activeInterventionIds,
        currentSceneId,
      } = get();

      const activeList = interventions.filter((i) => activeInterventionIds.includes(i.id));
      const requestedTypes = activeList.map((i) => i.type);
      const selectedTier = tierOverride || qualityTier;

      set({ isGenerating: true, apiError: null, generationStage: 'Analyzing site' });

      try {
        // Resolve file to send
        let fileToSend = currentFile;
        if (!fileToSend && rawImageUrl) {
          const resp = await fetch(rawImageUrl);
          const blob = await resp.blob();
          fileToSend = new File([blob], `${currentSceneId || 'street'}.jpg`, { type: blob.type || 'image/jpeg' });
        }

        if (!fileToSend) {
          throw new Error('Please select or upload a scene photo before generating design.');
        }

        // Stepped user-friendly stages
        await new Promise((r) => setTimeout(r, 200));
        set({ generationStage: 'Mapping urban surfaces' });

        await new Promise((r) => setTimeout(r, 200));
        set({ generationStage: 'Planning cooling interventions' });

        await new Promise((r) => setTimeout(r, 200));
        set({ generationStage: 'Generating resilient redesign' });

        const result = await fetchAnalyzeAndRedesign(
          fileToSend,
          designProfile,
          requestedTypes,
          selectedTier
        );

        set({ generationStage: 'Validating result' });
        await new Promise((r) => setTimeout(r, 200));

        // Update spatial plan
        if (result.design_plan) {
          set({
            spatialDesignPlan: result.design_plan,
            planSummary: result.design_plan.site_summary,
            planSource: 'vlm',
          });

          // Enrich interventions with AI planner details (why, feasibility, coverage)
          if (result.design_plan.interventions && result.design_plan.interventions.length > 0) {
            const planSpecs = result.design_plan.interventions;
            const updatedInterventions = get().interventions.map((item) => {
              const matchedSpec = planSpecs.find((s) => s.type === item.type);
              if (matchedSpec) {
                return {
                  ...item,
                  description: matchedSpec.visual_design || item.description,
                  coverage: matchedSpec.coverage || item.coverage,
                  coolingImpact: matchedSpec.cooling_impact_c || item.coolingImpact,
                  // Custom enriched fields
                  reason: matchedSpec.reason,
                  feasibility: matchedSpec.feasibility,
                  placement: matchedSpec.placement,
                };
              }
              return item;
            });
            set({ interventions: updatedInterventions });
          }
        }

        if (result.scene_analysis) {
          set({ sceneAnalysis: result.scene_analysis });
        }

        // Update visualization
        if (result.visualization.status === 'ready' && result.visualization.image_url) {
          const newUrl = result.visualization.image_url;
          const cacheKey = `${fileToSend.name}_${designProfile}_${selectedTier}_${requestedTypes.sort().join('_')}`;

          set((state) => ({
            generatedImageUrl: newUrl,
            visualizationOutput: result.visualization,
            generationCache: {
              ...state.generationCache,
              [cacheKey]: newUrl,
            },
          }));
        } else {
          // Clean fallback showing "Visualization unavailable" without fake overlays
          set({
            visualizationOutput: result.visualization,
          });
        }

        // Update thermal impact numbers
        if (result.thermal_impact) {
          const deltaC = result.thermal_impact.totalCoolingReductionC;
          set((state) => ({
            heatMetrics: {
              ...state.heatMetrics,
              estimatedTempReductionC: deltaC,
              projectedScore: round1(Math.max(1, state.heatMetrics.baseScore - deltaC * 0.7)),
              deltaScore: round1(deltaC * 0.7),
              tempSource: 'model+assumed',
            },
          }));
        }

        set({
          isGenerating: false,
          generationStage: 'Design complete',
        });

        setTimeout(() => {
          set({ generationStage: null });
        }, 1800);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Generative redesign failed.';
        set({
          isGenerating: false,
          generationStage: null,
          apiError: msg,
        });
      }
    },

    refineCurrentDesign: async (instruction: string) => {
      const { generatedImageUrl, rawImageUrl, qualityTier, refinementHistory } = get();
      const targetUrl = generatedImageUrl || rawImageUrl;
      if (!targetUrl) {
        set({ apiError: 'No generated design available to refine.' });
        return;
      }

      set({ isRefining: true, apiError: null, generationStage: 'Refining image' });

      try {
        const resp = await fetch(targetUrl);
        const blob = await resp.blob();

        const result = await fetchRefineDesign(blob, instruction, qualityTier);

        if (result.status === 'ready' && result.image_url) {
          const nextUrl = result.image_url;
          const nextViz: VisualizationOutput = {
            status: 'ready',
            image_url: nextUrl,
            width: result.width,
            height: result.height,
            provider: result.provider || 'gemini',
            model: result.model || 'gemini-3.1-flash-image',
            quality_tier: qualityTier,
            generation_time_ms: 0,
            refinement_count: refinementHistory.length + 1,
          };
          set({
            generatedImageUrl: nextUrl,
            visualizationOutput: nextViz,
            refinementHistory: [
              ...refinementHistory,
              { prompt: instruction, imageUrl: nextUrl, timestamp: Date.now() },
            ],
            generationStage: 'Design complete',
          });
        } else {
          set({
            apiError: 'Refinement was unavailable.',
          });
        }

        set({ isRefining: false });
        setTimeout(() => set({ generationStage: null }), 1800);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Design refinement failed.';
        set({
          isRefining: false,
          generationStage: null,
          apiError: msg,
        });
      }
    },

    toggleIntervention: (interventionId: string) => {
      // Per spec: DO NOT trigger image generation on checkbox toggle.
      // Simply update selected interventions and numerical thermal metrics.
      // Image generation is initiated explicitly by pressing [ GENERATE RESILIENT DESIGN ].
      const { activeInterventionIds, interventions, segmentationMasks, climateZone } = get();

      const nextActiveIds = activeInterventionIds.includes(interventionId)
        ? activeInterventionIds.filter((id) => id !== interventionId)
        : [...activeInterventionIds, interventionId];

      const activeInterventionsList = interventions.filter((i) => nextActiveIds.includes(i.id));
      const nextMetrics = calculateHeatMetrics(segmentationMasks, activeInterventionsList, climateZone);

      set({
        activeInterventionIds: nextActiveIds,
        heatMetrics: nextMetrics,
      });

      void get().refreshModelEstimate();
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

      const assumedCoolingC = round1(
        otherIvs.reduce((sum, i) => sum + (i.literatureCoolingC ?? i.coolingImpact * TEMP_PER_SCORE_POINT), 0)
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
        })
      );

      try {
        const res = await fetchHeat(heatMetrics.surfaceComposition, shifts);
        if (myId !== estimateRequestId) return;
        const modelOk = res.reliable !== false;
        const fallbackC = round1(
          modelIvs.reduce((sum, i) => sum + (i.literatureCoolingC ?? i.coolingImpact * TEMP_PER_SCORE_POINT), 0)
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
            generativeProvider: health.generative_provider,
            geminiConfigured: health.gemini_configured,
            geminiImageModel: health.gemini_image_model,
            geminiFinalModel: health.gemini_final_model,
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
