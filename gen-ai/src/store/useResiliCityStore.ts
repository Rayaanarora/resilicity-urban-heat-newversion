import { create } from 'zustand';
import {
  fetchHeat,
  fetchSegmentation,
  fetchHealth,
  fetchAnalyzeAndRedesign,
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

  // Autonomous Generative AI Spatial Design
  designProfile: DesignProfile;
  setDesignProfile: (profile: DesignProfile) => void;
  qualityTier: 'fast' | 'final';
  setQualityTier: (tier: 'fast' | 'final') => void;
  spatialDesignPlan: SpatialDesignPlan | null;
  sceneAnalysis: SceneAnalysis | null;
  visualizationOutput: VisualizationOutput | null;
  generationStage: string | null;

  // Interventions (Autonomous AI Selected - Informational)
  interventions: Intervention[];
  activeInterventionIds: string[];
  climateZone: ClimateZone;

  // Inference State & Caching
  isUploadedPlaceholder: boolean;
  isSegmenting: boolean;
  isGenerating: boolean;
  apiError: string | null;
  generationCache: Record<string, string>;

  // Metrics
  heatMetrics: HeatScoreMetrics;
  backendStatus: BackendStatus;

  // Actions
  loadSampleScene: (sceneId: string) => void;
  uploadCustomImage: (file: File) => Promise<void>;
  generateResilientDesign: (tierOverride?: 'fast' | 'final') => Promise<void>;
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

    interventions: initialScene.interventions,
    activeInterventionIds: initialDefaultActiveIds,
    climateZone: 'tropical',

    isUploadedPlaceholder: false,
    isSegmenting: false,
    isGenerating: false,
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
        imageDimensions: { width: 1024, height: 683 },
        activeTab: 'analysis',
      });
      void get().refreshModelEstimate();
    },

    uploadCustomImage: async (file: File) => {
      set({
        isSegmenting: true,
        apiError: null,
        generationStage: 'ANALYZING SITE',
        generatedImageUrl: null, // Generated image must be null until successful generation
        visualizationOutput: null,
      });

      try {
        const dataUrl = await new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });

        // 1. Semantic segmentation via SegFormer
        set({ generationStage: 'MAPPING INTERVENTIONS' });
        const segResult = await fetchSegmentation(file);
        const realMasks = segResult.masks;

        set({
          currentFile: file,
          currentSceneId: `upload-${Date.now()}`,
          rawImageUrl: dataUrl,
          generatedImageUrl: null,
          visualizationOutput: null,
          segmentationMasks: realMasks,
          visibleMaskIds: realMasks.map((m) => m.id),
          isOverlayActive: false, // MASKS OFF by default
          isSegmenting: false,
          isUploadedPlaceholder: false,
          segmentationSource: 'segformer',
          segmentationModelInfo: segResult.model,
          planSource: 'vlm',
          planSummary: `Site analysis mapped ${realMasks.length} urban surface categories. Autonomous resilient redesign generating...`,
          imageDimensions: segResult.image,
          apiError: null,
          activeTab: 'analysis',
        });

        // Trigger automatic generation immediately (Requirement 18)
        void get().generateResilientDesign();
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
        currentSceneId,
      } = get();

      const selectedTier = tierOverride || qualityTier;

      set({
        isGenerating: true,
        apiError: null,
        generationStage: 'ANALYZING SITE',
      });

      try {
        // Resolve file to send
        let fileToSend = currentFile;
        if (!fileToSend && rawImageUrl) {
          const resp = await fetch(rawImageUrl);
          const blob = await resp.blob();
          fileToSend = new File([blob], `${currentSceneId || 'street'}.jpg`, { type: blob.type || 'image/jpeg' });
        }

        if (!fileToSend) {
          throw new Error('Please select or upload a street photo to generate design.');
        }

        // Stepped user-friendly stages (Requirement 21)
        await new Promise((r) => setTimeout(r, 200));
        set({ generationStage: 'PLANNING RESILIENCE STRATEGY' });

        await new Promise((r) => setTimeout(r, 250));
        set({ generationStage: 'MAPPING INTERVENTIONS' });

        await new Promise((r) => setTimeout(r, 250));
        set({ generationStage: 'GENERATING LOCALLY' });

        // Call autonomous redesign pipeline (NO manual requested_interventions)
        const result = await fetchAnalyzeAndRedesign(
          fileToSend,
          designProfile,
          undefined, // Autonomous: planner decides interventions
          selectedTier
        );

        set({ generationStage: 'VALIDATING RESULT' });
        await new Promise((r) => setTimeout(r, 200));

        // Update spatial plan and interventions
        if (result.design_plan) {
          set({
            spatialDesignPlan: result.design_plan,
            planSummary: result.design_plan.site_summary,
            planSource: 'vlm',
          });

          if (result.design_plan.interventions && result.design_plan.interventions.length > 0) {
            const mappedInterventions: Intervention[] = result.design_plan.interventions.map((spec, idx) => ({
              id: `spec-${spec.type}-${idx}`,
              type: spec.type as any,
              title: spec.title || spec.type.replace('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
              targetRegion: spec.target_region as any,
              priority: spec.priority,
              coverage: spec.coverage,
              estCostTier: 'med',
              estCostText: 'Medium cost',
              coolingImpact: spec.cooling_impact_c || 1.0,
              description: spec.visual_design || spec.reason,
              promptTemplate: spec.visual_design,
              defaultEnabled: true,
              // Extra enriched fields
              ...(spec as any),
            }));

            set({
              interventions: mappedInterventions,
              activeInterventionIds: mappedInterventions.map((i) => i.id),
            });
          }
        }

        if (result.scene_analysis) {
          set({ sceneAnalysis: result.scene_analysis });
        }

        // Update visualization
        if (result.visualization.status === 'ready' && result.visualization.image_url) {
          const newUrl = result.visualization.image_url;
          const cacheKey = `${fileToSend.name}_${designProfile}_${selectedTier}`;

          set((state) => ({
            generatedImageUrl: newUrl,
            visualizationOutput: result.visualization,
            generationCache: {
              ...state.generationCache,
              [cacheKey]: newUrl,
            },
          }));
        } else {
          // If generation failed or was rejected by validation, keep generatedImageUrl as null!
          set({
            generatedImageUrl: null,
            visualizationOutput: result.visualization,
            apiError: result.visualization.error_message || 'Generated image could not pass validation.',
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
        }, 1500);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Autonomous generative redesign failed.';
        set({
          isGenerating: false,
          generationStage: null,
          generatedImageUrl: null,
          apiError: msg,
        });
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
