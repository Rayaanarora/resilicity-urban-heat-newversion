import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import {
  RotateCw,
  Sliders,
  Maximize2,
  Minimize2,
  Columns,
  Download,
  Sparkles,
  AlertCircle,
  Trees,
  Layers,
  CloudSun,
  Cpu,
} from 'lucide-react';

export const BeforeAfterPreview: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [naturalAspect, setNaturalAspect] = useState<number>(3 / 2);

  const {
    rawImageUrl,
    generatedImageUrl,
    sliderPosition,
    setSliderPosition,
    viewMode,
    setViewMode,
    isGenerating,
    generationStage,
    visualizationOutput,
    spatialDesignPlan,
    interventions,
    activeInterventionIds,
  } = useResiliCityStore();

  // Load natural aspect ratio of the raw image to prevent distortion
  useEffect(() => {
    if (!rawImageUrl) return;
    const img = new Image();
    img.src = rawImageUrl;
    img.onload = () => {
      if (img.naturalWidth && img.naturalHeight) {
        setNaturalAspect(img.naturalWidth / img.naturalHeight);
      }
    };
  }, [rawImageUrl]);

  const handleMove = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
      setSliderPosition(percentage);
    },
    [setSliderPosition]
  );

  const onMouseDown = () => setIsDragging(true);
  const onMouseUp = () => setIsDragging(false);

  const onMouseMove = (e: React.MouseEvent) => {
    if (isDragging) handleMove(e.clientX);
  };

  const onTouchMove = (e: React.TouchEvent) => {
    if (e.touches[0]) handleMove(e.touches[0].clientX);
  };

  const handleDownload = () => {
    const targetUrl = generatedImageUrl || rawImageUrl;
    if (!targetUrl) return;
    const a = document.createElement('a');
    a.href = targetUrl;
    a.download = `resilicity-design-${Date.now()}.png`;
    a.click();
  };

  const isUnavailable = visualizationOutput?.status === 'unavailable';
  const hasGeneratedDesign = Boolean(generatedImageUrl);

  // Strategy summary lines from spatial design plan or active interventions
  const activeList = interventions.filter((i) => activeInterventionIds.includes(i.id));

  return (
    <div
      className={`rc-card p-6 flex flex-col gap-5 transition-all duration-300 ${
        isFullscreen ? 'fixed inset-0 z-50 rounded-none bg-slate-950 p-8 overflow-y-auto' : ''
      }`}
    >
      {/* Header and Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="rc-card-header-label">HERO GENERATIVE VISUALIZATION</span>
            <span className="text-[10px] font-mono uppercase px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-800 font-bold border border-emerald-200 flex items-center gap-1">
              <Cpu className="w-3 h-3 text-emerald-600" />
              Local SDXL Inpainting · RTX 3050
            </span>
          </div>
          <h3
            className={`text-xl font-bold font-sans ${
              isFullscreen ? 'text-white' : 'text-slate-900'
            }`}
          >
            Resilient Urban Redesign
          </h3>
        </div>

        {/* View Controls & Action Toolbar */}
        <div className="flex items-center flex-wrap gap-2">
          {/* Mode Switcher */}
          <div className="flex items-center bg-slate-100 p-1 rounded-xl border border-slate-200/80">
            <button
              onClick={() => setViewMode('slider')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                viewMode === 'slider'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Split Interactive Slider"
            >
              <Sliders className="w-3.5 h-3.5" />
              <span>Slider</span>
            </button>
            <button
              onClick={() => setViewMode('side-by-side')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                viewMode === 'side-by-side'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
              title="Side-by-Side Dual View"
            >
              <Columns className="w-3.5 h-3.5" />
              <span>Side-by-Side</span>
            </button>
          </div>

          {/* Reset Slider */}
          {viewMode === 'slider' && (
            <button
              onClick={() => setSliderPosition(50)}
              className="p-2 border border-slate-200 bg-white rounded-xl hover:bg-slate-50 text-slate-600 transition-colors"
              title="Reset slider position to 50%"
            >
              <RotateCw className="w-4 h-4" />
            </button>
          )}

          {/* Download Image */}
          <button
            onClick={handleDownload}
            disabled={!hasGeneratedDesign}
            className="p-2 border border-slate-200 bg-white rounded-xl hover:bg-slate-50 disabled:opacity-40 text-slate-600 transition-colors"
            title="Download resilient design photograph"
          >
            <Download className="w-4 h-4" />
          </button>

          {/* Fullscreen Toggle */}
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className={`p-2 border rounded-xl transition-colors ${
              isFullscreen
                ? 'bg-slate-800 border-slate-700 text-white'
                : 'bg-white border-slate-200 hover:bg-slate-50 text-slate-600'
            }`}
            title={isFullscreen ? 'Exit Fullscreen' : 'View Fullscreen'}
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Unavailable State Notice (Requirement 19: Clear Real Error, No Silent Fallback) */}
      {isUnavailable && !isGenerating && (
        <div className="bg-rose-50 border border-rose-300 rounded-xl p-4 flex items-start gap-3 text-rose-900 text-sm">
          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div className="flex flex-col gap-1">
            <span className="font-bold">AI redesign unavailable</span>
            <span className="text-xs text-rose-800">
              {visualizationOutput?.error_message ||
                'Local SDXL Inpainting could not complete generation on this photograph. The original image is not shown as a fake result.'}
            </span>
          </div>
        </div>
      )}

      {/* Main Visual Display (Aspect Ratio Preserved, Clean Real Photography) */}
      <div
        className="w-full relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-200/80 shadow-md transition-all flex items-center justify-center min-h-[380px]"
        style={{
          aspectRatio: `${naturalAspect}`,
          maxHeight: isFullscreen ? '78vh' : '640px',
        }}
      >
        {viewMode === 'slider' ? (
          /* Interactive Before / After Split Slider */
          <div
            ref={containerRef}
            className="relative w-full h-full select-none touch-none cursor-ew-resize group"
            onMouseDown={onMouseDown}
            onMouseUp={onMouseUp}
            onMouseLeave={onMouseUp}
            onMouseMove={onMouseMove}
            onTouchMove={onTouchMove}
          >
            {/* Raw Original Photo */}
            <img
              src={rawImageUrl || '/samples/sample_1_dense_urban.jpg'}
              alt="Original street photograph"
              className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            />

            {/* AI-Generated Resilient Redesign Image (Right Side) */}
            <div
              className="absolute inset-0 overflow-hidden pointer-events-none z-20"
              style={{ clipPath: `inset(0 0 0 ${sliderPosition}%)` }}
            >
              {hasGeneratedDesign ? (
                <img
                  src={generatedImageUrl!}
                  alt="AI Resilient Redesign photograph"
                  className="absolute inset-0 w-full h-full object-contain"
                />
              ) : (
                <div className="absolute inset-0 bg-slate-900/95 flex flex-col items-center justify-center p-6 text-center text-slate-300 gap-3">
                  <AlertCircle className="w-8 h-8 text-amber-500/80" />
                  <div className="flex flex-col gap-1 max-w-sm">
                    <span className="font-bold text-white text-sm">
                      {isGenerating ? 'Generating Redesign...' : 'AI Redesign Unavailable'}
                    </span>
                    <span className="text-xs text-slate-400">
                      {isGenerating
                        ? 'Diffusion inpainting in progress on GPU.'
                        : 'Upload an urban photograph to generate an autonomous redesign.'}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Labels */}
            <div className="absolute top-4 left-4 bg-slate-900/85 backdrop-blur-md border border-slate-700 text-white px-3 py-1.5 rounded-lg text-xs font-mono font-bold tracking-wide shadow-md z-30">
              ORIGINAL PHOTOGRAPH
            </div>

            <div className="absolute top-4 right-4 bg-[#0d7a5f]/95 backdrop-blur-md border border-emerald-400/40 text-white px-3 py-1.5 rounded-lg text-xs font-mono font-bold tracking-wider uppercase shadow-md flex items-center gap-1.5 z-30">
              <Sparkles className="w-3.5 h-3.5 text-emerald-300" />
              <span>AI RESILIENT REDESIGN</span>
            </div>

            {/* Divider Handle */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-white shadow-2xl pointer-events-none z-30"
              style={{ left: `${sliderPosition}%` }}
            >
              <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 left-1/2 w-9 h-9 rounded-full bg-[#0d7a5f] border-2 border-white shadow-2xl flex items-center justify-center text-white">
                <Sliders className="w-4 h-4 rotate-90" />
              </div>
            </div>
          </div>
        ) : (
          /* Side-by-Side Dual View */
          <div className="grid grid-cols-2 w-full h-full gap-3 p-3">
            <div className="relative w-full h-full rounded-xl overflow-hidden bg-slate-900 flex items-center justify-center border border-slate-800">
              <img
                src={rawImageUrl || '/samples/sample_1_dense_urban.jpg'}
                alt="Original Photograph"
                className="w-full h-full object-contain"
              />
              <span className="absolute top-3 left-3 bg-slate-900/90 text-white font-mono text-xs px-2.5 py-1 rounded-md border border-slate-700">
                ORIGINAL PHOTOGRAPH
              </span>
            </div>
            <div className="relative w-full h-full rounded-xl overflow-hidden bg-slate-900 flex items-center justify-center border border-emerald-900/40">
              {hasGeneratedDesign ? (
                <img
                  src={generatedImageUrl!}
                  alt="AI Resilient Redesign"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center text-slate-400 gap-2">
                  <AlertCircle className="w-8 h-8 text-amber-500/80" />
                  <span className="font-semibold text-xs text-slate-300">
                    {isGenerating ? 'Generating Redesign...' : 'AI Redesign Unavailable'}
                  </span>
                </div>
              )}
              <span className="absolute top-3 left-3 bg-[#0d7a5f]/90 text-white font-mono text-xs px-2.5 py-1 rounded-md border border-emerald-400/40 flex items-center gap-1.5">
                <Sparkles className="w-3 h-3 text-emerald-300" />
                AI RESILIENT REDESIGN
              </span>
            </div>
          </div>
        )}

        {/* Progress Overlay during Autonomous Generation */}
        {isGenerating && (
          <div className="absolute inset-0 bg-slate-950/85 backdrop-blur-md flex flex-col items-center justify-center gap-4 text-white font-sans z-40 p-6 text-center">
            <div className="relative">
              <div className="w-14 h-14 border-4 border-emerald-500/20 border-t-emerald-400 rounded-full animate-spin" />
              <Sparkles className="w-6 h-6 text-emerald-400 absolute inset-0 m-auto animate-pulse" />
            </div>

            <div className="flex flex-col items-center gap-1 max-w-md">
              <span className="text-lg font-bold text-white font-sans tracking-wide">
                {generationStage || 'GENERATING RESILIENT REDESIGN'}
              </span>
              <p className="text-xs text-slate-300">
                Local SDXL inference running on RTX 3050 6GB. Grounding interventions into detected urban surfaces.
              </p>
            </div>

            {/* Stepper Dots */}
            <div className="flex items-center gap-2 mt-2">
              {[
                'ANALYZING SITE',
                'PLANNING RESILIENCE STRATEGY',
                'MAPPING INTERVENTIONS',
                'GENERATING LOCALLY',
                'VALIDATING RESULT',
              ].map((step, idx) => {
                const isCurrent = generationStage === step;
                return (
                  <div
                    key={idx}
                    className={`h-1.5 rounded-full transition-all duration-300 ${
                      isCurrent ? 'w-8 bg-emerald-400' : 'w-2 bg-slate-700'
                    }`}
                    title={step}
                  />
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* AI DESIGN STRATEGY (Requirement 22: Explanatory, Informational Summary) */}
      <div className="border border-slate-200 rounded-2xl p-5 bg-white shadow-xs flex flex-col gap-3">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
            <span className="text-xs font-bold text-slate-900 uppercase tracking-wider font-mono">
              AI Design Strategy
            </span>
          </div>
          <span className="text-[11px] font-mono text-emerald-700 font-semibold bg-emerald-50 px-2.5 py-0.5 rounded-md border border-emerald-200">
            Autonomous AI Decisions
          </span>
        </div>

        <p className="text-xs text-slate-700 leading-relaxed font-sans">
          {spatialDesignPlan?.overall_design_intent ||
            spatialDesignPlan?.site_summary ||
            'ResiliCity autonomously analyzes site geometry and surface composition, generating microclimate cooling interventions tailored to this physical corridor.'}
        </p>

        {/* Autonomous Interventions List */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
          {interventions.map((iv) => {
            const anyIv = iv as any;
            return (
              <div
                key={iv.id}
                className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 flex flex-col gap-2 text-xs font-sans"
              >
                <div className="flex items-start gap-2.5">
                  <div className="w-7 h-7 rounded-lg bg-emerald-100/70 border border-emerald-300 flex items-center justify-center shrink-0 mt-0.5 text-emerald-800">
                    {iv.type === 'tree_canopy' ? (
                      <Trees className="w-4 h-4" />
                    ) : iv.type === 'cool_roof' || iv.type === 'green_roof' ? (
                      <CloudSun className="w-4 h-4" />
                    ) : (
                      <Layers className="w-4 h-4" />
                    )}
                  </div>
                  <div className="flex flex-col min-w-0">
                    <span className="font-bold text-slate-900">{iv.title}</span>
                    <span className="text-[10px] font-mono text-emerald-700 font-semibold">
                      Target: {iv.targetRegion} · -{iv.coolingImpact.toFixed(1)}°C expected cooling
                    </span>
                  </div>
                </div>

                {anyIv.reason && (
                  <p className="text-[11px] text-slate-700 leading-snug">
                    <strong className="text-slate-900">Reason:</strong> {anyIv.reason}
                  </p>
                )}
                {anyIv.placement && (
                  <p className="text-[11px] text-slate-600 leading-snug">
                    <strong className="text-slate-900">Placement:</strong> {anyIv.placement}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer Info */}
      <div className="flex flex-wrap items-center justify-between gap-4 text-xs font-mono text-slate-500 pt-1">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-rose-500" />
            <span>Original photograph</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>AI Resilient Redesign · preserved perspective & architecture</span>
          </div>
        </div>

        <span className="text-[11px] text-slate-400">
          Autonomous Generative AI · Local SDXL 1.0 Inpainting on RTX 3050 6GB
        </span>
      </div>
    </div>
  );
};
