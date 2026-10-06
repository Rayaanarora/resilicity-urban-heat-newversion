import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import {
  RotateCw,
  Sliders,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2,
  Columns,
  Download,
  Sparkles,
  Send,
  AlertCircle,
} from 'lucide-react';

const REFINEMENT_CHIPS = [
  'Add two more trees along the sidewalk',
  'Make the pavement lighter and more reflective',
  'Increase pedestrian greenery buffer',
  'Add a modern timber shade pergola',
  'Reduce tree canopy density',
  'Keep existing storefronts and buildings unchanged',
];

export const BeforeAfterPreview: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [refineInput, setRefineInput] = useState('');
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
    isRefining,
    generationStage,
    visualizationOutput,
    refineCurrentDesign,
    refinementHistory,
    segmentationMasks,
    visibleMaskIds,
    isOverlayActive,
    toggleOverlayActive,
    imageDimensions,
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

  const handleRefineSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!refineInput.trim() || isRefining || isGenerating) return;
    void refineCurrentDesign(refineInput.trim());
    setRefineInput('');
  };

  const handleDownload = () => {
    const targetUrl = generatedImageUrl || rawImageUrl;
    if (!targetUrl) return;
    const a = document.createElement('a');
    a.href = targetUrl;
    a.download = `resilicity-design-${Date.now()}.jpg`;
    a.click();
  };

  const imgW = imageDimensions?.width ?? 1024;
  const imgH = imageDimensions?.height ?? 683;

  const isUnavailable = visualizationOutput?.status === 'unavailable';

  return (
    <div
      className={`rc-card p-6 flex flex-col gap-4 transition-all duration-300 ${
        isFullscreen ? 'fixed inset-0 z-50 rounded-none bg-slate-950 p-8 overflow-y-auto' : ''
      }`}
    >
      {/* Header and Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="rc-card-header-label">HERO GENERATIVE VISUALIZATION</span>
            {visualizationOutput?.provider && (
              <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-teal-50 text-teal-700 font-bold border border-teal-200">
                {visualizationOutput.model} · {visualizationOutput.quality_tier}
              </span>
            )}
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

          {/* Mask Overlay Toggle (OFF by default) */}
          <button
            onClick={toggleOverlayActive}
            className={`flex items-center gap-1.5 text-xs font-mono font-semibold px-3 py-1.5 rounded-xl border transition-colors ${
              isOverlayActive
                ? 'text-[#0d7a5f] bg-emerald-50 border-emerald-300'
                : 'text-slate-600 bg-white border-slate-200 hover:bg-slate-50'
            }`}
            title="Toggle SegFormer semantic surface masks (off by default)"
          >
            {isOverlayActive ? (
              <Eye className="w-3.5 h-3.5 text-emerald-600" />
            ) : (
              <EyeOff className="w-3.5 h-3.5 text-slate-400" />
            )}
            <span>{isOverlayActive ? 'Masks: ON' : 'Masks: OFF'}</span>
          </button>

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
            className="p-2 border border-slate-200 bg-white rounded-xl hover:bg-slate-50 text-slate-600 transition-colors"
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

      {/* Unavailable State Notice (if API quota or error occurs) */}
      {isUnavailable && (
        <div className="bg-amber-50 border border-amber-300 rounded-xl p-4 flex items-start gap-3 text-amber-900 text-sm">
          <AlertCircle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="flex flex-col gap-1">
            <span className="font-bold">Visualization unavailable</span>
            <span className="text-xs text-amber-800">
              {visualizationOutput?.error_message ||
                'The generative image editing provider returned a quota or connection limit. The AI spatial design plan and satellite thermal calculations remain fully active below.'}
            </span>
            <span className="text-[11px] text-amber-700 mt-1">
              Note: ResiliCity does not substitute crude procedural overlays or fake images. The AI urban design plan and recommendations below reflect the actual spatial reasoning.
            </span>
          </div>
        </div>
      )}

      {/* Main Visual Display (Aspect Ratio Preserved, No fixed 420px height) */}
      <div
        className="w-full relative rounded-2xl overflow-hidden bg-slate-950 border border-slate-200/80 shadow-md transition-all flex items-center justify-center"
        style={{
          // Use natural aspect ratio to eliminate distortion and letterboxing
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

            {/* Optional Semantic Mask Overlay */}
            {isOverlayActive && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none z-10 transition-opacity duration-300"
                viewBox={`0 0 ${imgW} ${imgH}`}
                preserveAspectRatio="xMidYMid meet"
                style={{
                  clipPath: `inset(0 ${Math.max(0, 100 - sliderPosition)}% 0 0)`,
                }}
              >
                {segmentationMasks
                  .filter((mask) => visibleMaskIds.includes(mask.id))
                  .map((mask) => {
                    const polySets: [number, number][][] =
                      mask.polygons && mask.polygons.length > 0
                        ? mask.polygons
                        : mask.polygonPoints
                        ? [mask.polygonPoints]
                        : [];

                    return polySets.map((poly, pIdx) => {
                      const pointsStr = poly
                        .map(
                          ([xPct, yPct]) =>
                            `${((xPct / 100) * imgW).toFixed(1)},${((yPct / 100) * imgH).toFixed(1)}`
                        )
                        .join(' ');

                      return (
                        <polygon
                          key={`${mask.id}-${pIdx}`}
                          points={pointsStr}
                          fill={mask.color}
                          fillOpacity={0.35}
                          stroke={mask.color}
                          strokeWidth={2}
                          vectorEffect="non-scaling-stroke"
                        >
                          <title>{`${mask.label} (${mask.areaPercentage}%)`}</title>
                        </polygon>
                      );
                    });
                  })}
              </svg>
            )}

            {/* AI-Generated Resilient Redesign Image */}
            <div
              className="absolute inset-0 overflow-hidden pointer-events-none z-20"
              style={{ clipPath: `inset(0 0 0 ${sliderPosition}%)` }}
            >
              <img
                src={
                  isUnavailable
                    ? rawImageUrl || '/samples/sample_1_dense_urban.jpg'
                    : generatedImageUrl || rawImageUrl || '/samples/sample_1_dense_urban.jpg'
                }
                alt="AI Resilient Redesign photograph"
                className="absolute inset-0 w-full h-full object-contain"
              />
            </div>

            {/* Labels */}
            <div className="absolute top-4 left-4 bg-slate-900/85 backdrop-blur-md border border-slate-700 text-white px-3 py-1.5 rounded-lg text-xs font-mono font-bold tracking-wide shadow-md z-30">
              ORIGINAL PHOTO {isOverlayActive ? '· +MASKS' : ''}
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
          <div className="grid grid-cols-2 w-full h-full gap-2 p-2">
            <div className="relative w-full h-full rounded-xl overflow-hidden bg-slate-900 flex items-center justify-center border border-slate-800">
              <img
                src={rawImageUrl || '/samples/sample_1_dense_urban.jpg'}
                alt="Original"
                className="w-full h-full object-contain"
              />
              <span className="absolute top-3 left-3 bg-slate-900/90 text-white font-mono text-xs px-2.5 py-1 rounded-md border border-slate-700">
                ORIGINAL PHOTO
              </span>
            </div>
            <div className="relative w-full h-full rounded-xl overflow-hidden bg-slate-900 flex items-center justify-center border border-emerald-900/40">
              <img
                src={
                  isUnavailable
                    ? rawImageUrl || '/samples/sample_1_dense_urban.jpg'
                    : generatedImageUrl || rawImageUrl || '/samples/sample_1_dense_urban.jpg'
                }
                alt="Redesign"
                className="w-full h-full object-contain"
              />
              <span className="absolute top-3 left-3 bg-[#0d7a5f]/90 text-white font-mono text-xs px-2.5 py-1 rounded-md border border-emerald-400/40 flex items-center gap-1.5">
                <Sparkles className="w-3 h-3 text-emerald-300" />
                AI RESILIENT REDESIGN
              </span>
            </div>
          </div>
        )}

        {/* Realistic Stepped Progress Overlay during Generation / Refinement */}
        {(isGenerating || isRefining) && (
          <div className="absolute inset-0 bg-slate-950/85 backdrop-blur-md flex flex-col items-center justify-center gap-4 text-white font-sans z-40 p-6 text-center">
            <div className="relative">
              <div className="w-14 h-14 border-4 border-emerald-500/20 border-t-emerald-400 rounded-full animate-spin" />
              <Sparkles className="w-6 h-6 text-emerald-400 absolute inset-0 m-auto animate-pulse" />
            </div>

            <div className="flex flex-col items-center gap-1 max-w-md">
              <span className="text-lg font-bold text-white font-sans">
                {generationStage || 'Generating resilient redesign'}
              </span>
              <p className="text-xs text-slate-300">
                Synthesizing architectural interventions, natural canopy geometry, and solar-reflective materials with Gemini.
              </p>
            </div>

            {/* Stepper Dots */}
            <div className="flex items-center gap-2 mt-2">
              {[
                'Analyzing site',
                'Mapping urban surfaces',
                'Planning cooling interventions',
                'Generating resilient redesign',
                'Validating result',
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

      {/* Conversational Refinement Bar */}
      <div className="border border-slate-200/90 rounded-2xl p-4 bg-slate-50/70 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-emerald-600" />
            <span className="text-xs font-bold text-slate-800 uppercase tracking-wider font-mono">
              Refine Redesign with Gemini
            </span>
          </div>
          {refinementHistory.length > 0 && (
            <span className="text-[11px] font-mono text-emerald-700 bg-emerald-100/60 px-2 py-0.5 rounded-md font-semibold">
              {refinementHistory.length} refinement{refinementHistory.length > 1 ? 's' : ''} applied
            </span>
          )}
        </div>

        <form onSubmit={handleRefineSubmit} className="flex items-center gap-2">
          <input
            type="text"
            value={refineInput}
            onChange={(e) => setRefineInput(e.target.value)}
            disabled={isGenerating || isRefining}
            placeholder="Ask Gemini to adjust this redesign (e.g. 'Add two more trees', 'Make pavement lighter')..."
            className="flex-1 bg-white border border-slate-300 rounded-xl px-4 py-2.5 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500/30 focus:border-emerald-500 disabled:opacity-50 transition-all font-sans"
          />
          <button
            type="submit"
            disabled={!refineInput.trim() || isGenerating || isRefining}
            className="px-4 py-2.5 bg-[#0d7a5f] hover:bg-[#0b6b53] disabled:opacity-50 text-white rounded-xl text-xs font-bold flex items-center gap-1.5 transition-colors shadow-sm font-sans shrink-0"
          >
            <Send className="w-3.5 h-3.5" />
            <span>Refine</span>
          </button>
        </form>

        {/* Suggestion Chips */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none">
          <span className="text-[11px] text-slate-400 shrink-0 font-sans">Quick prompts:</span>
          {REFINEMENT_CHIPS.map((chip, idx) => (
            <button
              key={idx}
              type="button"
              disabled={isGenerating || isRefining}
              onClick={() => {
                setRefineInput(chip);
              }}
              className="text-[11px] bg-white hover:bg-emerald-50 text-slate-600 hover:text-emerald-800 border border-slate-200 hover:border-emerald-200 px-2.5 py-1 rounded-lg shrink-0 transition-colors font-sans"
            >
              {chip}
            </button>
          ))}
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
          Drag slider or switch to side-by-side to compare spatial interventions
        </span>
      </div>
    </div>
  );
};
