import React, { useState, useRef, useCallback } from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import { RotateCw, Sliders, Eye, EyeOff } from 'lucide-react';

export const BeforeAfterPreview: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const {
    rawImageUrl,
    generatedImageUrl,
    sliderPosition,
    setSliderPosition,
    isGenerating,
    segmentationMasks,
    visibleMaskIds,
    isOverlayActive,
    toggleOverlayActive,
    imageDimensions,
    segmentationSource,
  } = useResiliCityStore();

  const handleMove = useCallback(
    (clientX: number) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = clientX - rect.left;
      const percentage = (x / rect.width) * 100;
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

  const imgW = imageDimensions?.width ?? 1024;
  const imgH = imageDimensions?.height ?? 1024;

  return (
    <div className="rc-card p-6 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <span className="rc-card-header-label">INTERVENTION PREVIEW</span>
          <h3 className="text-xl font-bold text-slate-900 font-sans">Before / after</h3>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={toggleOverlayActive}
            className={`flex items-center gap-1.5 text-xs font-mono font-semibold px-2.5 py-1 rounded-lg border transition-colors ${
              isOverlayActive
                ? 'text-[#0d7a5f] bg-emerald-50 border-emerald-300'
                : 'text-slate-500 bg-slate-50 border-slate-200 hover:bg-slate-100'
            }`}
            title="Toggle mask overlay on raw image"
          >
            {isOverlayActive ? <Eye className="w-3.5 h-3.5 text-emerald-600" /> : <EyeOff className="w-3.5 h-3.5 text-slate-400" />}
            <span>{isOverlayActive ? 'Mask overlay ON' : 'Mask overlay OFF'}</span>
          </button>

          <div className="flex items-center gap-1.5 text-xs font-mono text-emerald-600 font-semibold bg-emerald-50 px-2.5 py-1 rounded-lg border border-emerald-200/60">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>{segmentationSource === 'segformer' ? 'SegFormer live' : 'Sample scene'}</span>
          </div>

          <button
            onClick={() => setSliderPosition(50)}
            className="p-1.5 border border-slate-200 rounded-lg hover:bg-slate-50 text-slate-500 transition-colors"
            title="Reset slider position to 50%"
          >
            <RotateCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div
        ref={containerRef}
        className="relative w-full h-[420px] bg-slate-900 rounded-2xl overflow-hidden select-none touch-none cursor-ew-resize group shadow-sm border border-slate-200/80"
        onMouseDown={onMouseDown}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onMouseMove={onMouseMove}
        onTouchMove={onTouchMove}
      >
        <img
          src={rawImageUrl || '/samples/urban_street_before.png'}
          alt="Before scene"
          className="absolute inset-0 w-full h-full object-cover"
        />

        {/* Real SVG Segmentation Mask Overlay */}
        {isOverlayActive && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none z-10 transition-opacity duration-300"
            viewBox={`0 0 ${imgW} ${imgH}`}
            preserveAspectRatio="xMidYMid slice"
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
                    .map(([xPct, yPct]) => `${((xPct / 100) * imgW).toFixed(1)},${((yPct / 100) * imgH).toFixed(1)}`)
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

        <div
          className="absolute inset-0 overflow-hidden pointer-events-none z-20"
          style={{ clipPath: `inset(0 0 0 ${sliderPosition}%)` }}
        >
          <img
            src={generatedImageUrl || '/samples/urban_street_after.png'}
            alt="After scene"
            className="absolute inset-0 w-full h-full object-cover"
          />
        </div>

        <div className="absolute top-4 left-4 bg-slate-900/80 backdrop-blur border border-slate-700 text-white px-2.5 py-1 rounded-md text-[11px] font-mono font-bold z-30">
          BEFORE {isOverlayActive ? '· MASKS' : ''}
        </div>

        <div className="absolute top-4 right-4 bg-[#0d7a5f]/90 backdrop-blur border border-emerald-400/30 text-white px-2.5 py-1 rounded-md text-[11px] font-mono font-bold tracking-wider uppercase z-30">
          AFTER · ESTIMATE
        </div>

        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none z-30"
          style={{ left: `${sliderPosition}%` }}
        >
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 left-1/2 w-8 h-8 rounded-full bg-[#0d7a5f] border-2 border-white shadow-xl flex items-center justify-center text-white">
            <Sliders className="w-4 h-4 rotate-90" />
          </div>
        </div>

        {isGenerating && (
          <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex flex-col items-center justify-center gap-2 text-white font-mono text-xs z-40">
            <div className="w-8 h-8 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
            <span>Updating cached edit preview...</span>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-4 text-xs font-mono text-slate-500 pt-1">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-rose-500" />
            <span>Original scene</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>Selected edit preview · cached per intervention</span>
          </div>
        </div>

        <span className="text-[11px] text-slate-400">
          Drag slider to inspect segmented before vs. resilient after
        </span>
      </div>
    </div>
  );
};

