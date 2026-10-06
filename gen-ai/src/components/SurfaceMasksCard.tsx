import React from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import { Check } from 'lucide-react';

export const SurfaceMasksCard: React.FC = () => {
  const {
    segmentationMasks,
    visibleMaskIds,
    toggleMaskVisibility,
    isOverlayActive,
    toggleOverlayActive,
    segmentationSource,
    segmentationModelInfo,
  } = useResiliCityStore();

  return (
    <div className="rc-card p-5 flex flex-col justify-between gap-4 h-full">
      {/* Header with Switch */}
      <div className="flex items-center justify-between">
        <div>
          <span className="rc-card-header-label">SCENE UNDERSTANDING</span>
          <h3 className="text-lg font-bold text-slate-900 font-sans">Surface masks</h3>
        </div>

        {/* Master Toggle Switch */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-slate-400">
            {isOverlayActive ? 'Visible' : 'Hidden'}
          </span>
          <button
            onClick={toggleOverlayActive}
            className={`w-11 h-6 rounded-full transition-colors relative focus:outline-none cursor-pointer ${
              isOverlayActive ? 'bg-[#0d7a5f]' : 'bg-slate-200'
            }`}
            title="Toggle mask overlay on raw image"
          >
            <span
              className={`w-5 h-5 rounded-full bg-white shadow-md absolute top-0.5 transition-transform ${
                isOverlayActive ? 'left-5.5' : 'left-0.5'
              }`}
            />
          </button>
        </div>
      </div>

      <p className="text-xs text-slate-400 font-mono -mt-2">
        {segmentationSource === 'segformer'
          ? 'Live SegFormer B0 inference · exact pixel counts'
          : 'Calibrated reference scene surfaces'}
      </p>

      {/* Surface Bars with per-class toggles */}
      <div className="flex flex-col gap-3 my-1">
        {segmentationMasks.map((mask) => {
          const isVisible = visibleMaskIds.includes(mask.id);
          return (
            <div
              key={mask.id}
              onClick={() => toggleMaskVisibility(mask.id)}
              className={`flex flex-col gap-1.5 p-1.5 rounded-lg transition-all cursor-pointer hover:bg-slate-50 ${
                !isVisible ? 'opacity-40' : ''
              }`}
              title="Click to toggle mask visibility on image"
            >
              <div className="flex items-center justify-between text-xs font-sans font-medium text-slate-700">
                <div className="flex items-center gap-2">
                  <div
                    className="w-3.5 h-3.5 rounded flex items-center justify-center text-white shrink-0 shadow-xs"
                    style={{ backgroundColor: mask.color }}
                  >
                    {isVisible && <Check className="w-2.5 h-2.5 stroke-[3]" />}
                  </div>
                  <span className="font-semibold">{mask.label}</span>
                </div>
                <div className="flex items-center gap-2 font-mono">
                  {mask.pixelCount ? (
                    <span className="text-[10px] text-slate-400 hidden sm:inline">
                      {mask.pixelCount.toLocaleString()} px
                    </span>
                  ) : null}
                  <span className="font-bold text-slate-700">{mask.areaPercentage}%</span>
                </div>
              </div>

              {/* Progress Bar Container */}
              <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${mask.areaPercentage}%`,
                    backgroundColor: mask.color,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer Status */}
      <div className="flex items-center justify-between text-xs font-mono text-slate-400 border-t border-slate-100 pt-3">
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${isOverlayActive ? 'bg-emerald-500' : 'bg-slate-400'}`} />
          <span>{isOverlayActive ? 'Overlay active' : 'Overlay hidden'}</span>
        </div>
        <span className="text-[11px] text-slate-500">
          {segmentationSource === 'segformer'
            ? `${segmentationModelInfo?.name ? 'SegFormer-B0 (ADE20K)' : 'SegFormer baseline'}`
            : 'Pretrained SegFormer baseline'}
        </span>
      </div>
    </div>
  );
};

