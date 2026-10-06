import React from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import { Play, RotateCw } from 'lucide-react';

export const ProjectedImpactCard: React.FC = () => {
  const { heatMetrics, interventions, activeInterventionIds, isGenerating } = useResiliCityStore();

  return (
    <div className="rc-card p-6 flex flex-col justify-between gap-5 h-full">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <span className="rc-card-header-label">PROJECTED IMPACT</span>
          <h3 className="text-xl font-bold text-slate-900 font-sans">Score delta</h3>
        </div>
        <span className="text-xs font-mono font-semibold text-slate-400">RUN / 0248</span>
      </div>

      {/* Main Metric Comparison */}
      <div className="flex items-center justify-between py-2">
        {/* Current Score */}
        <div className="flex flex-col">
          <span className="text-[11px] font-mono uppercase font-semibold text-slate-400">CURRENT</span>
          <span className="text-4xl font-black font-sans text-teal-700 tracking-tight">
            {heatMetrics.projectedScore.toFixed(1)}
          </span>
        </div>

        {/* Arrow Icon */}
        <div className="text-slate-300 text-xl font-bold font-mono">→</div>

        {/* Baseline Score */}
        <div className="flex flex-col items-end">
          <span className="text-[11px] font-mono uppercase font-semibold text-slate-400">BASELINE</span>
          <span className="text-4xl font-black font-sans text-amber-700 tracking-tight">
            {heatMetrics.baseScore.toFixed(1)}
          </span>
        </div>
      </div>

      {/* Indicator Line */}
      <div className="h-1.5 w-full bg-[#0d7a5f] rounded-full" />

      {/* Score Summary Metrics */}
      <div className="flex flex-col gap-2 font-sans text-xs">
        <div className="flex justify-between items-center text-slate-600">
          <span>Potential reduction</span>
          <span className="font-mono font-bold text-teal-700">
            -{heatMetrics.deltaScore.toFixed(1)} points
          </span>
        </div>

        <div className="flex justify-between items-center text-slate-600">
          <span>Active measures</span>
          <span className="font-mono font-bold text-slate-800">
            {activeInterventionIds.length} / {interventions.length}
          </span>
        </div>
      </div>

      {/* Main Action Button */}
      <button
        disabled={isGenerating}
        className="w-full py-3.5 rc-btn-primary rounded-xl flex items-center justify-center gap-2 text-sm font-bold shadow-md cursor-pointer disabled:opacity-50"
      >
        <Play className="w-4 h-4 fill-current" />
        <span>Run analysis</span>
      </button>

      {/* Retry Colab Note */}
      <div className="flex items-center justify-center gap-1.5 text-xs font-mono text-slate-400">
        <RotateCw className="w-3.5 h-3.5" />
        <span>Retry if Colab times out</span>
      </div>
    </div>
  );
};
