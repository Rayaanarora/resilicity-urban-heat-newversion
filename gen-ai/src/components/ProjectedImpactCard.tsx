import React from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import { Play, RotateCw } from 'lucide-react';
import { AnimatedNumber } from './AnimatedNumber';

export const ProjectedImpactCard: React.FC = () => {
  const { heatMetrics, interventions, activeInterventionIds, isGenerating, generateResilientDesign } = useResiliCityStore();

  return (
    <div className="rc-card p-6 flex flex-col justify-between gap-5 h-full">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <span className="rc-card-header-label">PROJECTED IMPACT</span>
          <h3 className="text-xl font-semibold text-slate-900 font-sans">Score delta</h3>
        </div>
        <span className="text-xs font-mono font-medium text-teal-700">Indicative estimate</span>
      </div>

      {/* Main Metric Comparison */}
      <div className="flex items-center justify-between py-2">
        {/* Projected Score */}
        <div className="flex flex-col">
          <span className="text-[11px] font-mono uppercase font-medium text-slate-400">PROJECTED</span>
          <span className="text-4xl font-semibold font-sans text-teal-700 tracking-normal">
            <AnimatedNumber value={heatMetrics.projectedScore} />
          </span>
        </div>

        {/* Arrow Icon */}
        <div className="h-px w-12 bg-slate-300" aria-hidden="true" />

        {/* Baseline Score */}
        <div className="flex flex-col items-end">
          <span className="text-[11px] font-mono uppercase font-medium text-slate-400">BASELINE</span>
          <span className="text-4xl font-semibold font-sans text-amber-700 tracking-normal">
            <AnimatedNumber value={heatMetrics.baseScore} />
          </span>
        </div>
      </div>

      {/* Indicator Line */}
      <div className="h-1 w-full bg-[#0d7a5f] rc-bar" />

      {/* Score Summary Metrics */}
      <div className="flex flex-col gap-2 font-sans text-xs">
        <div className="flex justify-between items-center text-slate-600">
          <span>Potential reduction</span>
          <span className="font-mono font-medium text-teal-700">
            -{heatMetrics.deltaScore.toFixed(1)} points
          </span>
        </div>

        <div className="flex justify-between items-center text-slate-600">
          <span>Interventions applied</span>
          <span className="font-mono font-medium text-slate-800">
            {activeInterventionIds.length} / {interventions.length}
          </span>
        </div>
      </div>

      {/* Main Action Button */}
      <button
        onClick={() => void generateResilientDesign('fast')}
        disabled={isGenerating}
        className="w-full py-3.5 rc-btn-primary rounded-xl flex items-center justify-center gap-2 text-sm font-medium shadow-md cursor-pointer disabled:opacity-50"
      >
        <Play className="w-4 h-4 fill-current" />
        <span>{isGenerating ? 'Analyzing...' : 'Run Resilience Analysis'}</span>
      </button>

      <div className="flex items-center justify-center gap-1.5 text-xs font-mono text-slate-400">
        <RotateCw className="w-3.5 h-3.5" />
        <span>Indicative · see implementation plan for ranges</span>
      </div>
    </div>
  );
};
