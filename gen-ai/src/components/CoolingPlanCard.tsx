import React from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import {
  Check,
  CloudSun,
  Trees,
  Layers,
  Sparkles,
  Zap,
  TrendingDown,
  ShieldCheck,
  Percent,
} from 'lucide-react';

export const CoolingPlanCard: React.FC = () => {
  const {
    interventions,
    isGenerating,
    planSummary,
    generateResilientDesign,
    spatialDesignPlan,
  } = useResiliCityStore();

  const getInterventionIcon = (type: string) => {
    switch (type) {
      case 'cool_roof':
      case 'green_roof':
        return <CloudSun className="w-4 h-4 text-amber-600" />;
      case 'tree_canopy':
        return <Trees className="w-4 h-4 text-emerald-600" />;
      case 'cool_pavement':
      case 'permeable_pave':
        return <Layers className="w-4 h-4 text-sky-600" />;
      default:
        return <Zap className="w-4 h-4 text-teal-600" />;
    }
  };

  const getIconBg = (type: string) => {
    switch (type) {
      case 'cool_roof':
      case 'green_roof':
        return 'bg-amber-50 border-amber-200';
      case 'tree_canopy':
        return 'bg-emerald-50 border-emerald-200';
      case 'cool_pavement':
      case 'permeable_pave':
        return 'bg-sky-50 border-sky-200';
      default:
        return 'bg-teal-50 border-teal-200';
    }
  };

  const summaryText =
    spatialDesignPlan?.overall_design_intent ||
    spatialDesignPlan?.site_summary ||
    planSummary ||
    'AI urban design plan optimizing heat resilience through physical spatial reasoning.';

  return (
    <div className="rc-card p-6 flex flex-col justify-between gap-5 h-full">
      {/* Header */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div>
            <span className="rc-card-header-label">SPATIAL RESILIENCE STRATEGY</span>
            <h3 className="text-xl font-bold text-slate-900 font-sans">Autonomous Interventions</h3>
          </div>

          <div className="flex items-center gap-1.5 text-xs font-mono font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-lg">
            <Check className="w-3.5 h-3.5 text-emerald-600" />
            <span>AI Spatial Reasoning</span>
          </div>
        </div>

        {/* Strategy Summary */}
        <div className="bg-slate-50 border-l-4 border-teal-600 p-4 rounded-r-xl text-xs text-slate-700 font-sans leading-relaxed">
          <strong>Site Assessment:</strong> {summaryText}
        </div>
      </div>

      {/* Intervention Cards (Informational, No Checkboxes/Toggles) */}
      <div className="flex flex-col gap-3 my-1">
        {interventions.map((item) => {
          const anyItem = item as any;
          const reasonText = anyItem.reason || item.description;
          const placementText = anyItem.placement;
          const feasibilityPct = anyItem.feasibility ? Math.round(anyItem.feasibility * 100) : 92;
          const coveragePct = Math.round((item.coverage || 0.65) * 100);

          return (
            <div
              key={item.id}
              className="p-4 border rounded-2xl bg-white border-slate-200 shadow-xs flex flex-col gap-2.5"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div
                    className={`w-10 h-10 rounded-xl border flex items-center justify-center shrink-0 ${getIconBg(
                      item.type
                    )}`}
                  >
                    {getInterventionIcon(item.type)}
                  </div>

                  <div className="flex flex-col min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold text-slate-900 font-sans truncate">
                        {item.title}
                      </span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 uppercase border border-slate-200">
                        {item.targetRegion}
                      </span>
                    </div>

                    <span className="text-xs text-slate-500 font-sans line-clamp-1">
                      {item.description}
                    </span>
                  </div>
                </div>

                <span className="text-[10px] font-mono font-bold text-emerald-700 bg-emerald-50 px-2 py-1 rounded-md border border-emerald-200 shrink-0">
                  Priority #{item.priority}
                </span>
              </div>

              {/* "Why" Reason & Placement */}
              <div className="bg-slate-50/80 border border-slate-100 p-2.5 rounded-xl flex flex-col gap-1.5 text-[11px] font-sans">
                <div className="text-slate-700">
                  <strong className="text-emerald-800">Reason:</strong> {reasonText}
                </div>

                {placementText && (
                  <div className="text-slate-600">
                    <strong className="text-slate-800">Placement:</strong> {placementText}
                  </div>
                )}

                <div className="flex items-center flex-wrap gap-x-4 gap-y-1 text-slate-500 font-mono text-[10px] pt-1 border-t border-slate-200/60">
                  <span className="flex items-center gap-1">
                    <Percent className="w-3 h-3 text-slate-400" />
                    Coverage: {coveragePct}%
                  </span>
                  <span className="flex items-center gap-1">
                    <ShieldCheck className="w-3 h-3 text-emerald-600" />
                    Feasibility: {feasibilityPct}%
                  </span>
                  <span className="flex items-center gap-1 text-teal-700 font-bold">
                    <TrendingDown className="w-3 h-3 text-teal-600" />
                    -{item.coolingImpact.toFixed(1)}°C est. cooling
                  </span>
                  <span className="text-slate-400">Cost: {item.estCostText}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Action & Status Bar */}
      <div className="flex flex-col gap-2 pt-2 border-t border-slate-100">
        <div className="flex items-center gap-2">
          {/* Re-run Autonomous Redesign Button */}
          <button
            type="button"
            disabled={isGenerating}
            onClick={() => void generateResilientDesign('fast')}
            className="flex-1 py-3 px-4 bg-[#0d7a5f] hover:bg-[#0b6b53] disabled:opacity-50 text-white font-bold rounded-xl text-sm flex items-center justify-center gap-2 transition-all shadow-md font-sans cursor-pointer"
          >
            <Sparkles className="w-4 h-4 text-emerald-300" />
            <span>{isGenerating ? 'Generating Resilient Redesign...' : 'RE-RUN AUTONOMOUS REDESIGN'}</span>
          </button>
        </div>

        <div className="flex items-center justify-between text-[11px] text-slate-500 font-sans px-1">
          <span>Interventions autonomously selected from SegFormer surface perception.</span>
          <span className="font-mono text-emerald-700 font-semibold">Local SDXL · RTX 3050 6GB</span>
        </div>
      </div>
    </div>
  );
};
