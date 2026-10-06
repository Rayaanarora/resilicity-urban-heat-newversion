import React from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import type { DesignProfile } from '../types/resilicity';
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
  SlidersHorizontal,
} from 'lucide-react';

const DESIGN_PROFILES: Array<{
  id: DesignProfile;
  label: string;
  tagline: string;
}> = [
  { id: 'balanced', label: 'Balanced', tagline: 'Mix greenery, shade, and reflective materials.' },
  { id: 'pedestrian_first', label: 'Pedestrian First', tagline: 'Prioritize shaded walkways and pedestrian thermal comfort.' },
  { id: 'maximum_cooling', label: 'Maximum Cooling', tagline: 'Prioritize interventions with greatest surface temperature reduction.' },
  { id: 'green_infrastructure', label: 'Green Infra', tagline: 'Emphasize living tree canopies and biophilic planting.' },
  { id: 'low_cost', label: 'Low Cost', tagline: 'Prioritize lower-cost reflective surface sealants and light shading.' },
];

export const CoolingPlanCard: React.FC = () => {
  const {
    interventions,
    activeInterventionIds,
    toggleIntervention,
    isGenerating,
    isRefining,
    planSource,
    planSummary,
    designProfile,
    setDesignProfile,
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
      {/* Header and Profile Selector */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div>
            <span className="rc-card-header-label">SPATIAL RESILIENCE STRATEGY</span>
            <h3 className="text-xl font-bold text-slate-900 font-sans">Cooling Interventions</h3>
          </div>

          <div className="flex items-center gap-1.5 text-xs font-mono font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-lg">
            <Check className="w-3.5 h-3.5 text-emerald-600" />
            <span>{planSource === 'vlm' ? 'Gemini Spatial Plan' : 'Perception-Constrained'}</span>
          </div>
        </div>

        {/* Design Profile Pills */}
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-1.5 text-xs text-slate-500 font-mono">
            <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
            <span>Design Profile:</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-1.5">
            {DESIGN_PROFILES.map((dp) => {
              const isSelected = designProfile === dp.id;
              return (
                <button
                  key={dp.id}
                  type="button"
                  onClick={() => {
                    setDesignProfile(dp.id);
                    void generateResilientDesign('fast');
                  }}
                  title={dp.tagline}
                  className={`px-2.5 py-1.5 rounded-xl text-xs font-semibold border transition-all text-center truncate ${
                    isSelected
                      ? 'bg-[#0d7a5f] border-[#0d7a5f] text-white shadow-sm font-bold'
                      : 'bg-white border-slate-200 text-slate-600 hover:border-slate-300 hover:bg-slate-50'
                  }`}
                >
                  {dp.label}
                </button>
              );
            })}
          </div>

          <p className="text-[11px] text-slate-500 italic mt-0.5">
            Profile objective:{' '}
            {DESIGN_PROFILES.find((p) => p.id === designProfile)?.tagline}
          </p>
        </div>
      </div>

      {/* Strategy Summary */}
      <div className="bg-slate-50 border-l-4 border-teal-600 p-4 rounded-r-xl text-xs text-slate-700 font-sans leading-relaxed">
        <strong>Site Intent:</strong> {summaryText}
      </div>

      {/* Intervention Cards */}
      <div className="flex flex-col gap-3 my-1">
        {interventions.map((item) => {
          const isActive = activeInterventionIds.includes(item.id);
          const anyItem = item as any;
          const reasonText = anyItem.reason || item.description;
          const feasibilityPct = anyItem.feasibility ? Math.round(anyItem.feasibility * 100) : 92;
          const coveragePct = Math.round((item.coverage || 0.65) * 100);

          return (
            <div
              key={item.id}
              onClick={() => !isGenerating && !isRefining && toggleIntervention(item.id)}
              className={`p-4 border rounded-2xl cursor-pointer transition-all flex flex-col gap-2.5 ${
                isActive
                  ? 'bg-emerald-50/30 border-emerald-500/70 shadow-sm'
                  : 'bg-white border-slate-200 hover:border-slate-300 opacity-75'
              }`}
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

                {/* Checkbox */}
                <div
                  className={`w-6 h-6 rounded-lg border flex items-center justify-center transition-colors shrink-0 ${
                    isActive
                      ? 'bg-[#0d7a5f] border-[#0d7a5f] text-white'
                      : 'bg-white border-slate-300'
                  }`}
                >
                  {isActive && <Check className="w-4 h-4 stroke-[3]" />}
                </div>
              </div>

              {/* "Why" Reason & Specifications */}
              <div className="bg-white/80 border border-slate-100 p-2.5 rounded-xl flex flex-col gap-1 text-[11px] font-sans">
                <div className="text-slate-700">
                  <strong className="text-emerald-800">Why:</strong> {reasonText}
                </div>

                <div className="flex items-center flex-wrap gap-x-4 gap-y-1 text-slate-500 font-mono text-[10px] pt-1 border-t border-slate-100/80">
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
          {/* Primary Regenerate Button */}
          <button
            type="button"
            disabled={isGenerating}
            onClick={() => void generateResilientDesign('fast')}
            className="flex-1 py-3 px-4 bg-[#0d7a5f] hover:bg-[#0b6b53] disabled:opacity-50 text-white font-bold rounded-xl text-sm flex items-center justify-center gap-2 transition-all shadow-md font-sans"
          >
            <Sparkles className="w-4 h-4 text-emerald-300" />
            <span>{isGenerating ? 'Generating Resilient Redesign...' : 'RE-RUN AUTONOMOUS REDESIGN'}</span>
          </button>
        </div>

        <div className="flex items-center justify-between text-[11px] text-slate-500 font-sans px-1">
          <span>ResiliCity autonomously selects interventions based on site surfaces.</span>
          <span className="font-mono text-emerald-700 font-semibold">Local SDXL · RTX 3050</span>
        </div>
      </div>
    </div>
  );
};
