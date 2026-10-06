import React from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import { Check, CloudSun, Trees, MapPin, Zap } from 'lucide-react';

export const CoolingPlanCard: React.FC = () => {
  const { interventions, activeInterventionIds, toggleIntervention, isGenerating, planSource, planSummary } = useResiliCityStore();
  const label = { vlm: 'VLM RECOMMENDATION', rules: 'RULE-BASED PLAN', sample: 'SAMPLE PLAN' }[planSource];
  const badge = { vlm: 'Validated vs catalog', rules: 'AI planner offline', sample: 'Reference scene' }[planSource];
  const summary =
    planSource === 'vlm' && planSummary
      ? planSummary
      : planSource === 'rules'
        ? 'AI planner unavailable, so these measures are chosen by simple rules from the detected surfaces.'
        : 'The scene is dominated by hard, low-albedo surfaces. Start with roof reflectance and shade at the pedestrian edge.';

  const getInterventionIcon = (type: string) => {
    switch (type) {
      case 'cool_roof':
        return <CloudSun className="w-4 h-4 text-amber-600" />;
      case 'tree_canopy':
        return <Trees className="w-4 h-4 text-emerald-600" />;
      default:
        return <MapPin className="w-4 h-4 text-orange-600" />;
    }
  };

  const getIconBg = (type: string) => {
    switch (type) {
      case 'cool_roof':
        return 'bg-amber-50 border-amber-200';
      case 'tree_canopy':
        return 'bg-emerald-50 border-emerald-200';
      default:
        return 'bg-orange-50 border-orange-200';
    }
  };

  return (
    <div className="rc-card p-6 flex flex-col justify-between gap-5 h-full">
      <div className="flex items-center justify-between">
        <div>
          <span className="rc-card-header-label">{label}</span>
          <h3 className="text-xl font-bold text-slate-900 font-sans">Cooling plan</h3>
        </div>

        <div className="flex items-center gap-1.5 text-xs font-mono font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-lg">
          <Check className="w-3.5 h-3.5 text-emerald-600" />
          <span>{badge}</span>
        </div>
      </div>

      <div className="bg-slate-50 border-l-4 border-teal-600 p-4 rounded-r-xl text-xs text-slate-600 font-sans leading-relaxed italic">
        “{summary}”
      </div>

      <div className="flex flex-col gap-3 my-1">
        {interventions.map((item) => {
          const isActive = activeInterventionIds.includes(item.id);
          return (
            <div
              key={item.id}
              onClick={() => !isGenerating && toggleIntervention(item.id)}
              className={`p-3.5 border rounded-xl cursor-pointer transition-all flex items-center justify-between gap-3 ${
                isActive
                  ? 'bg-emerald-50/20 border-emerald-500/60 shadow-sm'
                  : 'bg-white border-slate-200 hover:border-slate-300'
              }`}
            >
              <div className="flex items-center gap-3 min-w-0">
                <div
                  className={`w-9 h-9 rounded-xl border flex items-center justify-center shrink-0 ${getIconBg(
                    item.type
                  )}`}
                >
                  {getInterventionIcon(item.type)}
                </div>

                <div className="flex flex-col min-w-0">
                  <span className="text-sm font-bold text-slate-900 font-sans truncate">
                    {item.title}
                  </span>
                  <span className="text-xs text-slate-400 font-sans truncate">
                    {item.description}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-3 shrink-0">
                <div className="flex flex-col items-end">
                  <span className="text-[10px] font-mono text-slate-400">{item.estCostText}</span>
                  <span className="text-sm font-mono font-extrabold text-teal-700">
                    -{item.coolingImpact.toFixed(1)}
                  </span>
                </div>

                <div
                  className={`w-6 h-6 rounded-lg border flex items-center justify-center transition-colors ${
                    isActive
                      ? 'bg-[#0d7a5f] border-[#0d7a5f] text-white'
                      : 'bg-white border-slate-300'
                  }`}
                >
                  {isActive && <Check className="w-4 h-4 stroke-[3]" />}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="flex items-center gap-2 text-xs font-sans text-slate-500 pt-2 border-t border-slate-100">
        <Zap className="w-4 h-4 text-teal-600 shrink-0" />
        <span>Toggle an intervention to update the composition. Cached edits avoid a full inpainting pass.</span>
      </div>
    </div>
  );
};
