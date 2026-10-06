import React from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import { AlertTriangle, LineChart } from 'lucide-react';
import { exposureLabel } from '../services/heatCalculator';

export const HeatRiskScoreCard: React.FC = () => {
  const { heatMetrics } = useResiliCityStore();

  return (
    <div className="rc-card p-6 flex flex-col gap-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <span className="rc-card-header-label">EXPLAINABLE ESTIMATE</span>
          <h3 className="text-xl font-bold text-slate-900 font-sans">Heat risk score</h3>
        </div>

        <div className="flex items-center gap-2">
          <div className="border border-emerald-300 text-emerald-800 bg-emerald-50 px-2.5 py-0.5 rounded-md text-[10px] font-mono font-bold uppercase tracking-wider flex items-center gap-1">
            <LineChart className="w-3 h-3 text-emerald-600" />
            <span>Rule-based albedo model</span>
          </div>

          <div className="border border-orange-300 text-orange-700 bg-orange-50/50 px-2.5 py-0.5 rounded-md text-[10px] font-mono font-bold uppercase tracking-wider">
            ESTIMATE
          </div>
        </div>
      </div>

      {/* Big Score Number & Exposure Badge */}
      <div className="flex items-baseline gap-3">
        <div className="text-5xl font-black font-sans text-teal-800 tracking-tight">
          {heatMetrics.baseScore.toFixed(1)}
          <span className="text-xl font-normal text-slate-400 font-mono">/10</span>
        </div>

        <div className="flex flex-col">
          <span className="text-xs font-bold text-orange-700 font-sans">{exposureLabel(heatMetrics.baseScore)}</span>
          <span className="text-xs text-slate-400 font-sans">
            baseline, before interventions
            {heatMetrics.deltaScore > 0 && ` · ${heatMetrics.projectedScore.toFixed(1)}/10 with selected plan`}
          </span>
        </div>
      </div>

      {/* Gradient Bar Gauge */}
      <div className="relative h-2.5 w-full rounded-full overflow-hidden bg-slate-100">
        <div
          className="h-full rounded-full transition-all duration-500 bg-gradient-to-r from-emerald-500 via-amber-500 to-rose-500"
          style={{ width: `${(heatMetrics.baseScore / 10) * 100}%` }}
        />
      </div>

      {/* Breakdown Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2 text-xs font-sans">
        <div className="flex justify-between items-center bg-slate-50 p-2.5 rounded-lg border border-slate-100">
          <span className="text-slate-500">Baseline load</span>
          <span className="font-mono font-bold text-orange-600">+{heatMetrics.baseScore.toFixed(1)}</span>
        </div>

        <div className="flex justify-between items-center bg-slate-50 p-2.5 rounded-lg border border-slate-100">
          <span className="text-slate-500">Surface treatments</span>
          <span className="font-mono font-bold text-teal-600">-{heatMetrics.surfaceCooling.toFixed(1)}</span>
        </div>

        <div className="flex justify-between items-center bg-slate-50 p-2.5 rounded-lg border border-slate-100">
          <span className="text-slate-500">Shade &amp; canopy</span>
          <span className="font-mono font-bold text-teal-600">-{heatMetrics.shadeCooling.toFixed(1)}</span>
        </div>
      </div>

      {/* Disclaimer Footer */}
      <div className="flex items-center gap-2 text-xs font-sans text-orange-800/90 bg-orange-50/60 border border-orange-200/60 px-3.5 py-2.5 rounded-xl">
        <AlertTriangle className="w-4 h-4 text-orange-600 shrink-0" />
        <span>Calibrated against satellite Land Surface Temperature (LST). Not a CFD physics air temperature measurement.</span>
      </div>
    </div>
  );
};
