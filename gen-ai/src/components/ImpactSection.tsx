import React from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import { exposureLabel } from '../services/heatCalculator';
import { AnimatedNumber } from './AnimatedNumber';
import { SourceChip } from './Figure';
import { Panel, SectionHeading, Stat } from './ui';
import { sentenceCase } from './format';

const ScoreScale: React.FC<{ base: number; projected: number }> = ({ base, projected }) => {
  const pos = (v: number) => `${Math.max(0, Math.min(100, v * 10))}%`;
  return (
    <div className="pt-6 pb-2">
      <div className="relative h-2 bg-gradient-to-r from-emerald-500 via-amber-400 to-rose-500">
        {/* Shaded span between projected and baseline */}
        <span
          className="absolute -top-px -bottom-px bg-white/60 border-y border-white transition-all duration-700"
          style={{ left: pos(Math.min(base, projected)), width: `${Math.abs(base - projected) * 10}%` }}
        />
        {[
          { v: base, label: 'Baseline', cls: 'bg-slate-900' },
          { v: projected, label: 'Proposed', cls: 'bg-[#0d7a5f]' },
        ].map((m) => (
          <span key={m.label} className="absolute -top-6 flex flex-col items-center transition-[left] duration-700 ease-out" style={{ left: pos(m.v), transform: 'translateX(-50%)' }}>
            <span className="text-[11px] text-slate-600 whitespace-nowrap">{m.label}</span>
            <span className={`w-0.5 h-6 mt-0.5 ${m.cls}`} />
          </span>
        ))}
      </div>
      <div className="mt-2 flex justify-between text-[11px] text-slate-400">
        <span>0</span>
        <span>5</span>
        <span>10</span>
      </div>
    </div>
  );
};

const CoolingBreakdown: React.FC = () => {
  const thermal = useResiliCityStore((s) => s.thermalImpact);
  const items = thermal?.perIntervention ?? [];
  if (!items.length) return <div className="h-32 rc-shimmer" />;
  const max = Math.max(...items.map((i) => i.rangeC[1]), 0.5);
  const pct = (v: number) => `${(v / max) * 100}%`;
  return (
    <ul className="rc-stagger flex flex-col gap-4">
      {items.map((it, idx) => (
        <li key={`${it.type}-${idx}`} className="flex flex-col gap-1.5">
          <div className="flex justify-between text-sm">
            <span className="text-slate-700">{sentenceCase(it.type.replace(/_/g, ' '))}</span>
            <span className="text-slate-900 tabular-nums">
              {it.coolingC.toFixed(1)} °C <span className="text-xs text-slate-400">({it.rangeC[0].toFixed(1)}–{it.rangeC[1].toFixed(1)})</span>
            </span>
          </div>
          <div className="relative h-3">
            <span className="absolute top-1/2 h-px bg-slate-200 w-full" />
            {/* Range whisker */}
            <span className="absolute top-1/2 -translate-y-1/2 h-3 border-x border-[#0d7a5f]/50 bg-[#e8f3ef]" style={{ left: pct(it.rangeC[0]), width: `calc(${pct(it.rangeC[1])} - ${pct(it.rangeC[0])})` }} />
            {/* Central estimate */}
            <span className="rc-bar absolute top-1/2 -translate-y-1/2 h-1.5 left-0 bg-[#0d7a5f]" style={{ width: pct(it.coolingC), animationDelay: `${idx * 80}ms` }} />
          </div>
        </li>
      ))}
    </ul>
  );
};

export const ImpactSection: React.FC = () => {
  const { heatMetrics, thermalImpact } = useResiliCityStore();

  return (
    <section className="flex flex-col gap-5">
      <SectionHeading
        id="impact"
        number="04"
        title="Impact"
        description="Indicative local cooling with ranges. Treat these as planning estimates, not measured temperatures."
      />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Panel eyebrow="Heat risk index" title="Baseline and proposed" actions={<SourceChip source="assumed" />}>
          <div className="grid grid-cols-2 gap-2">
            <Stat
              label="Baseline"
              value={
                <>
                  <AnimatedNumber value={heatMetrics.baseScore} /> <span className="text-sm text-slate-400">/ 10</span>
                </>
              }
              sub={exposureLabel(heatMetrics.baseScore)}
            />
            <Stat
              label="Proposed"
              tone="green"
              value={
                <>
                  <AnimatedNumber value={heatMetrics.projectedScore} /> <span className="text-sm text-slate-400">/ 10</span>
                </>
              }
              sub={`${heatMetrics.deltaScore > 0 ? '-' : ''}${heatMetrics.deltaScore.toFixed(1)} points`}
            />
          </div>
          <ScoreScale base={heatMetrics.baseScore} projected={heatMetrics.projectedScore} />
        </Panel>

        <Panel
          eyebrow="Cooling by intervention"
          title={
            thermalImpact ? (
              <>
                <AnimatedNumber value={thermalImpact.totalCoolingReductionC} /> °C total
                {thermalImpact.coolingRangeC && (
                  <span className="text-sm text-slate-500 font-normal">
                    {' '}
                    (range {thermalImpact.coolingRangeC[0].toFixed(1)}–{thermalImpact.coolingRangeC[1].toFixed(1)})
                  </span>
                )}
              </>
            ) : (
              'Awaiting analysis'
            )
          }
          actions={<SourceChip source="literature" />}
        >
          <CoolingBreakdown />
          {thermalImpact?.mlModelDeltaRangeC && (
            <p className="mt-5 text-xs text-slate-500">
              Legacy land-cover model, shown for reference only (low confidence, not used for these cooling figures):{' '}
              {thermalImpact.mlModelDeltaC.toFixed(2)} °C, band {thermalImpact.mlModelDeltaRangeC[0].toFixed(1)} to{' '}
              {thermalImpact.mlModelDeltaRangeC[1].toFixed(1)} °C (±RMSE).
            </p>
          )}
          {thermalImpact?.confidenceNote && <p className="mt-2 text-xs text-slate-500">{thermalImpact.confidenceNote}</p>}
          {thermalImpact?.warning && <p className="mt-2 text-xs text-amber-800">{thermalImpact.warning}</p>}
        </Panel>
      </div>
    </section>
  );
};
