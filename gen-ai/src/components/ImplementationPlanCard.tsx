import React, { useState } from 'react';
import { ChevronDown, RefreshCw } from 'lucide-react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import type { ClimateZone } from '../types/resilicity';
import { AnimatedNumber } from './AnimatedNumber';
import { Button, Panel, Stat, Tag } from './ui';
import { zoneLabel } from './format';

const LAKH = 100_000;
const lakh = (n: number) => n / LAKH;
const inr = (n: number) => (n >= LAKH ? `₹${lakh(n).toFixed(1)} L` : `₹${Math.round(n).toLocaleString('en-IN')}`);
const inrRange = (lo: number, hi: number) => `${inr(lo)} – ${inr(hi)}`;
const qty = (q: number) => (Number.isInteger(q) ? q.toString() : q.toFixed(1));

const parseOptional = (v: string): number | null => {
  const n = Number(v);
  return v.trim() === '' || !Number.isFinite(n) || n <= 0 ? null : n;
};

const CLIMATES: { id: ClimateZone; label: string }[] = [
  { id: 'tropical', label: 'Tropical humid' },
  { id: 'arid', label: 'Arid' },
  { id: 'temperate', label: 'Temperate' },
  { id: 'mediterranean', label: 'Mediterranean' },
];

const Field: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <label className="flex flex-col gap-1.5">
    <span className="rc-card-header-label">{label}</span>
    {children}
  </label>
);

const inputCls =
  'h-9 w-40 px-3 border border-slate-300 bg-white text-sm text-slate-900 focus:outline-none focus:border-[#0d7a5f] transition-colors';

const Collapsible: React.FC<{ title: string; count: number; children: React.ReactNode; defaultOpen?: boolean }> = ({
  title,
  count,
  children,
  defaultOpen = false,
}) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-slate-200">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="rc-row w-full flex items-center justify-between px-4 py-3 text-sm text-slate-900 cursor-pointer"
      >
        <span>
          {title} <span className="text-slate-400">({count})</span>
        </span>
        <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform duration-300 ${open ? 'rotate-180' : ''}`} />
      </button>
      <div className="rc-collapse" data-open={open}>
        <div>
          <div className="border-t border-slate-200 px-4 py-3">{children}</div>
        </div>
      </div>
    </div>
  );
};

export const ImplementationPlanCard: React.FC = () => {
  const {
    spatialDesignPlan: plan,
    thermalImpact,
    roadWidthM,
    setRoadWidthM,
    budgetLakh,
    setBudgetLakh,
    climateZone,
    setClimateZone,
    isGenerating,
    generateResilientDesign,
  } = useResiliCityStore();
  const impl = plan?.implementation_plan;
  const rejected = (plan?.rejected_candidates ?? []).filter((r) => r.rule);
  const totalMid = impl ? (impl.total_cost_inr_low + impl.total_cost_inr_high) / 2 : 0;

  return (
    <Panel
      eyebrow="Implementation plan"
      title={impl ? `Bill of quantities per ${impl.street_length_m.toFixed(0)} m of street` : 'Bill of quantities'}
      actions={<Tag tone="amber">Indicative rates, not for tender</Tag>}
    >
      {/* Site inputs */}
      <div className="flex flex-wrap items-end gap-4 pb-5 border-b border-slate-200">
        <Field label="Carriageway width (m)">
          <input
            type="number"
            min={3}
            step={0.5}
            placeholder="7.0 assumed"
            defaultValue={roadWidthM ?? ''}
            onChange={(e) => setRoadWidthM(parseOptional(e.target.value))}
            className={inputCls}
          />
        </Field>
        <Field label="Budget cap (₹ lakh)">
          <input
            type="number"
            min={0}
            step={0.5}
            placeholder="No cap"
            defaultValue={budgetLakh ?? ''}
            onChange={(e) => setBudgetLakh(parseOptional(e.target.value))}
            className={inputCls}
          />
        </Field>
        <Field label="Climate">
          <select value={climateZone} onChange={(e) => setClimateZone(e.target.value as ClimateZone)} className={`${inputCls} cursor-pointer`}>
            {CLIMATES.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </Field>
        <Button variant="primary" icon={<RefreshCw className="w-4 h-4" />} disabled={isGenerating} onClick={() => void generateResilientDesign('fast')}>
          {isGenerating ? 'Re-planning…' : 'Re-plan'}
        </Button>
      </div>

      {!impl ? (
        <div className="pt-5 flex flex-col gap-3">
          <div className="h-16 rc-shimmer" />
          <div className="h-32 rc-shimmer" />
          <p className="text-sm text-slate-500">Run an analysis to produce quantities, costs and phasing.</p>
        </div>
      ) : (
        <div className="pt-5 flex flex-col gap-6">
          {/* Summary */}
          <div className="rc-stagger grid grid-cols-2 lg:grid-cols-4 gap-2">
            <Stat
              label="Indicative cost"
              value={
                <>
                  ₹<AnimatedNumber value={lakh(impl.total_cost_inr_low)} />–<AnimatedNumber value={lakh(impl.total_cost_inr_high)} /> L
                </>
              }
              sub="Range from unit rates"
            />
            <Stat
              label="Local cooling"
              tone="green"
              value={
                <>
                  <AnimatedNumber value={impl.total_cooling_c} /> °C
                </>
              }
              sub={
                thermalImpact?.coolingRangeC
                  ? `Range ${thermalImpact.coolingRangeC[0].toFixed(1)}–${thermalImpact.coolingRangeC[1].toFixed(1)} °C`
                  : 'Indicative'
              }
            />
            <Stat
              label="Cooling per lakh"
              value={
                <>
                  <AnimatedNumber value={impl.cooling_per_lakh_c} decimals={2} /> °C
                </>
              }
              sub="At mid cost"
            />
            <Stat
              label="Budget"
              value={impl.budget_lakh ? `₹${impl.budget_lakh} L cap` : 'No cap'}
              sub={impl.budget_lakh ? (impl.within_budget ? 'Within cap' : 'Over cap') : 'Set a cap to rank by value'}
            />
          </div>

          {/* Phasing timeline */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            {impl.phases.map((p) => {
              const share = totalMid ? ((p.cost_inr_low + p.cost_inr_high) / 2 / totalMid) * 100 : 0;
              return (
                <div key={p.phase} className="border border-slate-200 p-4 flex flex-col gap-2 transition-colors hover:border-[#b7d9cc]">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-900">{p.label}</span>
                    <span className="text-xs text-slate-500">{Math.round(share)}% of cost</span>
                  </div>
                  <span className="h-1 bg-slate-100 block">
                    <span className="rc-bar block h-full bg-[#0d7a5f]" style={{ width: `${share}%` }} />
                  </span>
                  <ul className="text-xs text-slate-600 flex flex-col gap-0.5">
                    {p.items.map((t, i) => (
                      <li key={`${t}-${i}`}>{t}</li>
                    ))}
                  </ul>
                  <span className="text-xs text-slate-500 tabular-nums">{inrRange(p.cost_inr_low, p.cost_inr_high)}</span>
                </div>
              );
            })}
          </div>

          {/* Bill of quantities */}
          <div className="overflow-x-auto -mx-5 px-5">
            <table className="w-full min-w-[720px] text-sm border-collapse">
              <thead>
                <tr className="text-left border-b border-slate-300">
                  {['Item', 'Side', 'Quantity', 'Cost range', 'Share', '°C / lakh', 'Phase'].map((h) => (
                    <th key={h} className="py-2 pr-4 rc-card-header-label">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="rc-stagger">
                {impl.line_items.map((li) => {
                  const share = totalMid ? ((li.cost_inr_low + li.cost_inr_high) / 2 / totalMid) * 100 : 0;
                  return (
                    <tr key={`${li.type}-${li.target_zone}`} className="rc-row border-b border-slate-100 align-top">
                      <td className="py-3 pr-4">
                        <div className="text-slate-900">{li.title}</div>
                        <div className="text-xs text-slate-500">{li.basis}</div>
                        {li.conditions.map((c) => (
                          <div key={c} className="text-xs text-amber-800 mt-0.5">
                            Condition: {c}
                          </div>
                        ))}
                      </td>
                      <td className="py-3 pr-4 text-slate-600 whitespace-nowrap">{zoneLabel(li.target_zone)}</td>
                      <td className="py-3 pr-4 text-slate-900 whitespace-nowrap tabular-nums">
                        {qty(li.quantity)} {li.unit}
                        {li.quantity_flagged && (
                          <span className="block text-xs text-amber-700" title="Uses a default, not a measurement">
                            default width
                          </span>
                        )}
                        {li.extra?.kwp ? <span className="block text-xs text-slate-500">{li.extra.kwp} kWp</span> : null}
                      </td>
                      <td className="py-3 pr-4 text-slate-900 whitespace-nowrap tabular-nums">{inrRange(li.cost_inr_low, li.cost_inr_high)}</td>
                      <td className="py-3 pr-4 w-28">
                        <span className="h-1.5 bg-slate-100 block mt-1.5">
                          <span className="rc-bar block h-full bg-[#0d7a5f]" style={{ width: `${share}%` }} />
                        </span>
                      </td>
                      <td className="py-3 pr-4 text-[#0d7a5f] tabular-nums">{li.cooling_per_lakh_c.toFixed(2)}</td>
                      <td className="py-3 text-slate-600">
                        {li.phase_label}
                        <span className="block text-xs text-slate-400">{li.lifespan_years} year life</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="flex flex-col gap-2">
            {impl.prerequisites.length > 0 && (
              <Collapsible title="Prerequisites and permits" count={impl.prerequisites.length}>
                <ul className="list-disc pl-5 text-sm text-slate-600 flex flex-col gap-1">
                  {impl.prerequisites.map((p) => (
                    <li key={p}>{p}</li>
                  ))}
                </ul>
              </Collapsible>
            )}
            {rejected.length > 0 && (
              <Collapsible title="Considered and rejected" count={rejected.length} defaultOpen>
                <ul className="flex flex-col divide-y divide-slate-100">
                  {rejected.map((r, i) => (
                    <li key={`${r.type}-${r.target_zone}-${r.rule}-${i}`} className="py-2 grid grid-cols-1 sm:grid-cols-[180px_1fr] gap-1 sm:gap-4 text-sm">
                      <span className="text-slate-900">
                        {r.type.replace(/_/g, ' ')}
                        <span className="block text-xs text-slate-500">{zoneLabel(r.target_zone)}</span>
                      </span>
                      <span className="text-slate-600">{r.reason}</span>
                    </li>
                  ))}
                </ul>
              </Collapsible>
            )}
          </div>

          <ul className="text-xs text-slate-500 flex flex-col gap-0.5 border-t border-slate-200 pt-4">
            {impl.assumptions.map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  );
};
