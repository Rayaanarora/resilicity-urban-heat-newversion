import React from 'react';
import { ClipboardList, Ban, CalendarClock, FileCheck2, RefreshCw } from 'lucide-react';
import { useResiliCityStore } from '../store/useResiliCityStore';

const LAKH = 100_000;

const inr = (n: number) => (n >= LAKH ? `₹${(n / LAKH).toFixed(1)} L` : `₹${Math.round(n).toLocaleString('en-IN')}`);
const inrRange = (lo: number, hi: number) => `${inr(lo)} – ${inr(hi)}`;
const side = (zone?: string | null) =>
  zone === 'left_sidewalk' ? 'Left' : zone === 'right_sidewalk' ? 'Right' : zone === 'roadway' ? 'Road' : '—';
const qty = (q: number) => (Number.isInteger(q) ? q.toString() : q.toFixed(1));

const parseOptional = (v: string): number | null => {
  const n = Number(v);
  return v.trim() === '' || !Number.isFinite(n) || n <= 0 ? null : n;
};

export const ImplementationPlanCard: React.FC = () => {
  const {
    spatialDesignPlan: plan,
    thermalImpact,
    roadWidthM,
    setRoadWidthM,
    budgetLakh,
    setBudgetLakh,
    isGenerating,
    generateResilientDesign,
  } = useResiliCityStore();
  const impl = plan?.implementation_plan;
  const rejected = (plan?.rejected_candidates ?? []).filter((r) => r.rule);

  return (
    <div className="rc-card p-6 flex flex-col gap-5">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <span className="rc-card-header-label">
            IMPLEMENTATION PLAN{impl ? ` · PER ${impl.street_length_m.toFixed(0)} M OF STREET` : ''}
          </span>
          <h3 className="text-xl font-semibold text-slate-900 font-sans">Bill of quantities &amp; phasing</h3>
        </div>
        <span className="text-[11px] font-mono font-medium text-amber-800 bg-amber-50 border border-amber-200 px-2.5 py-1 rounded-lg">
          Indicative rates · not for tender
        </span>
      </div>

      {/* Site inputs */}
      <div className="flex flex-wrap items-end gap-3 text-xs font-sans">
        <label className="flex flex-col gap-1">
          <span className="text-slate-600 font-medium">Carriageway width (m)</span>
          <input
            type="number" min={3} step={0.5} placeholder="7.0 (assumed)"
            defaultValue={roadWidthM ?? ''}
            onChange={(e) => setRoadWidthM(parseOptional(e.target.value))}
            className="w-36 px-3 py-2 rounded-lg border border-slate-300 bg-white text-slate-900"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-slate-600 font-medium">Budget cap (₹ lakh)</span>
          <input
            type="number" min={0} step={0.5} placeholder="No cap"
            defaultValue={budgetLakh ?? ''}
            onChange={(e) => setBudgetLakh(parseOptional(e.target.value))}
            className="w-36 px-3 py-2 rounded-lg border border-slate-300 bg-white text-slate-900"
          />
        </label>
        <button
          type="button"
          disabled={isGenerating}
          onClick={() => void generateResilientDesign('fast')}
          className="px-4 py-2 rc-btn-primary rounded-lg flex items-center gap-2 disabled:opacity-50 cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          {isGenerating ? 'Re-planning…' : 'Re-plan'}
        </button>
      </div>

      {!impl ? (
        <p className="text-sm text-slate-500">Run an analysis to produce the implementation plan.</p>
      ) : (
        <>
          {/* Summary */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
              <div className="text-[10px] font-mono uppercase text-slate-500">Indicative cost</div>
              <div className="text-sm font-medium text-slate-900">{inrRange(impl.total_cost_inr_low, impl.total_cost_inr_high)}</div>
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
              <div className="text-[10px] font-mono uppercase text-slate-500">Est. local cooling</div>
              <div className="text-sm font-medium text-teal-700">
                {impl.total_cooling_c.toFixed(1)} °C
                {thermalImpact?.coolingRangeC && (
                  <span className="font-normal text-slate-500">
                    {' '}({thermalImpact.coolingRangeC[0].toFixed(1)}–{thermalImpact.coolingRangeC[1].toFixed(1)})
                  </span>
                )}
              </div>
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
              <div className="text-[10px] font-mono uppercase text-slate-500">Cooling per lakh</div>
              <div className="text-sm font-medium text-slate-900">{impl.cooling_per_lakh_c.toFixed(2)} °C</div>
            </div>
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-3">
              <div className="text-[10px] font-mono uppercase text-slate-500">Budget</div>
              <div className={`text-sm font-medium ${impl.within_budget === false ? 'text-red-600' : 'text-slate-900'}`}>
                {impl.budget_lakh ? `${impl.within_budget ? 'Within' : 'Over'} ₹${impl.budget_lakh} L cap` : 'No cap set'}
              </div>
            </div>
          </div>

          {/* Bill of quantities */}
          <div className="overflow-x-auto -mx-2 px-2">
            <table className="w-full min-w-[640px] text-xs font-sans border-collapse">
              <thead>
                <tr className="text-left text-[10px] font-mono uppercase text-slate-500 border-b border-slate-200">
                  <th className="py-2 pr-3">Item</th>
                  <th className="py-2 pr-3">Side</th>
                  <th className="py-2 pr-3">Quantity</th>
                  <th className="py-2 pr-3">Cost range</th>
                  <th className="py-2 pr-3">°C / lakh</th>
                  <th className="py-2">Phase · life</th>
                </tr>
              </thead>
              <tbody>
                {impl.line_items.map((li) => (
                  <tr key={`${li.type}-${li.target_zone}`} className="border-b border-slate-100 align-top">
                    <td className="py-2 pr-3">
                      <div className="font-medium text-slate-900">{li.title}</div>
                      <div className="text-[11px] text-slate-500">{li.basis}</div>
                      {li.conditions.map((c) => (
                        <div key={c} className="text-[11px] text-amber-800">Condition: {c}</div>
                      ))}
                    </td>
                    <td className="py-2 pr-3 text-slate-600">{side(li.target_zone)}</td>
                    <td className="py-2 pr-3 font-mono text-slate-800 whitespace-nowrap">
                      {qty(li.quantity)} {li.unit}
                      {li.quantity_flagged && <span className="text-amber-700" title="Uses a default, not a measurement"> *</span>}
                      {li.extra?.kwp ? <div className="text-[11px] text-slate-500">{li.extra.kwp} kWp</div> : null}
                    </td>
                    <td className="py-2 pr-3 font-mono text-slate-800 whitespace-nowrap">
                      {inrRange(li.cost_inr_low, li.cost_inr_high)}
                    </td>
                    <td className="py-2 pr-3 font-mono text-teal-700">{li.cooling_per_lakh_c.toFixed(2)}</td>
                    <td className="py-2 text-slate-600">
                      {li.phase_label}
                      <div className="text-[11px] text-slate-400">{li.lifespan_years} yr life</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Phases */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {impl.phases.map((p) => (
              <div key={p.phase} className="border border-slate-200 rounded-xl p-3 flex flex-col gap-1.5">
                <div className="flex items-center gap-1.5 text-xs font-medium text-slate-800">
                  <CalendarClock className="w-3.5 h-3.5 text-teal-600" />
                  {p.label}
                </div>
                <ul className="text-[11px] text-slate-600 list-disc pl-4">
                  {p.items.map((t, i) => (
                    <li key={`${t}-${i}`}>{t}</li>
                  ))}
                </ul>
                <div className="text-[11px] font-mono text-slate-500">{inrRange(p.cost_inr_low, p.cost_inr_high)}</div>
              </div>
            ))}
          </div>

          {/* Prerequisites */}
          {impl.prerequisites.length > 0 && (
            <details className="border border-slate-200 rounded-xl p-3 text-xs">
              <summary className="cursor-pointer font-medium text-slate-800 flex items-center gap-1.5">
                <FileCheck2 className="w-3.5 h-3.5 text-teal-600" />
                Prerequisites &amp; permits ({impl.prerequisites.length})
              </summary>
              <ul className="mt-2 list-disc pl-5 text-slate-600 flex flex-col gap-0.5">
                {impl.prerequisites.map((p) => (
                  <li key={p}>{p}</li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}

      {/* Rejected with reasons */}
      {rejected.length > 0 && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-1.5 text-xs font-medium text-slate-800">
            <Ban className="w-3.5 h-3.5 text-rose-600" />
            Considered and rejected ({rejected.length})
          </div>
          <ul className="flex flex-col gap-1.5">
            {rejected.map((r, i) => (
              <li key={`${r.type}-${r.target_zone}-${r.rule}-${i}`}
                  className="text-[11px] font-sans bg-rose-50/60 border border-rose-100 rounded-lg px-3 py-2 text-slate-700">
                <span className="font-medium text-rose-800">
                  {r.type.replace(/_/g, ' ')}
                  {r.target_zone ? ` · ${side(r.target_zone)}` : ''}:
                </span>{' '}
                {r.reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Assumptions */}
      {(impl || thermalImpact?.confidenceNote) && (
        <div className="flex items-start gap-2 text-[11px] text-slate-500 border-t border-slate-100 pt-3">
          <ClipboardList className="w-3.5 h-3.5 mt-0.5 shrink-0" />
          <div className="flex flex-col gap-0.5">
            {impl?.assumptions.map((a) => <span key={a}>{a}</span>)}
            {thermalImpact?.confidenceNote && <span>{thermalImpact.confidenceNote}</span>}
          </div>
        </div>
      )}
    </div>
  );
};
