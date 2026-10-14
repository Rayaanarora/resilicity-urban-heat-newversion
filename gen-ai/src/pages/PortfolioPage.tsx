import React, { useEffect, useMemo, useState } from 'react';
import { Download, Play } from 'lucide-react';
import { downloadPortfolioExport, listProjects, optimizePortfolio } from '../services/plannerApi';
import { useResiliCityStore } from '../store/useResiliCityStore';
import type { Objective, OptimizeRequest, PortfolioResult, ProjectSummary } from '../types/planner';
import { Figure, SourceChip } from '../components/Figure';
import { inrLakh, zoneLabel } from '../components/format';
import { Button, Eyebrow, Panel, Tag } from '../components/ui';

const OBJECTIVES: { id: Objective; label: string; help: string }[] = [
  { id: 'person_c', label: 'Most person-°C', help: 'Cooling weighted by the people in each hotspot cell' },
  { id: 'max_cooling', label: 'Most cooling', help: 'Sum of indicative local cooling, ignoring population' },
  { id: 'min_cost_for_target', label: 'Cheapest for a target', help: 'Lowest cost that reaches a person-°C target' },
];

const PHASES = [
  { id: '', label: 'Any phase' },
  { id: 'quick_win', label: 'Quick wins only' },
  { id: 'medium', label: 'Up to 12 months' },
];

/** person-°C (or the chosen objective) against budget, with the current budget marked. */
const FrontierChart: React.FC<{ result: PortfolioResult }> = ({ result }) => {
  const pts = result.frontier;
  if (!pts.length) return <p className="text-sm text-slate-500">No frontier for this objective.</p>;
  const W = 640, H = 260, L = 56, R = 16, T = 16, B = 40;
  const maxX = Math.max(...pts.map((p) => p.budget_lakh));
  const maxY = Math.max(...pts.map((p) => p.objective_value), 1);
  const x = (v: number) => L + (v / maxX) * (W - L - R);
  const y = (v: number) => H - B - (v / maxY) * (H - T - B);
  const path = pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.budget_lakh)},${y(p.objective_value)}`).join(' ');
  const unit = result.objective === 'max_cooling' ? '°C' : 'person-°C';
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <svg key={`${result.budget_lakh}-${result.objective}`} viewBox={`0 0 ${W} ${H}`} className="w-full text-slate-500" role="img" aria-label="Efficient frontier">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t * maxY)} y2={y(t * maxY)} stroke="#e2e8f0" />
          <text x={L - 6} y={y(t * maxY) + 4} textAnchor="end" fontSize={10} fill="currentColor">
            {Math.round(t * maxY).toLocaleString('en-IN')}
          </text>
          <text x={x(t * maxX)} y={H - B + 16} textAnchor="middle" fontSize={10} fill="currentColor">
            {(t * maxX).toFixed(0)}
          </text>
        </g>
      ))}
      <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" fontSize={11} fill="currentColor">
        Budget (₹ lakh)
      </text>
      <text x={12} y={(T + H - B) / 2} textAnchor="middle" fontSize={11} fill="currentColor" transform={`rotate(-90 12 ${(T + H - B) / 2})`}>
        {unit}
      </text>
      <path d={path} fill="none" stroke="#0d7a5f" strokeWidth={2} className="rc-draw" />
      {pts.map((p) => (
        <circle key={p.budget_lakh} cx={x(p.budget_lakh)} cy={y(p.objective_value)} r={3} fill="#0d7a5f">
          <title>
            ₹{p.budget_lakh} L: {Math.round(p.objective_value).toLocaleString('en-IN')} {unit}, {p.n_items} items
          </title>
        </circle>
      ))}
      <line x1={x(result.budget_lakh)} x2={x(result.budget_lakh)} y1={T} y2={H - B} stroke="#0f172a" strokeDasharray="4 3" />
      <text x={x(result.budget_lakh) + 4} y={T + 10} fontSize={10} fill="#0f172a">
        your budget
      </text>
    </svg>
  );
};

export const PortfolioPage: React.FC = () => {
  const openProject = useResiliCityStore((s) => s.openProject);
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [chosen, setChosen] = useState<string[]>([]);
  const [budget, setBudget] = useState(25);
  const [objective, setObjective] = useState<Objective>('person_c');
  const [target, setTarget] = useState('');
  const [phase, setPhase] = useState('');
  const [result, setResult] = useState<PortfolioResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listProjects()
      .then((d) => {
        setProjects(d.projects);
        const ids = d.projects.map((p) => p.id);
        setChosen(ids);
        // Show a first result straight away with the default budget and objective.
        if (ids.length) optimizePortfolio({ project_ids: ids, budget_lakh: 25, objective: 'person_c' }).then(setResult).catch(() => undefined);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const request: OptimizeRequest = useMemo(
    () => ({
      project_ids: chosen,
      budget_lakh: budget,
      objective,
      phase_limit: (phase || null) as OptimizeRequest['phase_limit'],
      target_person_c: objective === 'min_cost_for_target' && target ? Number(target) : null,
    }),
    [chosen, budget, objective, phase, target]
  );

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setResult(await optimizePortfolio(request));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Optimisation failed');
    } finally {
      setBusy(false);
    }
  };

  const toggle = (id: string) => setChosen((c) => (c.includes(id) ? c.filter((x) => x !== id) : [...c, id]));
  const t = result?.totals;
  const demo = projects?.some((p) => p.data_mode === 'demo');

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="flex flex-col gap-1">
          <Eyebrow>Decide</Eyebrow>
          <h1 className="text-2xl sm:text-3xl font-semibold text-slate-900">Spend the budget where it cools most people</h1>
          <p className="text-sm text-slate-500 max-w-2xl">
            Chooses among the accepted interventions of the selected projects, at most one per conflicting group, to maximise
            person-°C within the budget. Cooling is a literature band, not a model prediction.
          </p>
        </div>
        {demo && <Tag tone="amber">Demo data: population is a constant</Tag>}
      </div>

      {error && <div className="border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">{error}</div>}

      <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-5 items-start">
        <div className="flex flex-col gap-5">
          <Panel eyebrow="Projects" title={projects ? `${chosen.length} of ${projects.length} selected` : 'Loading…'} bodyClassName="p-0">
            {!projects ? (
              <div className="h-32 rc-shimmer" />
            ) : projects.length === 0 ? (
              <p className="p-5 text-sm text-slate-500">Create projects first; the optimiser chooses among their accepted interventions.</p>
            ) : (
              <ul className="divide-y divide-slate-100 max-h-80 overflow-y-auto">
                {projects.map((p) => (
                  <li key={p.id}>
                    <label className="rc-row flex items-center gap-3 px-5 py-2.5 cursor-pointer">
                      <input type="checkbox" checked={chosen.includes(p.id)} onChange={() => toggle(p.id)} className="accent-[#0d7a5f]" />
                      <span className="flex-1 min-w-0">
                        <span className="block text-sm text-slate-900 truncate">{p.name}</span>
                        <span className="block text-xs text-slate-500">
                          {p.hotspot_rank ? `Rank ${p.hotspot_rank}` : 'Unlinked'} · {p.n_accepted} accepted
                        </span>
                      </span>
                    </label>
                  </li>
                ))}
              </ul>
            )}
          </Panel>

          <Panel eyebrow="Budget and objective" title={`₹${budget.toFixed(1)} lakh`}>
            <input
              type="range"
              min={1}
              max={200}
              step={0.5}
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="w-full accent-[#0d7a5f]"
              aria-label="Budget in lakh"
            />
            <div className="flex justify-between text-[11px] text-slate-400">
              <span>₹1 L</span>
              <span>₹200 L</span>
            </div>
            <div className="mt-4 flex flex-col gap-1.5">
              {OBJECTIVES.map((o) => (
                <label key={o.id} className={`flex items-start gap-2 px-3 py-2 border cursor-pointer transition-colors ${objective === o.id ? 'border-[#0d7a5f] bg-[#e8f3ef]' : 'border-slate-200 hover:border-[#b7d9cc]'}`}>
                  <input type="radio" name="objective" checked={objective === o.id} onChange={() => setObjective(o.id)} className="mt-0.5 accent-[#0d7a5f]" />
                  <span>
                    <span className="block text-sm text-slate-900">{o.label}</span>
                    <span className="block text-xs text-slate-500">{o.help}</span>
                  </span>
                </label>
              ))}
            </div>
            {objective === 'min_cost_for_target' && (
              <input
                type="number"
                min={1}
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="Target person-°C"
                className="mt-3 h-9 w-full px-3 border border-slate-300 text-sm focus:outline-none focus:border-[#0d7a5f]"
              />
            )}
            <select value={phase} onChange={(e) => setPhase(e.target.value)} className="mt-3 h-9 w-full px-3 border border-slate-300 bg-white text-sm cursor-pointer">
              {PHASES.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
            <Button variant="primary" className="mt-4 w-full" icon={<Play className="w-4 h-4" />} disabled={busy || !chosen.length} onClick={() => void run()}>
              {busy ? 'Optimising…' : 'Optimise'}
            </Button>
          </Panel>
        </div>

        <div className="flex flex-col gap-5">
          {!result ? (
            <Panel eyebrow="Results" title="Run the optimiser to see the selection">
              <p className="text-sm text-slate-500">Results, the efficient frontier and the phase schedule appear here.</p>
            </Panel>
          ) : !result.feasible ? (
            <Panel eyebrow="Results" title="No feasible selection">
              <p className="text-sm text-slate-600">{result.reason}</p>
            </Panel>
          ) : (
            <>
              <div className="rc-stagger grid grid-cols-2 lg:grid-cols-4 gap-px bg-slate-200 border border-slate-200">
                <div className="bg-white px-4 py-3">
                  <Figure label="Cost" value={t!.cost_inr.central} format={inrLakh} band={[t!.cost_inr.low, t!.cost_inr.high]} source="assumed" />
                </div>
                <div className="bg-white px-4 py-3">
                  <Figure label="Person-°C" value={t!.person_c.central} decimals={0} band={[t!.person_c.low, t!.person_c.high]} source="literature" />
                </div>
                <div className="bg-white px-4 py-3">
                  <Figure label="°C per ₹ lakh" value={t!.c_per_lakh} decimals={3} source="literature" />
                </div>
                <div className="bg-white px-4 py-3">
                  <Figure label="Items selected" value={t!.n_items} decimals={0} source="rule" />
                  <span className="text-xs text-slate-500">of {result.n_candidates} candidates</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-slate-200 border border-slate-200 text-sm">
                {(['worst', 'best'] as const).map((s) => {
                  const r = result.robustness[s];
                  return (
                    <div key={s} className="bg-white px-4 py-3">
                      <span className="rc-card-header-label">{s} case · {r.assumption}</span>
                      <p className="mt-1 text-slate-700">
                        Same selection: {inrLakh(r.same_selection.cost_inr)}, {Math.round(r.same_selection.person_c).toLocaleString('en-IN')} person-°C.{' '}
                        <span className={r.same_selection.within_budget ? 'text-[#0a6550]' : 'text-rose-700'}>
                          {r.same_selection.within_budget ? 'Within budget.' : 'Over budget.'}
                        </span>
                      </p>
                    </div>
                  );
                })}
              </div>

              <Panel eyebrow="Efficient frontier" title="What each extra lakh buys" actions={<SourceChip source="literature" />}>
                <FrontierChart result={result} />
              </Panel>

              <Panel eyebrow="Phase schedule" title="When the money is spent">
                <div className="flex flex-col gap-3">
                  {result.phase_schedule.map((ph) => {
                    const share = t!.cost_inr.central ? (ph.cost_inr_central / t!.cost_inr.central) * 100 : 0;
                    return (
                      <div key={ph.phase} className="grid grid-cols-[180px_1fr_110px] items-center gap-3 text-sm">
                        <span className="text-slate-900">{ph.label}</span>
                        <span className="h-3 bg-slate-100 block">
                          <span className="rc-bar block h-full bg-[#0d7a5f]" style={{ width: `${share}%` }} />
                        </span>
                        <span className="text-right tabular-nums text-slate-700">
                          {inrLakh(ph.cost_inr_central)} · {ph.items.length}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </Panel>

              <Panel
                eyebrow="Selection"
                title="Interventions to fund"
                actions={
                  <>
                    {(['pdf', 'csv', 'json'] as const).map((f) => (
                      <Button key={f} icon={<Download className="w-4 h-4" />} onClick={() => void downloadPortfolioExport(request, f)}>
                        {f.toUpperCase()}
                      </Button>
                    ))}
                  </>
                }
                bodyClassName="p-0"
              >
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[720px] text-sm border-collapse">
                    <thead>
                      <tr className="text-left border-b border-slate-300">
                        {['Project', 'Intervention', 'Cost', 'Cooling', 'Person-°C', 'Phase'].map((h) => (
                          <th key={h} className="py-2 pr-4 first:pl-5 rc-card-header-label">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="rc-stagger">
                      {result.selected.flatMap((p) =>
                        p.items.map((it) => (
                          <tr key={it.key} className="rc-row border-b border-slate-100">
                            <td className="py-2.5 pr-4 pl-5">
                              <button type="button" onClick={() => openProject(p.project_id)} className="text-slate-900 hover:text-[#0d7a5f] cursor-pointer text-left">
                                {p.name}
                              </button>
                            </td>
                            <td className="py-2.5 pr-4">
                              {it.title}
                              <span className="block text-xs text-slate-500">{zoneLabel(it.zone)}</span>
                            </td>
                            <td className="py-2.5 pr-4 tabular-nums whitespace-nowrap">{inrLakh(it.cost_inr.central)}</td>
                            <td className="py-2.5 pr-4 tabular-nums whitespace-nowrap">
                              {it.cooling_c.central.toFixed(1)} °C <span className="text-xs text-slate-500">({it.cooling_c.low.toFixed(1)}–{it.cooling_c.high.toFixed(1)})</span>
                            </td>
                            <td className="py-2.5 pr-4 tabular-nums">{Math.round(it.person_c.central).toLocaleString('en-IN')}</td>
                            <td className="py-2.5 pr-4 text-slate-600">{it.phase.replace('_', ' ')}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </Panel>

              <ul className="text-xs text-slate-500 flex flex-col gap-1">
                <li>
                  Exact optimum (dynamic programming); greedy cross-check {Math.round(result.greedy_cross_check.objective_value).toLocaleString('en-IN')}
                  {result.greedy_cross_check.dp_at_least_greedy ? ', optimum is at least as good.' : '.'}
                </li>
                {result.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
