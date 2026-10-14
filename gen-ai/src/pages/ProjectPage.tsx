import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowLeft, ChevronDown, Download, ImagePlus, RefreshCw, Upload } from 'lucide-react';
import {
  apiUrl,
  getProject,
  projectExportUrl,
  replanProject,
  requestRedesign,
  uploadMeasurement,
} from '../services/plannerApi';
import { useResiliCityStore } from '../store/useResiliCityStore';
import type { AcceptedItem, Measurement, Project } from '../types/planner';
import { Figure, SourceChip } from '../components/Figure';
import { inrLakh, zoneLabel } from '../components/format';
import { ComparisonViewer } from '../components/ProposalSection';
import { StreetCrossSection } from '../components/StreetCrossSection';
import { Button, Eyebrow, Panel, Tag } from '../components/ui';

const inputCls = 'h-9 w-36 px-3 border border-slate-300 bg-white text-sm text-slate-900 focus:outline-none focus:border-[#0d7a5f]';

const AcceptedRow: React.FC<{ item: AcceptedItem; index: number }> = ({ item, index }) => {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr className="rc-row border-b border-slate-100 align-top cursor-pointer" onClick={() => setOpen(!open)}>
        <td className="py-3 pr-3 text-xs text-[#0d7a5f] tabular-nums">{String(index + 1).padStart(2, '0')}</td>
        <td className="py-3 pr-4">
          <div className="text-slate-900">{item.title}</div>
          <div className="text-xs text-slate-500">{zoneLabel(item.target_zone)}</div>
        </td>
        <td className="py-3 pr-4 whitespace-nowrap tabular-nums">
          {item.quantity.value} {item.quantity.unit} <SourceChip source={item.quantity.source} />
        </td>
        <td className="py-3 pr-4 whitespace-nowrap tabular-nums">
          {inrLakh(item.cost_inr.central)} <SourceChip source="assumed" />
          <div className="text-xs text-slate-500">
            {inrLakh(item.cost_inr.low)} to {inrLakh(item.cost_inr.high)}
          </div>
        </td>
        <td className="py-3 pr-4 whitespace-nowrap tabular-nums">
          {item.cooling_c ? (
            <>
              {item.cooling_c.central.toFixed(1)} °C <SourceChip source="literature" />
              <div className="text-xs text-slate-500">
                {item.cooling_c.low.toFixed(1)} to {item.cooling_c.high.toFixed(1)} °C
              </div>
            </>
          ) : (
            '—'
          )}
        </td>
        <td className="py-3 pr-4 text-slate-600">{item.phase_label}</td>
        <td className="py-3">
          <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform duration-300 ${open ? 'rotate-180' : ''}`} />
        </td>
      </tr>
      <tr>
        <td colSpan={7} className="p-0">
          <div className="rc-collapse" data-open={open}>
            <div>
              <div className="pl-10 pr-4 pb-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-600">
                <div>
                  <span className="rc-card-header-label">Quantity basis</span>
                  <p className="mt-1">{item.quantity.basis}</p>
                  <p className="mt-1">{item.lifespan_years} year life. {item.maintenance}</p>
                </div>
                <div>
                  <span className="rc-card-header-label">Permits and prerequisites</span>
                  <ul className="mt-1 list-disc pl-4">
                    {item.permits.map((p) => (
                      <li key={p}>{p}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <span className="rc-card-header-label">Verify on site</span>
                  {item.conditions.length ? (
                    <ul className="mt-1 list-disc pl-4 text-amber-800">
                      {item.conditions.map((c) => (
                        <li key={c}>{c}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-1">No extra conditions.</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </td>
      </tr>
    </>
  );
};

/** Predicted (literature cooling band, shown as a negative change) vs observed difference-in-differences. */
const ObservedVsPredicted: React.FC<{ m: Measurement }> = ({ m }) => {
  const r = m.result;
  const pred = r.predicted_cooling_c;
  const vals = [r.did_c ?? 0, ...(r.interval_95_c ?? []), ...(pred ? [-pred.high, -pred.low] : []), 0];
  const lo = Math.min(...vals) - 0.5;
  const hi = Math.max(...vals) + 0.5;
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  return (
    <div className="flex flex-col gap-3">
      <div className="relative h-20 border-y border-slate-200">
        <span className="absolute top-0 bottom-0 w-px bg-slate-400" style={{ left: x(0) }} />
        <span className="absolute -top-5 text-[10px] text-slate-500 -translate-x-1/2" style={{ left: x(0) }}>
          0 °C
        </span>
        {pred && (
          <span
            className="absolute top-3 h-4 bg-sky-100 border border-sky-700/50"
            style={{ left: x(-pred.high), width: `calc(${x(-pred.low)} - ${x(-pred.high)})` }}
            title="Predicted cooling, literature band"
          />
        )}
        {pred && (
          <span className="absolute top-3 h-4 w-0.5 bg-sky-800" style={{ left: x(-pred.central) }} />
        )}
        {r.did_c !== null && r.interval_95_c && (
          <>
            <span
              className="absolute bottom-3 h-px bg-slate-900"
              style={{ left: x(r.interval_95_c[0]), width: `calc(${x(r.interval_95_c[1])} - ${x(r.interval_95_c[0])})` }}
            />
            <span className="absolute bottom-[7px] w-2.5 h-2.5 bg-slate-900 -translate-x-1/2" style={{ left: x(r.did_c) }} />
          </>
        )}
      </div>
      <div className="flex flex-wrap gap-4 text-xs text-slate-600">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 bg-sky-100 border border-sky-700/50" /> Predicted cooling (literature)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 bg-slate-900" /> Observed vs controls, 95% interval (measured)
        </span>
      </div>
    </div>
  );
};

export const ProjectPage: React.FC = () => {
  const { activeProjectId, setActiveTab } = useResiliCityStore();
  const [project, setProject] = useState<Project | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<'plan' | 'monitor'>('plan');
  const [busy, setBusy] = useState(false);
  const [inputs, setInputs] = useState({ road: '', climate: 'tropical_humid', budget: '' });
  const fileRef = useRef<HTMLInputElement>(null);

  // Push project data into the shared store so the existing section components render it.
  const apply = useCallback((p: Project) => {
    setProject(p);
    useResiliCityStore.setState({
      spatialDesignPlan: p.plan,
      thermalImpact: p.thermal as never,
      rawImageUrl: apiUrl(p.photo_url),
      currentFile: null,
      generatedImageUrl: p.redesign.image_url ? `${apiUrl(p.redesign.image_url)}?v=${p.updated_at}` : null,
      visualizationOutput: {
        status: p.redesign.status === 'failed' ? 'unavailable' : 'ready',
        image_url: p.redesign.image_url,
        provider: '',
        model: '',
        quality_tier: 'fast',
        generation_time_ms: 0,
        refinement_count: 0,
        error_message: p.redesign.error,
        illustrated_interventions: p.redesign.illustrated_interventions,
        cached: p.redesign.cached,
      },
      isGenerating: p.redesign.status === 'queued' || p.redesign.status === 'running',
    });
  }, []);

  const load = useCallback(() => {
    if (!activeProjectId) return;
    getProject(activeProjectId)
      .then((p) => {
        apply(p);
        setInputs({
          road: p.inputs.road_width_m ? String(p.inputs.road_width_m) : '',
          climate: p.inputs.climate_zone ?? 'tropical_humid',
          budget: p.inputs.budget_lakh ? String(p.inputs.budget_lakh) : '',
        });
      })
      .catch((e: Error) => setError(e.message));
  }, [activeProjectId, apply]);

  useEffect(load, [load]);

  // Poll while the illustrative image is being generated.
  useEffect(() => {
    if (!project || !['queued', 'running'].includes(project.redesign.status)) return;
    const t = setInterval(() => getProject(project.id).then(apply).catch(() => undefined), 3000);
    return () => clearInterval(t);
  }, [project, apply]);

  useEffect(() => () => useResiliCityStore.setState({ isGenerating: false }), []);

  if (!activeProjectId) {
    return (
      <Panel eyebrow="Project" title="No project selected">
        <Button onClick={() => setActiveTab('projects')}>Go to projects</Button>
      </Panel>
    );
  }
  if (error) return <div className="border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">{error}</div>;
  if (!project) return <div className="h-96 rc-shimmer" />;

  const hs = project.hotspot;
  const m = project.street_metrics;
  const impl = project.implementation_plan;

  const replan = async () => {
    setBusy(true);
    try {
      apply(await replanProject(project.id, inputs.road ? Number(inputs.road) : null, inputs.climate, inputs.budget ? Number(inputs.budget) : null));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Re-plan failed');
    } finally {
      setBusy(false);
    }
  };

  const generate = async () => {
    await requestRedesign(project.id);
    load();
  };

  const upload = async (file: File) => {
    setBusy(true);
    try {
      await uploadMeasurement(project.id, file);
      setTab('monitor');
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-4">
        <button type="button" onClick={() => setActiveTab('projects')} className="self-start flex items-center gap-1.5 text-sm text-slate-500 hover:text-[#0d7a5f] cursor-pointer">
          <ArrowLeft className="w-4 h-4" /> All projects
        </button>
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div className="flex flex-col gap-1">
            <Eyebrow>Street project</Eyebrow>
            <h1 className="text-2xl sm:text-3xl font-semibold text-slate-900">{project.name}</h1>
            <div className="flex flex-wrap gap-2">
              {project.data_mode === 'demo' && <Tag tone="amber">Demo data</Tag>}
              {hs && <Tag>{hs.link === 'selected' ? 'Hotspot selected on map' : `Nearest hotspot cell, ${Math.round(hs.distance_m ?? 0)} m away`}</Tag>}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {(['pdf', 'csv', 'json'] as const).map((f) => (
              <a key={f} href={projectExportUrl(project.id, f)} className="inline-flex items-center gap-2 px-3 h-9 text-sm border border-slate-300 text-slate-700 hover:border-[#0d7a5f] hover:text-[#0d7a5f] transition-colors">
                <Download className="w-4 h-4" /> {f.toUpperCase()}
              </a>
            ))}
          </div>
        </div>
      </div>

      {hs && (
        <div className="rc-stagger grid grid-cols-2 md:grid-cols-4 gap-px bg-slate-200 border border-slate-200">
          {[
            <Figure key="r" label="Hotspot rank" value={hs.rank} decimals={0} source="model" />,
            <Figure key="t" label="Predicted LST" value={hs.lst_pred_c} unit=" °C" band={hs.lst_interval_90_c} source="model" />,
            <Figure key="e" label="Above city median" value={hs.heat_excess_c} unit=" °C" source="model" />,
            <Figure key="p" label="Population in cell" value={hs.population} decimals={0} source={project.data_mode === 'grid' ? 'measured' : 'assumed'} />,
          ].map((f, i) => (
            <div key={i} className="bg-white px-4 py-3">
              {f}
            </div>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="flex border-b border-slate-200">
        {(['plan', 'monitor'] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={`px-4 py-2.5 text-sm border-b-2 -mb-px cursor-pointer ${tab === t ? 'border-[#0d7a5f] text-slate-900' : 'border-transparent text-slate-500 hover:text-slate-900'}`}
          >
            {t === 'plan' ? 'Plan' : `Monitor${project.measurements?.length ? ` (${project.measurements.length})` : ''}`}
          </button>
        ))}
      </div>

      {tab === 'plan' ? (
        <div className="flex flex-col gap-6">
          <Panel eyebrow="Inputs" title="Street conditions">
            <div className="flex flex-wrap items-end gap-4">
              <label className="flex flex-col gap-1.5">
                <span className="rc-card-header-label">Carriageway (m)</span>
                <input type="number" min={3} step={0.5} value={inputs.road} placeholder="7.0 assumed" onChange={(e) => setInputs({ ...inputs, road: e.target.value })} className={inputCls} />
              </label>
              <label className="flex flex-col gap-1.5">
                <span className="rc-card-header-label">Climate</span>
                <select value={inputs.climate} onChange={(e) => setInputs({ ...inputs, climate: e.target.value })} className={`${inputCls} w-44 cursor-pointer`}>
                  <option value="tropical_humid">Tropical humid</option>
                  <option value="arid">Arid</option>
                  <option value="temperate">Temperate</option>
                  <option value="mediterranean">Mediterranean</option>
                </select>
              </label>
              <label className="flex flex-col gap-1.5">
                <span className="rc-card-header-label">Budget (₹ lakh)</span>
                <input type="number" min={0} step={0.5} value={inputs.budget} placeholder="No cap" onChange={(e) => setInputs({ ...inputs, budget: e.target.value })} className={inputCls} />
              </label>
              <Button variant="primary" icon={<RefreshCw className="w-4 h-4" />} disabled={busy} onClick={() => void replan()}>
                {busy ? 'Re-planning…' : 'Re-plan'}
              </Button>
            </div>
            {m && (
              <div className="mt-5 grid grid-cols-2 md:grid-cols-4 gap-4">
                <Figure label="Carriageway" value={m.road_width_m} unit=" m" source={m.road_width_source} />
                <Figure label="Left footpath" value={m.left_sidewalk_m ?? null} unit=" m" band={m.left_range_m ?? null} source="measured" />
                <Figure label="Right footpath" value={m.right_sidewalk_m ?? null} unit=" m" band={m.right_range_m ?? null} source="measured" />
                <Figure label="Width uncertainty" value={m.uncertainty_pct} unit="%" decimals={0} source="rule" />
              </div>
            )}
          </Panel>

          <Panel
            eyebrow="Gated plan"
            title={impl ? `Accepted per ${impl.street_length_m} m: ${inrLakh(impl.total_cost_inr_low)} to ${inrLakh(impl.total_cost_inr_high)}` : 'Accepted interventions'}
            actions={<SourceChip source="assumed" />}
            bodyClassName="p-0"
          >
            {project.accepted.length === 0 ? (
              <p className="p-5 text-sm text-slate-500">No intervention passed the rules for this street; see the rejected options below.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[760px] text-sm border-collapse">
                  <thead>
                    <tr className="text-left border-b border-slate-300">
                      {['', 'Intervention', 'Quantity', 'Cost', 'Cooling', 'Phase', ''].map((h, i) => (
                        <th key={i} className="py-2 pr-4 first:pl-5 rc-card-header-label">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="[&_td:first-child]:pl-5">
                    {project.accepted.map((a, i) => (
                      <AcceptedRow key={`${a.type}-${a.target_zone}`} item={a} index={i} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>

          {project.rejected.length > 0 && (
            <Panel eyebrow="Considered and rejected" title={`${project.rejected.length} options, with the rule that excluded each`} bodyClassName="p-0">
              <ul className="rc-stagger divide-y divide-slate-100">
                {project.rejected.map((r, i) => (
                  <li key={`${r.type}-${r.target_zone}-${r.rule}-${i}`} className="px-5 py-3 grid grid-cols-1 md:grid-cols-[200px_1fr_220px] gap-2 text-sm">
                    <span className="text-slate-900">
                      {r.type.replace(/_/g, ' ')}
                      <span className="block text-xs text-slate-500">{zoneLabel(r.target_zone)}</span>
                    </span>
                    <span className="text-slate-600">
                      {r.reason} <SourceChip source="rule" />
                    </span>
                    <span className="text-xs text-slate-500 font-mono break-words">
                      {Object.entries(r.measured).map(([k, v]) => `${k}=${Array.isArray(v) ? v.join('–') : String(v)}`).join(', ')}
                    </span>
                  </li>
                ))}
              </ul>
            </Panel>
          )}

          <StreetCrossSection />

          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <span className="text-sm text-slate-600">
                Illustrative image: {project.redesign.status === 'ready' ? 'ready' : project.redesign.status === 'failed' ? 'failed' : project.redesign.status}
              </span>
              <Button
                variant="primary"
                icon={<ImagePlus className="w-4 h-4" />}
                disabled={['queued', 'running'].includes(project.redesign.status) || project.accepted.length === 0}
                onClick={() => void generate()}
              >
                {['queued', 'running'].includes(project.redesign.status) ? 'Generating…' : project.redesign.status === 'ready' ? 'Regenerate image' : 'Generate image'}
              </Button>
            </div>
            <ComparisonViewer />
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-6">
          <Panel
            eyebrow="Verify"
            title="Post-construction measurements"
            actions={
              <>
                <input ref={fileRef} type="file" accept=".json,.csv" className="hidden" onChange={(e) => e.target.files?.[0] && void upload(e.target.files[0])} />
                <Button variant="primary" icon={<Upload className="w-4 h-4" />} disabled={busy} onClick={() => fileRef.current?.click()}>
                  Upload measurement
                </Button>
              </>
            }
          >
            <p className="text-sm text-slate-600 max-w-3xl">
              Upload the JSON or CSV from <span className="font-mono text-xs">data/export_monitor.py</span>: Landsat surface
              temperature for this cell and comparable untreated control cells, before and after construction.
            </p>
          </Panel>

          {!project.measurements?.length ? (
            <Panel eyebrow="Results" title="No measurements yet">
              <p className="text-sm text-slate-500">Nothing to compare until a post-construction measurement is uploaded.</p>
            </Panel>
          ) : (
            project.measurements
              .slice()
              .reverse()
              .map((ms) => (
                <Panel
                  key={ms.id}
                  eyebrow={`${ms.result.pre_period ?? 'pre'} to ${ms.result.post_period ?? 'post'}`}
                  title={ms.result.supports_claim ? 'Change detected relative to controls' : 'No supportable claim'}
                  actions={<Tag tone={ms.result.supports_claim ? 'green' : 'amber'}>{ms.result.n_controls} controls</Tag>}
                >
                  <p className="text-sm text-slate-900">{ms.result.statement}</p>
                  <div className="mt-5 grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Figure label="Observed vs controls" value={ms.result.did_c} unit=" °C" band={ms.result.interval_95_c} source="measured" decimals={2} />
                    <Figure
                      label="Predicted cooling"
                      value={ms.result.predicted_cooling_c?.central ?? null}
                      unit=" °C"
                      band={ms.result.predicted_cooling_c ? [ms.result.predicted_cooling_c.low, ms.result.predicted_cooling_c.high] : null}
                      source="literature"
                      decimals={2}
                    />
                    <Figure label="Project cell change" value={ms.result.project_change_c} unit=" °C" source="measured" decimals={2} />
                  </div>
                  <div className="mt-6">
                    <ObservedVsPredicted m={ms} />
                  </div>
                  <p className="mt-4 text-xs text-slate-500">{ms.result.comparison_note}</p>
                </Panel>
              ))
          )}
        </div>
      )}
    </div>
  );
};
