import React, { useCallback, useEffect, useState } from 'react';
import { Plus, Trash2 } from 'lucide-react';
import { apiUrl, deleteProject, listProjects } from '../services/plannerApi';
import { useResiliCityStore } from '../store/useResiliCityStore';
import type { ProjectSummary } from '../types/planner';
import { SourceChip } from '../components/Figure';
import { inrLakh } from '../components/format';
import { Button, Eyebrow, Panel, Tag } from '../components/ui';
import { NewProjectDialog } from './NewProjectDialog';

export const ProjectsPage: React.FC = () => {
  const { openProject, setActiveTab } = useResiliCityStore();
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const load = useCallback(() => {
    listProjects()
      .then((d) => setProjects(d.projects))
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(load, [load]);

  const remove = async (p: ProjectSummary) => {
    if (!window.confirm(`Delete project "${p.name}"? This removes its photo, plan and measurements.`)) return;
    await deleteProject(p.id);
    load();
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="flex flex-col gap-1">
          <Eyebrow>Plan</Eyebrow>
          <h1 className="text-2xl sm:text-3xl font-semibold text-slate-900">Street projects</h1>
          <p className="text-sm text-slate-500 max-w-2xl">
            Each project is one street: its photo, measured dimensions, the gated plan with costs and permits, and any
            post-construction measurements.
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => setActiveTab('map')}>Pick from map</Button>
          <Button variant="primary" icon={<Plus className="w-4 h-4" />} onClick={() => setCreating(true)}>
            New project
          </Button>
        </div>
      </div>

      {error && <div className="border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">{error}</div>}

      <Panel eyebrow="Projects" title={projects ? `${projects.length} saved` : 'Loading…'} bodyClassName="p-0">
        {!projects ? (
          <div className="h-40 rc-shimmer" />
        ) : projects.length === 0 ? (
          <div className="p-8 text-center text-sm text-slate-500 flex flex-col items-center gap-3">
            <span>No projects yet. Start from a hotspot on the map, or create one from a photo and a location.</span>
            <Button variant="primary" icon={<Plus className="w-4 h-4" />} onClick={() => setCreating(true)}>
              New project
            </Button>
          </div>
        ) : (
          <ul className="rc-stagger divide-y divide-slate-200">
            {projects.map((p) => (
              <li key={p.id} className="rc-row flex items-center gap-4 px-5 py-3">
                <button type="button" onClick={() => openProject(p.id)} className="flex items-center gap-4 flex-1 min-w-0 text-left cursor-pointer">
                  <img src={apiUrl(p.photo_url)} alt="" className="w-20 h-14 object-cover border border-slate-200 shrink-0" />
                  <span className="flex flex-col min-w-0">
                    <span className="text-sm text-slate-900 truncate">{p.name}</span>
                    <span className="text-xs text-slate-500">
                      {p.hotspot_rank ? `Hotspot rank ${p.hotspot_rank}` : 'Not linked to a hotspot'}
                      {p.lst_pred_c ? ` · ${p.lst_pred_c.toFixed(1)} °C predicted` : ''}
                    </span>
                  </span>
                </button>
                <span className="hidden md:block text-xs text-slate-600 w-40">
                  {p.n_accepted} accepted · {p.n_rejected} rejected
                </span>
                <span className="hidden md:flex items-center gap-2 text-xs text-slate-900 w-48 tabular-nums">
                  {p.cost_inr_low != null && p.cost_inr_high != null ? `${inrLakh(p.cost_inr_low)} – ${inrLakh(p.cost_inr_high)}` : '—'}
                  <SourceChip source="assumed" />
                </span>
                <span className="hidden lg:block w-28">
                  {p.redesign_status === 'ready' ? <Tag tone="green">Image ready</Tag> : <Tag>No image</Tag>}
                </span>
                {p.data_mode === 'demo' && <Tag tone="amber">Demo</Tag>}
                <button type="button" onClick={() => void remove(p)} className="text-slate-400 hover:text-rose-700 cursor-pointer" title="Delete project">
                  <Trash2 className="w-4 h-4" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      {creating && <NewProjectDialog onClose={() => setCreating(false)} />}
    </div>
  );
};
