import React, { useState } from 'react';
import { X } from 'lucide-react';
import { createProject } from '../services/plannerApi';
import { useResiliCityStore } from '../store/useResiliCityStore';
import type { HotspotCell } from '../types/planner';
import { Button, Eyebrow } from '../components/ui';

const inputCls = 'h-9 w-full px-3 border border-slate-300 bg-white text-sm text-slate-900 focus:outline-none focus:border-[#0d7a5f] transition-colors';

const CLIMATES = [
  { id: 'tropical_humid', label: 'Tropical humid' },
  { id: 'arid', label: 'Arid' },
  { id: 'temperate', label: 'Temperate' },
  { id: 'mediterranean', label: 'Mediterranean' },
];

export const NewProjectDialog: React.FC<{ cell?: HotspotCell | null; onClose: () => void }> = ({ cell, onClose }) => {
  const openProject = useResiliCityStore((s) => s.openProject);
  const [photo, setPhoto] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [lat, setLat] = useState(cell ? String(cell.lat) : '');
  const [lon, setLon] = useState(cell ? String(cell.lon) : '');
  const [road, setRoad] = useState('');
  const [climate, setClimate] = useState('tropical_humid');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!photo) return setError('Choose a street photo.');
    if (!cell && (!lat || !lon)) return setError('Enter a latitude and longitude, or start from a map cell.');
    setBusy(true);
    setError(null);
    try {
      const p = await createProject({
        photo,
        name: name || undefined,
        cellId: cell?.cell_id,
        lat: cell ? undefined : Number(lat),
        lon: cell ? undefined : Number(lon),
        roadWidthM: road ? Number(road) : undefined,
        climateZone: climate,
      });
      openProject(p.id);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create the project.');
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[1000] bg-slate-900/40 flex items-center justify-center p-4 rc-fade-in" onClick={onClose}>
      <form
        onSubmit={submit}
        onClick={(e) => e.stopPropagation()}
        className="rc-card w-full max-w-lg bg-white flex flex-col"
        style={{ animation: 'rc-rise 0.3s cubic-bezier(0.2,0.7,0.2,1) both' }}
      >
        <header className="flex items-start justify-between px-5 py-4 border-b border-slate-200">
          <div className="flex flex-col gap-0.5">
            <Eyebrow>New street project</Eyebrow>
            <h3 className="text-base font-semibold text-slate-900">
              {cell ? `At hotspot rank ${cell.rank} (${cell.lst_pred_c.toFixed(1)} °C)` : 'Photo and location'}
            </h3>
          </div>
          <button type="button" onClick={onClose} className="text-slate-400 hover:text-slate-700 cursor-pointer" aria-label="Close">
            <X className="w-5 h-5" />
          </button>
        </header>

        <div className="p-5 flex flex-col gap-4">
          <label className="flex flex-col gap-1.5">
            <span className="rc-card-header-label">Street photo</span>
            <input type="file" accept="image/*" onChange={(e) => setPhoto(e.target.files?.[0] ?? null)} className="text-sm" />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="rc-card-header-label">Name (optional)</span>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Anna Salai, east side" className={inputCls} />
          </label>
          {!cell && (
            <div className="grid grid-cols-2 gap-3">
              <label className="flex flex-col gap-1.5">
                <span className="rc-card-header-label">Latitude</span>
                <input value={lat} onChange={(e) => setLat(e.target.value)} placeholder="13.0500" className={inputCls} />
              </label>
              <label className="flex flex-col gap-1.5">
                <span className="rc-card-header-label">Longitude</span>
                <input value={lon} onChange={(e) => setLon(e.target.value)} placeholder="80.2200" className={inputCls} />
              </label>
            </div>
          )}
          <div className="grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1.5">
              <span className="rc-card-header-label">Carriageway width (m)</span>
              <input value={road} onChange={(e) => setRoad(e.target.value)} placeholder="7.0 assumed" type="number" min={3} step={0.5} className={inputCls} />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="rc-card-header-label">Climate</span>
              <select value={climate} onChange={(e) => setClimate(e.target.value)} className={`${inputCls} cursor-pointer`}>
                {CLIMATES.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <p className="text-xs text-slate-500">
            Creating a project segments the photo, measures the street and builds the gated plan. The illustrative image is
            generated later, on request.
          </p>
          {error && <p className="text-sm text-rose-800">{error}</p>}
          {busy && <div className="rc-progress" />}
        </div>

        <footer className="flex justify-end gap-2 px-5 py-4 border-t border-slate-200">
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" type="submit" disabled={busy}>
            {busy ? 'Analysing…' : 'Create project'}
          </Button>
        </footer>
      </form>
    </div>
  );
};
