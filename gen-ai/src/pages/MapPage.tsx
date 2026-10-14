import React, { useEffect, useMemo, useState } from 'react';
import { CircleMarker, MapContainer, TileLayer, Tooltip, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { MapPin, Plus } from 'lucide-react';
import { fetchHotspotDetail, fetchHotspots } from '../services/plannerApi';
import type { HotspotCell, HotspotDetail, HotspotsResponse } from '../types/planner';
import { Figure, SourceChip } from '../components/Figure';
import { Button, Eyebrow, Panel, Tag } from '../components/ui';
import { NewProjectDialog } from './NewProjectDialog';

const CENTER: [number, number] = [12.9, 80.12];

/** Sequential colour for LST: cool green to hot red, matching the app's heat scale. */
function lstColor(t: number, lo: number, hi: number): string {
  const x = Math.max(0, Math.min(1, (t - lo) / (hi - lo || 1)));
  const stops = [
    [16, 185, 129],
    [250, 204, 21],
    [244, 63, 94],
  ];
  const seg = x < 0.5 ? 0 : 1;
  const f = x < 0.5 ? x / 0.5 : (x - 0.5) / 0.5;
  const c = stops[seg].map((v, i) => Math.round(v + (stops[seg + 1][i] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

const FlyTo: React.FC<{ cell: HotspotCell | null }> = ({ cell }) => {
  const map = useMap();
  useEffect(() => {
    if (cell) map.flyTo([cell.lat, cell.lon], Math.max(map.getZoom(), 13), { duration: 0.6 });
  }, [cell, map]);
  return null;
};

export const MapPage: React.FC = () => {
  const [data, setData] = useState<HotspotsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<HotspotCell | null>(null);
  const [fetchedDetail, setDetail] = useState<HotspotDetail | null>(null);
  const [startFor, setStartFor] = useState<HotspotCell | null>(null);

  useEffect(() => {
    fetchHotspots().then(setData).catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!selected) return;
    fetchHotspotDetail(selected.cell_id).then(setDetail).catch(() => setDetail(null));
  }, [selected]);
  // Only show detail that belongs to the current selection (a stale response is ignored).
  const detail = fetchedDetail && selected && fetchedDetail.cell.cell_id === selected.cell_id ? fetchedDetail : null;

  const scale = useMemo(() => {
    if (!data?.cells.length) return { lo: 30, hi: 50, maxPop: 1 };
    const t = data.cells.map((c) => c.lst_pred_c).sort((a, b) => a - b);
    return {
      lo: t[Math.floor(t.length * 0.05)],
      hi: t[Math.floor(t.length * 0.95)],
      maxPop: Math.max(...data.cells.map((c) => c.population), 1),
    };
  }, [data]);

  const demo = data?.data_mode === 'demo';
  const top = data?.cells.slice(0, 15) ?? [];
  const radius = (c: HotspotCell) => (demo ? 4 : 3 + 7 * Math.sqrt(c.population / scale.maxPop));

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="flex flex-col gap-1">
          <Eyebrow>Find</Eyebrow>
          <h1 className="text-2xl sm:text-3xl font-semibold text-slate-900">Where is it hottest, and who is exposed?</h1>
          <p className="text-sm text-slate-500 max-w-2xl">
            Cells are ranked by predicted pre-monsoon surface temperature above the city median, weighted by population.
            Pick a cell to start a street project there.
          </p>
        </div>
        {data && (
          <div className="flex flex-wrap items-center gap-2">
            {demo ? <Tag tone="amber">Demo data: sample points, constant population</Tag> : <Tag tone="green">250 m city grid</Tag>}
            <Tag>{data.n_cells.toLocaleString('en-IN')} cells</Tag>
          </div>
        )}
      </div>

      {error && <div className="border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">{error}</div>}

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_360px] gap-5 items-start">
        <Panel
          eyebrow="Heat map"
          title={data ? `Predicted surface temperature, city median ${data.city_median_lst_c.toFixed(1)} °C` : 'Loading cells…'}
          actions={data ? <SourceChip source="model" /> : undefined}
          bodyClassName="p-0"
        >
          <div className="relative h-[560px]">
            {!data && !error && <div className="absolute inset-0 rc-shimmer" />}
            {data && (
              <MapContainer center={CENTER} zoom={11} preferCanvas className="h-full w-full" scrollWheelZoom>
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                {data.cells.map((c) => (
                  <CircleMarker
                    key={c.cell_id}
                    center={[c.lat, c.lon]}
                    radius={selected?.cell_id === c.cell_id ? radius(c) + 4 : radius(c)}
                    pathOptions={{
                      color: selected?.cell_id === c.cell_id ? '#0f172a' : lstColor(c.lst_pred_c, scale.lo, scale.hi),
                      weight: selected?.cell_id === c.cell_id ? 2 : 0.5,
                      fillColor: lstColor(c.lst_pred_c, scale.lo, scale.hi),
                      fillOpacity: 0.75,
                    }}
                    eventHandlers={{ click: () => setSelected(c) }}
                  >
                    <Tooltip>
                      Rank {c.rank}: {c.lst_pred_c.toFixed(1)} °C ({c.lst_interval_90_c[0].toFixed(1)} to {c.lst_interval_90_c[1].toFixed(1)})
                    </Tooltip>
                  </CircleMarker>
                ))}
                <FlyTo cell={selected} />
              </MapContainer>
            )}
            {data && (
              <div className="absolute left-3 bottom-3 z-[500] bg-white/95 border border-slate-200 px-3 py-2 text-[11px] text-slate-600 flex flex-col gap-1.5">
                <span>Predicted LST (°C)</span>
                <span className="block h-2 w-40" style={{ background: 'linear-gradient(90deg, rgb(16,185,129), rgb(250,204,21), rgb(244,63,94))' }} />
                <span className="flex justify-between w-40 tabular-nums">
                  <span>{scale.lo.toFixed(0)}</span>
                  <span>{scale.hi.toFixed(0)}</span>
                </span>
                <span>{demo ? 'Dot size fixed (no population data)' : 'Dot size: population'}</span>
              </div>
            )}
          </div>
        </Panel>

        <div className="flex flex-col gap-5">
          {selected ? (
            <Panel
              eyebrow={`Cell ${selected.cell_id}`}
              title={`Rank ${selected.rank} of ${data?.n_cells.toLocaleString('en-IN')}`}
              actions={
                <Button variant="primary" icon={<Plus className="w-4 h-4" />} onClick={() => setStartFor(selected)}>
                  Start project
                </Button>
              }
            >
              <div className="rc-stagger grid grid-cols-2 gap-4">
                <Figure label="Predicted LST" value={selected.lst_pred_c} unit=" °C" band={selected.lst_interval_90_c} source="model" />
                <Figure label="Above median" value={selected.heat_excess_c} unit=" °C" source="model" />
                <Figure label="Observed LST" value={selected.lst_observed_c} unit=" °C" source="measured" />
                <Figure label="Population" value={selected.population} decimals={0} source={demo ? 'assumed' : 'measured'} />
              </div>
              {detail?.cell.land_cover && (
                <div className="mt-4 flex flex-col gap-1.5">
                  <span className="rc-card-header-label">Land cover</span>
                  {Object.entries(detail.cell.land_cover)
                    .filter(([, v]) => v > 0.01)
                    .sort((a, b) => b[1] - a[1])
                    .map(([k, v]) => (
                      <div key={k} className="flex items-center gap-2 text-xs text-slate-600">
                        <span className="w-14">{k.replace('f_', '')}</span>
                        <span className="flex-1 h-1.5 bg-slate-100">
                          <span className="rc-bar block h-full bg-[#0d7a5f]" style={{ width: `${v * 100}%` }} />
                        </span>
                        <span className="w-10 text-right tabular-nums">{Math.round(v * 100)}%</span>
                      </div>
                    ))}
                </div>
              )}
              {selected.outside_training_range && (
                <p className="mt-3 text-xs text-amber-800">Inputs outside the training range; treat the prediction with extra caution.</p>
              )}
              {detail && (
                <div className="mt-4">
                  <span className="rc-card-header-label">Nearest cells</span>
                  <ul className="mt-1 divide-y divide-slate-100 text-xs">
                    {detail.nearest.map((n) => (
                      <li key={n.cell_id}>
                        <button type="button" onClick={() => setSelected(n)} className="rc-row w-full flex justify-between py-1.5 px-1 cursor-pointer">
                          <span className="text-slate-600">{Math.round(n.distance_m ?? 0)} m · rank {n.rank}</span>
                          <span className="tabular-nums text-slate-900">{n.lst_pred_c.toFixed(1)} °C</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </Panel>
          ) : (
            <Panel eyebrow="Cell" title="Select a cell on the map">
              <p className="text-sm text-slate-500">Click a dot or a row below to see its prediction, interval and land cover.</p>
            </Panel>
          )}

          <Panel eyebrow="Top priorities" title="Highest heat × exposure" bodyClassName="p-0">
            {!data ? (
              <div className="h-48 rc-shimmer" />
            ) : (
              <ol className="rc-stagger divide-y divide-slate-100">
                {top.map((c) => (
                  <li key={c.cell_id}>
                    <button
                      type="button"
                      onClick={() => setSelected(c)}
                      className={`rc-row w-full flex items-center gap-3 px-5 py-2.5 text-left cursor-pointer ${selected?.cell_id === c.cell_id ? 'bg-[#e8f3ef]' : ''}`}
                    >
                      <span className="text-xs text-[#0d7a5f] w-6 tabular-nums">{c.rank}</span>
                      <MapPin className="w-3.5 h-3.5 text-slate-400" />
                      <span className="flex-1 text-xs text-slate-500 tabular-nums">
                        {c.lat.toFixed(3)}, {c.lon.toFixed(3)}
                      </span>
                      <span className="text-sm text-slate-900 tabular-nums">{c.lst_pred_c.toFixed(1)} °C</span>
                      <span className="text-xs text-rose-700 tabular-nums w-12 text-right">+{c.heat_excess_c.toFixed(1)}</span>
                    </button>
                  </li>
                ))}
              </ol>
            )}
          </Panel>

          {data && (
            <ul className="text-xs text-slate-500 flex flex-col gap-1">
              <li>
                Site model spatial-CV R² {data.model.spatial_cv.R2?.toFixed(2)}, RMSE {data.model.spatial_cv.RMSE?.toFixed(2)} °C;
                90% interval ±{data.model.interval_half_width_c['90']?.toFixed(1)} °C. Predictions shown: {data.prediction}.
              </li>
              {data.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {startFor && <NewProjectDialog cell={startFor} onClose={() => setStartFor(null)} />}
    </div>
  );
};
