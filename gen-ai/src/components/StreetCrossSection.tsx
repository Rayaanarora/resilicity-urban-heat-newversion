import React from 'react';
import { Ruler } from 'lucide-react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import type { ImplementationLineItem } from '../types/resilicity';

// Drawing geometry (viewBox units). All real dimensions come from the backend plan.
const VB_W = 720;
const VB_H = 300;
const BUILDING_W = 46;
const GROUND_Y = 214;
const KERB_H = 7;
const MAX_V_PX_PER_M = 30;

const fmt = (m: number) => `${m.toFixed(1)} m`;

const num = (v: unknown, fallback = 0): number => (typeof v === 'number' ? v : Array.isArray(v) ? Number(v[0]) : fallback);

type Side = 'left_sidewalk' | 'right_sidewalk';

export const StreetCrossSection: React.FC = () => {
  const plan = useResiliCityStore((s) => s.spatialDesignPlan);
  const metrics = plan?.street_metrics;
  const impl = plan?.implementation_plan;

  if (!metrics) {
    return (
      <div className="rc-card p-6 text-sm text-slate-500">
        <span className="rc-card-header-label">STREET CROSS-SECTION</span>
        <p className="mt-2">Run an analysis to see the dimensioned cross-section.</p>
      </div>
    );
  }

  const clear = impl?.clearances ?? {};
  const minClear = clear.min_clear_walkway_m ?? 0;
  const kerbBuffer = clear.kerb_buffer_m ?? 0;
  const headroom = clear.min_vertical_clearance_m ?? 0;
  const parkingW = clear.parking_lane_width_m ?? 0;

  const items = impl?.line_items ?? [];
  const on = (type: string, side: Side): ImplementationLineItem | undefined =>
    items.find((i) => i.type === type && i.target_zone === side);

  const leftM = metrics.left_sidewalk_m ?? 0;
  const rightM = metrics.right_sidewalk_m ?? 0;
  const roadM = metrics.road_width_m;
  const totalM = leftM + roadM + rightM;

  const x0 = BUILDING_W + 8;
  const x1 = VB_W - BUILDING_W - 8;
  const s = (x1 - x0) / totalM; // horizontal px per metre
  const v = Math.min(s, MAX_V_PX_PER_M); // vertical px per metre (capped so tall items fit)
  const kerbL = x0 + leftM * s;
  const kerbR = kerbL + roadM * s;
  const swTop = GROUND_Y - KERB_H;

  const sideGeom = (side: Side) => {
    const isLeft = side === 'left_sidewalk';
    return {
      isLeft,
      facadeX: isLeft ? x0 : x1,
      kerbX: isLeft ? kerbL : kerbR,
      dir: isLeft ? 1 : -1, // direction from facade toward the road
      widthM: isLeft ? metrics.left_sidewalk_m : metrics.right_sidewalk_m,
      range: isLeft ? metrics.left_range_m : metrics.right_range_m,
    };
  };

  const renderSide = (side: Side) => {
    const g = sideGeom(side);
    const parts: React.ReactNode[] = [];
    const tree = on('tree_canopy', side);
    const awning = on('facade_awning', side);
    const canopy = on('shade_structure', side) ?? on('solar_canopy', side);
    const kerbTree = on('parking_lane_tree', side);
    const swale = on('kerb_bioswale', side);

    // Clear walkway: facade to pit (or to kerb buffer when no pit)
    let clearM = (g.widthM ?? 0) - kerbBuffer;
    if (tree && g.widthM) {
      const pitM = num(tree.dimensions?.pit_m);
      const heightM = num(tree.dimensions?.planting_height_m);
      const canopyM = num(tree.dimensions?.canopy_diameter_m);
      const pitOuter = g.kerbX - g.dir * kerbBuffer * s;
      const pitInner = pitOuter - g.dir * pitM * s;
      const pitX = Math.min(pitOuter, pitInner);
      const cx = (pitOuter + pitInner) / 2;
      clearM = g.widthM - kerbBuffer - pitM;
      parts.push(
        <g key={`${side}-tree`}>
          <rect x={pitX} y={swTop - 2} width={pitM * s} height={4} fill="#92400e" />
          <line x1={cx} y1={swTop} x2={cx} y2={swTop - heightM * v * 0.55} stroke="#78350f" strokeWidth={3} />
          <ellipse cx={cx} cy={swTop - heightM * v * 0.72} rx={(canopyM / 2) * s} ry={(canopyM / 2) * v}
                   fill="#16a34a" fillOpacity={0.75} />
          <text x={cx} y={swTop + 16} textAnchor="middle" className="fill-amber-800" fontSize={10}>
            pit {fmt(pitM)}
          </text>
        </g>
      );
    }

    if (awning) {
      const depthM = num(awning.dimensions?.depth_m);
      const y = swTop - headroom * v;
      const tip = g.facadeX + g.dir * depthM * s;
      parts.push(
        <g key={`${side}-awning`}>
          <line x1={g.facadeX} y1={y} x2={tip} y2={y + 6} stroke="#0f766e" strokeWidth={4} />
          <line x1={g.facadeX + g.dir * 4} y1={y + 3} x2={g.facadeX + g.dir * 4} y2={swTop} stroke="#0f766e"
                strokeDasharray="2 3" strokeWidth={1} />
          <text x={(g.facadeX + tip) / 2} y={y - 6} textAnchor="middle" className="fill-teal-800" fontSize={10}>
            awning {fmt(depthM)} @ {fmt(headroom)}
          </text>
        </g>
      );
    }

    if (canopy && g.widthM) {
      const depthM = Math.min(num(canopy.dimensions?.depth_m), g.widthM - kerbBuffer);
      const postX = g.kerbX - g.dir * kerbBuffer * s;
      const y = swTop - num(canopy.dimensions?.clearance_m, headroom) * v;
      const end = postX - g.dir * depthM * s;
      parts.push(
        <g key={`${side}-canopy`}>
          <line x1={postX} y1={swTop} x2={postX} y2={y} stroke="#475569" strokeWidth={2} />
          <line x1={postX} y1={y} x2={end} y2={y - 4} stroke={canopy.type === 'solar_canopy' ? '#1e3a8a' : '#e2e8f0'}
                strokeWidth={5} />
          <text x={(postX + end) / 2} y={y - 10} textAnchor="middle" className="fill-slate-600" fontSize={10}>
            {canopy.type === 'solar_canopy' ? 'PV pergola' : 'shade canopy'} {fmt(depthM)}
          </text>
        </g>
      );
    }

    if (kerbTree || swale) {
      const laneEnd = g.kerbX + g.dir * parkingW * s;
      const lx = Math.min(g.kerbX, laneEnd);
      parts.push(
        <g key={`${side}-lane`}>
          <rect x={lx} y={GROUND_Y - 1} width={parkingW * s} height={6} fill={swale ? '#65a30d' : '#a8a29e'} />
          {kerbTree && (() => {
            const h = num(kerbTree.dimensions?.planting_height_m);
            const c = num(kerbTree.dimensions?.canopy_diameter_m);
            const cx = (g.kerbX + laneEnd) / 2;
            return (
              <>
                <line x1={cx} y1={GROUND_Y} x2={cx} y2={GROUND_Y - h * v * 0.55} stroke="#78350f" strokeWidth={3} />
                <ellipse cx={cx} cy={GROUND_Y - h * v * 0.72} rx={(c / 2) * s} ry={(c / 2) * v}
                         fill="#16a34a" fillOpacity={0.75} />
              </>
            );
          })()}
          <text x={(g.kerbX + laneEnd) / 2} y={GROUND_Y + 16} textAnchor="middle" className="fill-stone-600" fontSize={10}>
            build-out {fmt(parkingW)}
          </text>
        </g>
      );
    }

    // Clear walkway dimension (only when a sidewalk was measured)
    if (g.widthM) {
      const ok = clearM >= minClear;
      const end = g.facadeX + g.dir * clearM * s;
      const y = swTop - 12;
      parts.push(
        <g key={`${side}-clear`}>
          <line x1={g.facadeX} y1={y} x2={end} y2={y} stroke={ok ? '#059669' : '#dc2626'} strokeWidth={1.5}
                markerStart="url(#tick)" markerEnd="url(#tick)" />
          <text x={(g.facadeX + end) / 2} y={y - 4} textAnchor="middle" fontSize={10}
                className={ok ? 'fill-emerald-700' : 'fill-red-600'}>
            clear {fmt(Math.max(0, clearM))}
          </text>
        </g>
      );
    } else {
      parts.push(
        <text key={`${side}-none`} x={g.facadeX + g.dir * 6} y={swTop - 24} textAnchor={g.isLeft ? 'start' : 'end'}
              fontSize={10} className="fill-slate-400">
          no footpath measured
        </text>
      );
    }
    return parts;
  };

  const dim = (a: number, b: number, label: string, y: number) => (
    <g>
      <line x1={a} y1={y} x2={b} y2={y} stroke="#334155" strokeWidth={1} markerStart="url(#tick)" markerEnd="url(#tick)" />
      <text x={(a + b) / 2} y={y + 14} textAnchor="middle" fontSize={11} className="fill-slate-700">
        {label}
      </text>
    </g>
  );

  const rangeLabel = (m: number | null | undefined, r: [number, number] | null | undefined) =>
    m ? `${fmt(m)}${r ? ` (${r[0].toFixed(1)}-${r[1].toFixed(1)})` : ''}` : 'n/a';

  return (
    <div className="rc-card p-6 flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <span className="rc-card-header-label">STREET CROSS-SECTION</span>
          <h3 className="text-xl font-semibold text-slate-900 font-sans">Dimensioned section (proposed)</h3>
        </div>
        <span className="flex items-center gap-1.5 text-[11px] font-mono text-slate-600 bg-slate-50 border border-slate-200 px-2.5 py-1 rounded-lg">
          <Ruler className="w-3.5 h-3.5" />
          ±{metrics.uncertainty_pct}% · scale {metrics.scale_source === 'user' ? 'from your road width' : 'assumed'}
        </span>
      </div>

      <div className="overflow-x-auto -mx-2 px-2">
        <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full min-w-[560px] text-slate-700" role="img"
             aria-label="Dimensioned street cross-section">
          <defs>
            <marker id="tick" viewBox="0 0 2 10" refX="1" refY="5" markerWidth="2" markerHeight="10" orient="auto">
              <line x1="1" y1="0" x2="1" y2="10" stroke="currentColor" strokeWidth="2" />
            </marker>
          </defs>
          {/* Buildings */}
          <rect x={8} y={40} width={BUILDING_W} height={swTop - 40} fill="#e2e8f0" stroke="#94a3b8" />
          <rect x={VB_W - BUILDING_W - 8} y={40} width={BUILDING_W} height={swTop - 40} fill="#e2e8f0" stroke="#94a3b8" />
          {/* Footpaths and carriageway */}
          <rect x={x0} y={swTop} width={kerbL - x0} height={GROUND_Y - swTop + 10} fill="#cbd5e1" />
          <rect x={kerbR} y={swTop} width={x1 - kerbR} height={GROUND_Y - swTop + 10} fill="#cbd5e1" />
          <rect x={kerbL} y={GROUND_Y} width={kerbR - kerbL} height={10} fill="#475569" />
          {renderSide('left_sidewalk')}
          {renderSide('right_sidewalk')}
          {/* Overall dimensions */}
          {leftM > 0 && dim(x0, kerbL, `footpath ${rangeLabel(metrics.left_sidewalk_m, metrics.left_range_m)}`, 246)}
          {dim(kerbL, kerbR, `carriageway ${fmt(roadM)}${metrics.scale_source === 'user' ? '' : ' (assumed)'}`, 246)}
          {rightM > 0 && dim(kerbR, x1, `footpath ${rangeLabel(metrics.right_sidewalk_m, metrics.right_range_m)}`, 246)}
          <text x={VB_W / 2} y={VB_H - 8} textAnchor="middle" fontSize={10} className="fill-slate-400">
            Minimum clear walkway {fmt(minClear)} · not to scale vertically
          </text>
        </svg>
      </div>

      {metrics.notes.length > 0 && (
        <ul className="text-[11px] text-slate-500 font-sans list-disc pl-5 flex flex-col gap-0.5">
          {metrics.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </div>
  );
};
