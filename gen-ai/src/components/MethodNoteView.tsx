import React, { useEffect, useState } from 'react';
import { fetchModelCards } from '../services/plannerApi';
import type { ModelCards } from '../types/planner';
import { SourceChip } from './Figure';
import { Eyebrow, Panel, Tag } from './ui';

const LOOP = [
  { n: '01', title: 'Find', body: 'The site heat model ranks 250 m cells by predicted surface temperature above the city median, weighted by population.' },
  { n: '02', title: 'Plan', body: 'A street photo is segmented and measured in metres; rules keep only interventions that physically fit and price them per 100 m.' },
  { n: '03', title: 'Decide', body: 'An exact knapsack chooses interventions across projects to maximise person-°C within the budget, with worst and best cases.' },
  { n: '04', title: 'Export', body: 'Work orders as PDF, CSV and JSON: quantities, cost bands, permits, conditions to verify, and rejected options with reasons.' },
  { n: '05', title: 'Verify', body: 'After construction, Landsat surface temperature at the project cell is compared with untreated control cells.' },
];

const SOURCES: { tag: string; what: string }[] = [
  { tag: 'model', what: 'Predicted surface temperature and its interval per cell; hotspot rank and priority.' },
  { tag: 'measured', what: 'Footpath widths from the photo (with a band), observed Landsat LST, GHSL population on the city grid.' },
  { tag: 'rule', what: 'Gating decisions (clear walkway, junctions, obstacles, climate, heritage) and quantities derived from measured widths.' },
  { tag: 'literature', what: 'Cooling per intervention (low, central, high), summed without interaction effects.' },
  { tag: 'assumed', what: 'Unit costs, carriageway width when not entered, population in demo mode, the single-photo heat index weights.' },
];

const RULES = [
  ['Tree pits', 'Only where the footpath keeps the minimum clear walkway after the pit and kerb buffer.'],
  ['Obstacles', 'No trees beside dense poles, kerbside vehicles or crowded frontage, measured in a kerb strip.'],
  ['Junctions', 'A setback keeps sight lines clear near a visible side street.'],
  ['Climate', 'Misting and evaporative cooling are rejected in humid climates.'],
  ['Cool roads', 'Reflective road coatings only in deep street canyons, to avoid bouncing sun onto pedestrians.'],
  ['Heritage', 'Historic paving is preserved rather than coated.'],
];

const fmt = (m: Record<string, number> | null | undefined, k: string, d = 3) => (m && typeof m[k] === 'number' ? m[k].toFixed(d) : '—');

export const MethodNoteView: React.FC = () => {
  const [cards, setCards] = useState<ModelCards | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModelCards().then(setCards).catch((e: Error) => setError(e.message));
  }, []);

  const site = cards?.site_model;
  const legacy = cards?.legacy_land_cover_model;
  const siteCard = site?.card as { intervals?: Record<string, { half_width_c: number; empirical_coverage_nested_spatial: number }>; n_samples?: number; features?: string[]; leakage_comparison_random_kfold?: Record<string, number> } | undefined;

  return (
    <div className="flex flex-col gap-10">
      <div className="flex flex-col gap-2 max-w-3xl">
        <Eyebrow>Method</Eyebrow>
        <h1 className="text-2xl sm:text-3xl font-semibold text-slate-900">Machine learning finds hotspots; rules and literature decide what to build</h1>
        <p className="text-sm text-slate-500">
          The site heat model answers where it is hot. It does not predict the cooling of an intervention: no data exists to
          validate that, and land-cover changes cannot be read back into the satellite indices it uses. Cooling comes from
          published ranges, and buildability from design rules.
        </p>
      </div>

      <ol className="rc-stagger grid grid-cols-1 md:grid-cols-5 gap-px bg-slate-200 border border-slate-200">
        {LOOP.map((s) => (
          <li key={s.n} className="bg-white p-5 flex flex-col gap-2 transition-colors hover:bg-[#f3f8f6]">
            <span className="text-sm text-[#0d7a5f]">{s.n}</span>
            <span className="text-base font-semibold text-slate-900">{s.title}</span>
            <p className="text-sm text-slate-600">{s.body}</p>
          </li>
        ))}
      </ol>

      {error && <div className="border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">Model cards unavailable: {error}</div>}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Panel eyebrow="Site heat model" title="Where is it hot" actions={<SourceChip source="model" />}>
          {!site ? (
            <div className="h-40 rc-shimmer" />
          ) : (
            <>
              <p className="text-sm text-slate-600">
                Gradient boosting on land cover, Sentinel-2 NDVI/NDBI/MNDWI and location
                {siteCard?.n_samples ? `, ${siteCard.n_samples.toLocaleString('en-IN')} Landsat points` : ''}. Evaluated with
                spatial cross-validation (5 km blocks held out).
              </p>
              <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-px bg-slate-200 border border-slate-200">
                {[
                  ['Spatial-CV R²', fmt(site.spatial_cv, 'R2')],
                  ['RMSE °C', fmt(site.spatial_cv, 'RMSE', 2)],
                  ['MAE °C', fmt(site.spatial_cv, 'MAE', 2)],
                  ['Within-block R²', fmt(site.spatial_cv, 'within_block_R2')],
                ].map(([k, v]) => (
                  <div key={k} className="bg-white px-3 py-2.5">
                    <span className="rc-card-header-label">{k}</span>
                    <div className="text-lg font-semibold text-slate-900 tabular-nums">{v}</div>
                  </div>
                ))}
              </div>
              {siteCard?.intervals && (
                <p className="mt-3 text-sm text-slate-600">
                  Intervals (split-conformal): 80% ±{siteCard.intervals['80']?.half_width_c.toFixed(2)} °C, measured coverage{' '}
                  {(siteCard.intervals['80']?.empirical_coverage_nested_spatial * 100).toFixed(1)}%; 90% ±
                  {siteCard.intervals['90']?.half_width_c.toFixed(2)} °C, coverage{' '}
                  {(siteCard.intervals['90']?.empirical_coverage_nested_spatial * 100).toFixed(1)}%.
                </p>
              )}
              {siteCard?.leakage_comparison_random_kfold && (
                <p className="mt-2 text-xs text-slate-500">
                  Random cross-validation gives R² {fmt(siteCard.leakage_comparison_random_kfold, 'R2')}; it leaks spatial
                  information and is shown only for comparison.
                </p>
              )}
            </>
          )}
        </Panel>

        <Panel eyebrow="Legacy land-cover model" title="Single-photo reference only" actions={<Tag tone="amber">Low confidence</Tag>}>
          {!legacy ? (
            <div className="h-40 rc-shimmer" />
          ) : (
            <>
              <p className="text-sm text-slate-600">
                Gradient boosting on land-cover fractions alone. It explains little of the variation and predicts small effects
                from land-cover changes, so it is not used for intervention cooling.
              </p>
              <div className="mt-4 grid grid-cols-3 gap-px bg-slate-200 border border-slate-200">
                {[
                  ['Spatial-CV R²', fmt(legacy.spatial_cv, 'R2')],
                  ['RMSE °C', fmt(legacy.spatial_cv, 'RMSE', 2)],
                  ['MAE °C', fmt(legacy.spatial_cv, 'MAE', 2)],
                ].map(([k, v]) => (
                  <div key={k} className="bg-white px-3 py-2.5">
                    <span className="rc-card-header-label">{k}</span>
                    <div className="text-lg font-semibold text-slate-900 tabular-nums">{v}</div>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-sm text-slate-600">
                Its <span className="font-mono text-xs">/heat</span> endpoint marks a result unreliable only when the change has
                the wrong sign or is smaller than 0.05 °C. The RMSE is reported as a band around the change, not used as a gate.
              </p>
            </>
          )}
        </Panel>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <Panel eyebrow="Where each number comes from" title="Source tags used in the app" bodyClassName="p-0">
          <ul className="divide-y divide-slate-100">
            {SOURCES.map((s) => (
              <li key={s.tag} className="px-5 py-3 grid grid-cols-[96px_1fr] gap-4 text-sm items-start">
                <span className="justify-self-start">
                  <SourceChip source={s.tag} />
                </span>
                <span className="text-slate-600">{s.what}</span>
              </li>
            ))}
          </ul>
        </Panel>

        <Panel eyebrow="Buildability rules" title="Why an intervention is accepted or rejected">
          <dl className="divide-y divide-slate-100">
            {RULES.map(([k, v]) => (
              <div key={k} className="py-2.5 grid grid-cols-[110px_1fr] gap-4 text-sm">
                <dt className="text-slate-900">{k}</dt>
                <dd className="text-slate-600">{v}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-4 text-xs text-slate-500">
            Every threshold, dimension and unit cost lives in one configuration file and is labelled indicative.
          </p>
        </Panel>
      </div>

      <Panel eyebrow="Caveats" title="Read the numbers with these in mind">
        <ul className="list-disc pl-5 text-sm text-slate-600 flex flex-col gap-1.5">
          <li>Land surface temperature is the temperature of surfaces, not the air temperature people feel.</li>
          <li>Satellite data are March to May medians (2021 to 2025): the pre-monsoon season.</li>
          <li>Without the exported city grid the app runs on sample points with a constant population (demo data).</li>
          <li>Widths come from one photo and an entered or assumed carriageway width; survey before design.</li>
          <li>Cooling estimates are indicative and summed without interaction effects; this is not a CFD simulation.</li>
        </ul>
      </Panel>
    </div>
  );
};
