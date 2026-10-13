import React from 'react';
import { Eyebrow, Panel, Tag } from './ui';

const STAGES = [
  {
    n: '01',
    title: 'Scene segmentation',
    body: 'SegFormer-B0 (ADE20K) labels road, footpath, walls, vegetation, sky and protected objects such as vehicles, people and poles.',
  },
  {
    n: '02',
    title: 'Metric measurement',
    body: 'Footpath widths are measured in metres on the same image rows as the carriageway, using its width as the scale. Every estimate carries a ±30% band.',
  },
  {
    n: '03',
    title: 'Buildable planning',
    body: 'Candidate interventions are gated by clear-walkway, junction, obstacle, climate and heritage rules, then ranked by utility or by cooling per lakh under a budget.',
  },
  {
    n: '04',
    title: 'Illustrative image',
    body: 'FLUX.1 Kontext edits the photo to show the top-priority items. The image is illustrative; the bill of quantities and section are the deliverable.',
  },
];

const RULES = [
  ['Tree pits', 'Only where the footpath keeps the minimum clear walkway after the pit and kerb buffer.'],
  ['Obstacles', 'No trees beside dense poles, kerbside vehicles or crowded frontage, measured in a kerb strip.'],
  ['Junctions', 'A setback keeps sight lines clear near a visible side street.'],
  ['Climate', 'Misting and evaporative cooling are rejected in humid climates.'],
  ['Cool roads', 'Reflective road coatings only in deep street canyons, to avoid bouncing sun onto pedestrians.'],
  ['Heritage', 'Historic paving is preserved rather than coated.'],
];

export const MethodNoteView: React.FC = () => (
  <div className="flex flex-col gap-10">
    <div className="flex flex-col gap-2 max-w-3xl">
      <Eyebrow>Method</Eyebrow>
      <h1 className="text-2xl sm:text-3xl font-semibold text-slate-900">How ResiliCity reaches a plan</h1>
      <p className="text-sm text-slate-500">
        The system separates seeing the street, measuring it, deciding what can be built, and drawing an illustration, so each
        step can be checked on its own.
      </p>
    </div>

    <ol className="rc-stagger grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-px bg-slate-200 border border-slate-200">
      {STAGES.map((s) => (
        <li key={s.n} className="bg-white p-5 flex flex-col gap-2 transition-colors hover:bg-[#f3f8f6]">
          <span className="text-sm text-[#0d7a5f]">{s.n}</span>
          <span className="text-base font-semibold text-slate-900">{s.title}</span>
          <p className="text-sm text-slate-600">{s.body}</p>
        </li>
      ))}
    </ol>

    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <Panel eyebrow="Buildability rules" title="Why an intervention is accepted or rejected">
        <dl className="divide-y divide-slate-100">
          {RULES.map(([k, v]) => (
            <div key={k} className="py-2.5 grid grid-cols-[120px_1fr] gap-4 text-sm">
              <dt className="text-slate-900">{k}</dt>
              <dd className="text-slate-600">{v}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 text-xs text-slate-500">
          Every threshold, dimension and unit cost lives in one configuration file and is labelled indicative. Replace it with
          the local Schedule of Rates and street design guidelines before use.
        </p>
      </Panel>

      <Panel eyebrow="Heat model" title="Satellite land-surface regression" actions={<Tag tone="amber">Weak fit</Tag>}>
        <p className="text-sm text-slate-600">
          A gradient-boosting model trained on 3,117 Landsat sample points around Chennai and Chengalpattu predicts land
          surface temperature from land cover. Spatial cross-validation holds out whole ~5 km blocks.
        </p>
        <div className="mt-4 grid grid-cols-2 gap-2">
          <div className="border border-slate-200 px-4 py-3">
            <Eyebrow>Spatial-CV R²</Eyebrow>
            <div className="text-xl font-semibold text-slate-900">0.26</div>
          </div>
          <div className="border border-slate-200 px-4 py-3">
            <Eyebrow>RMSE</Eyebrow>
            <div className="text-xl font-semibold text-slate-900">2.5 °C</div>
          </div>
        </div>
        <p className="mt-4 text-sm text-slate-600">
          Because the model explains only about a quarter of the variation, cooling figures use literature-based ranges per
          intervention, and the model's own change is shown with a ±RMSE band.
        </p>
      </Panel>
    </div>

    <Panel eyebrow="Caveats" title="Read the numbers with these in mind">
      <ul className="list-disc pl-5 text-sm text-slate-600 flex flex-col gap-1.5">
        <li>Land surface temperature is skin temperature of surfaces, not the air temperature people feel.</li>
        <li>Satellite data is a March to May median (2021 to 2025), so it reflects the pre-monsoon season.</li>
        <li>Widths come from a single photo and an assumed or entered carriageway width; survey before design.</li>
        <li>Cooling estimates are summed without interaction effects and are indicative, not a CFD simulation.</li>
      </ul>
    </Panel>
  </div>
);
