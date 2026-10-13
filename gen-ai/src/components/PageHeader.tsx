import React from 'react';
import { Download, Play, RotateCcw } from 'lucide-react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import { Button, Eyebrow } from './ui';

const STEPS = [
  { id: 'site', number: '01', label: 'Site' },
  { id: 'proposal', number: '02', label: 'Proposal' },
  { id: 'specification', number: '03', label: 'Specification' },
  { id: 'impact', number: '04', label: 'Impact' },
];

export const PageHeader: React.FC<{ onExport: () => void }> = ({ onExport }) => {
  const { resetAll, generateResilientDesign, isGenerating, isSegmenting, rawImageUrl, spatialDesignPlan, thermalImpact } =
    useResiliCityStore();

  const done = [
    Boolean(rawImageUrl),
    Boolean(spatialDesignPlan),
    Boolean(spatialDesignPlan?.implementation_plan),
    Boolean(thermalImpact),
  ];
  const completed = done.filter(Boolean).length;
  const busy = isGenerating || isSegmenting;

  const jump = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="flex flex-col gap-1">
          <Eyebrow>Street heat inspection</Eyebrow>
          <h1 className="text-2xl sm:text-3xl font-semibold text-slate-900">From street photo to buildable cooling plan</h1>
          <p className="text-sm text-slate-500 max-w-2xl">
            Measure the street, choose interventions that physically fit, and price them per 100 m. Every figure is an
            estimate with its assumptions shown.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button icon={<RotateCcw className="w-4 h-4" />} onClick={resetAll}>
            Reset
          </Button>
          <Button icon={<Download className="w-4 h-4" />} onClick={onExport}>
            Export report
          </Button>
          <Button
            variant="primary"
            icon={<Play className="w-4 h-4" />}
            disabled={busy}
            onClick={() => void generateResilientDesign('fast')}
          >
            {busy ? 'Working…' : 'Run analysis'}
          </Button>
        </div>
      </div>

      {/* Step rail: fills as each stage produces output; click to jump */}
      <div className="relative">
        <div className="absolute left-0 right-0 top-[15px] h-px bg-slate-200" />
        <div
          className="absolute left-0 top-[15px] h-px bg-[#0d7a5f] transition-all duration-700 ease-out"
          style={{ width: `${(Math.max(0, completed - 1) / (STEPS.length - 1)) * 100}%` }}
        />
        <ol className="relative grid grid-cols-4">
          {STEPS.map((s, i) => (
            <li key={s.id} className={i === 0 ? 'justify-self-start' : i === STEPS.length - 1 ? 'justify-self-end' : 'justify-self-center'}>
              <button type="button" onClick={() => jump(s.id)} className="group flex flex-col items-center gap-2 cursor-pointer">
                <span
                  className={`w-[31px] h-[31px] flex items-center justify-center text-xs border transition-colors duration-300 ${
                    done[i]
                      ? 'bg-[#0d7a5f] border-[#0d7a5f] text-white'
                      : 'bg-white border-slate-300 text-slate-500 group-hover:border-[#0d7a5f] group-hover:text-[#0d7a5f]'
                  }`}
                >
                  {s.number}
                </span>
                <span className={`hidden sm:inline text-xs ${done[i] ? 'text-slate-900' : 'text-slate-500'}`}>{s.label}</span>
              </button>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
};
