import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ChevronDown, Columns2, Download, Maximize2, Play, SplitSquareHorizontal } from 'lucide-react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import type { DesignProfile } from '../types/resilicity';
import { Button, Panel, SectionHeading, Tag } from './ui';
import { framedAspect, sentenceCase, zoneLabel } from './format';

const PROFILES: { id: DesignProfile; label: string }[] = [
  { id: 'balanced', label: 'Balanced' },
  { id: 'pedestrian_first', label: 'Pedestrian first' },
  { id: 'maximum_cooling', label: 'Maximum cooling' },
  { id: 'green_infrastructure', label: 'Green infrastructure' },
  { id: 'low_cost', label: 'Low cost' },
];

const STAGES = ['ANALYZING SITE', 'PLANNING RESILIENCE STRATEGY', 'MAPPING INTERVENTIONS', 'GENERATING LOCALLY', 'VALIDATING RESULT'];
const STAGE_LABELS: Record<string, string> = {
  'GENERATING LOCALLY': 'Rendering illustrative image',
};

const RunControls: React.FC = () => {
  const { designProfile, setDesignProfile, qualityTier, setQualityTier, generateResilientDesign, isGenerating, isSegmenting } =
    useResiliCityStore();
  return (
    <div className="flex flex-wrap items-end gap-4 border border-slate-200 bg-white px-5 py-4">
      <div className="flex flex-col gap-1.5">
        <span className="rc-card-header-label">Design priority</span>
        <div className="flex flex-wrap border border-slate-300">
          {PROFILES.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => setDesignProfile(p.id)}
              className={`px-3 h-8 text-xs border-r border-slate-300 last:border-r-0 cursor-pointer ${
                designProfile === p.id ? 'bg-[#0d7a5f] text-white' : 'bg-white text-slate-600 hover:bg-[#e8f3ef] hover:text-[#0a6550]'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        <span className="rc-card-header-label">Image quality</span>
        <div className="flex border border-slate-300">
          {(['fast', 'final'] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setQualityTier(t)}
              className={`px-3 h-8 text-xs border-r border-slate-300 last:border-r-0 cursor-pointer ${
                qualityTier === t ? 'bg-[#0d7a5f] text-white' : 'bg-white text-slate-600 hover:bg-[#e8f3ef] hover:text-[#0a6550]'
              }`}
            >
              {t === 'fast' ? 'Preview' : 'Final'}
            </button>
          ))}
        </div>
      </div>
      <Button
        variant="primary"
        className="ml-auto"
        icon={<Play className="w-4 h-4" />}
        disabled={isGenerating || isSegmenting}
        onClick={() => void generateResilientDesign()}
      >
        {isGenerating ? 'Generating…' : 'Generate proposal'}
      </Button>
    </div>
  );
};

const ComparisonViewer: React.FC = () => {
  const { rawImageUrl, generatedImageUrl, sliderPosition, setSliderPosition, viewMode, setViewMode, isGenerating, generationStage, visualizationOutput } =
    useResiliCityStore();
  const frameRef = useRef<HTMLDivElement>(null);
  const shellRef = useRef<HTMLDivElement>(null);
  const [aspect, setAspect] = useState(3 / 2);
  const [dragging, setDragging] = useState(false);
  const [touched, setTouched] = useState(false);

  useEffect(() => {
    if (!rawImageUrl) return;
    const img = new Image();
    img.onload = () => img.naturalWidth && setAspect(img.naturalWidth / img.naturalHeight);
    img.src = rawImageUrl;
  }, [rawImageUrl]);

  const moveTo = useCallback(
    (clientX: number) => {
      const r = frameRef.current?.getBoundingClientRect();
      if (r) setSliderPosition(Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100)));
    },
    [setSliderPosition]
  );

  useEffect(() => {
    if (!dragging) return;
    const move = (e: PointerEvent) => moveTo(e.clientX);
    const up = () => setDragging(false);
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
  }, [dragging, moveTo]);

  const hasResult = Boolean(generatedImageUrl);
  const unavailable = visualizationOutput?.status === 'unavailable' && !isGenerating;
  const stageIndex = STAGES.indexOf(generationStage ?? '');

  const download = () => {
    if (!generatedImageUrl) return;
    const a = document.createElement('a');
    a.href = generatedImageUrl;
    a.download = 'resilicity-proposal.jpg';
    a.click();
  };

  const proposalLayer = (
    <>
      {hasResult && <img key={generatedImageUrl!} src={generatedImageUrl!} alt="Illustrative proposal" className="rc-wipe absolute inset-0 w-full h-full object-cover" />}
      {!hasResult && isGenerating && <div className="absolute inset-0 rc-shimmer" />}
      {!hasResult && !isGenerating && (
        <div className="absolute inset-0 bg-slate-50 flex items-center justify-center p-6 text-center text-sm text-slate-500">
          {unavailable ? 'Image not generated. The plan and specification below are still valid.' : 'Generate a proposal to see the illustrative image.'}
        </div>
      )}
    </>
  );

  return (
    <Panel
      eyebrow="Before and after"
      title="Existing street and illustrative proposal"
      actions={
        <>
          <div className="flex border border-slate-300">
            <button
              type="button"
              title="Slider"
              onClick={() => setViewMode('slider')}
              className={`h-8 px-2.5 cursor-pointer ${viewMode !== 'side-by-side' ? 'bg-[#0d7a5f] text-white' : 'text-slate-600 hover:bg-slate-50'}`}
            >
              <SplitSquareHorizontal className="w-4 h-4" />
            </button>
            <button
              type="button"
              title="Side by side"
              onClick={() => setViewMode('side-by-side')}
              className={`h-8 px-2.5 border-l border-slate-300 cursor-pointer ${viewMode === 'side-by-side' ? 'bg-[#0d7a5f] text-white' : 'text-slate-600 hover:bg-slate-50'}`}
            >
              <Columns2 className="w-4 h-4" />
            </button>
          </div>
          <Button variant="ghost" className="px-2" title="Download proposal image" disabled={!hasResult} onClick={download}>
            <Download className="w-4 h-4" />
          </Button>
          <Button variant="ghost" className="px-2" title="Fullscreen" onClick={() => void shellRef.current?.requestFullscreen?.()}>
            <Maximize2 className="w-4 h-4" />
          </Button>
        </>
      }
      bodyClassName="p-0"
    >
      <div ref={shellRef} className="bg-slate-50 flex flex-col justify-center">
        {viewMode === 'side-by-side' ? (
          <div className="grid grid-cols-2 gap-px bg-slate-200">
            {[false, true].map((isProposal) => (
              <div key={String(isProposal)} className="relative bg-slate-100 overflow-hidden" style={framedAspect(aspect, 440)}>
                {isProposal ? proposalLayer : rawImageUrl && <img src={rawImageUrl} alt="Existing street" className="absolute inset-0 w-full h-full object-cover" />}
                <span className="absolute top-3 left-3 bg-white/95 px-2 py-1 text-[11px] text-slate-700 border border-slate-200">
                  {isProposal ? 'Proposal (illustrative)' : 'Existing'}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div
            ref={frameRef}
            className="relative bg-slate-100 overflow-hidden select-none touch-none cursor-ew-resize"
            style={framedAspect(aspect, 560)}
            onPointerDown={(e) => {
              setDragging(true);
              setTouched(true);
              moveTo(e.clientX);
            }}
          >
            {rawImageUrl && <img src={rawImageUrl} alt="Existing street" className="absolute inset-0 w-full h-full object-cover pointer-events-none" />}
            <div className="absolute inset-0 pointer-events-none" style={{ clipPath: `inset(0 0 0 ${sliderPosition}%)` }}>
              {proposalLayer}
            </div>
            <span className="absolute top-3 left-3 bg-white/95 px-2 py-1 text-[11px] text-slate-700 border border-slate-200">Existing</span>
            <span className="absolute top-3 right-3 bg-[#0d7a5f] px-2 py-1 text-[11px] text-white">Proposal (illustrative)</span>

            <div className="absolute inset-y-0 w-px bg-white shadow-[0_0_0_1px_rgba(15,23,42,0.15)]" style={{ left: `${sliderPosition}%` }}>
              <button
                type="button"
                role="slider"
                aria-label="Comparison position"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(sliderPosition)}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowLeft') setSliderPosition(Math.max(0, sliderPosition - 5));
                  if (e.key === 'ArrowRight') setSliderPosition(Math.min(100, sliderPosition + 5));
                  setTouched(true);
                }}
                className={`absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white border border-slate-300 flex items-center justify-center gap-0.5 cursor-ew-resize ${
                  touched ? '' : 'rc-pulse'
                }`}
              >
                <span className="w-px h-3 bg-slate-500" />
                <span className="w-px h-3 bg-slate-500" />
              </button>
            </div>
          </div>
        )}

        {/* Generation progress: stage list with a moving line */}
        {isGenerating && (
          <div className="rc-fade-in border-t border-slate-200 px-5 py-4 flex flex-col gap-3">
            <div className="rc-progress" />
            <ol className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
              {STAGES.map((s, i) => (
                <li
                  key={s}
                  className={`flex items-center gap-2 transition-colors ${
                    i < stageIndex ? 'text-[#0d7a5f]' : i === stageIndex ? 'text-slate-900' : 'text-slate-400'
                  }`}
                >
                  <span className={`w-1.5 h-1.5 ${i <= stageIndex ? 'bg-[#0d7a5f]' : 'bg-slate-300'}`} />
                  {STAGE_LABELS[s] ?? sentenceCase(s)}
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>

      {unavailable && visualizationOutput?.error_message && (
        <div className="border-t border-amber-200 bg-amber-50 px-5 py-3 text-xs text-amber-900">{visualizationOutput.error_message}</div>
      )}

      <div className="border-t border-slate-200 px-5 py-3 text-xs text-slate-500 flex flex-wrap gap-x-6 gap-y-1">
        <span>Illustrative only, not to scale. The specification in section 03 is the deliverable.</span>
        {visualizationOutput?.illustrated_interventions?.length ? (
          <span>
            Image shows: <span className="text-slate-700">{visualizationOutput.illustrated_interventions.map((t) => t.replace(/_/g, ' ')).join(', ')}</span>
          </span>
        ) : null}
        {visualizationOutput?.cached && <span>Served from cache, no generation quota used.</span>}
      </div>
    </Panel>
  );
};

interface Row {
  key: string;
  title: string;
  zone?: string | null;
  cooling: number;
  feasibility?: number;
  reason?: string;
  placement?: string;
  evidence?: string[];
  conditions?: string[];
}

const InterventionList: React.FC = () => {
  const { spatialDesignPlan, interventions } = useResiliCityStore();
  const [open, setOpen] = useState<string | null>(null);

  const rows: Row[] = spatialDesignPlan?.interventions?.length
    ? spatialDesignPlan.interventions.map((s, i) => ({
        key: `${s.type}-${s.target_zone ?? i}`,
        title: s.title || sentenceCase(s.type.replace(/_/g, ' ')),
        zone: s.target_zone,
        cooling: s.cooling_impact_c,
        feasibility: s.feasibility,
        reason: s.reason,
        placement: s.placement,
        evidence: s.evidence,
        conditions: s.spatial_constraints,
      }))
    : interventions.map((iv) => ({ key: iv.id, title: iv.title, cooling: iv.coolingImpact, reason: iv.description }));

  return (
    <Panel eyebrow="Selected interventions" title={`${rows.length} items, ranked by priority`} bodyClassName="p-0">
      <ol className="rc-stagger divide-y divide-slate-200">
        {rows.map((r, i) => {
          const isOpen = open === r.key;
          return (
            <li key={r.key}>
              <button
                type="button"
                onClick={() => setOpen(isOpen ? null : r.key)}
                className="rc-row w-full flex items-center gap-4 px-5 py-3.5 text-left cursor-pointer"
                aria-expanded={isOpen}
              >
                <span className="text-xs text-[#0d7a5f] tabular-nums w-5">{String(i + 1).padStart(2, '0')}</span>
                <span className="flex-1 min-w-0">
                  <span className="block text-sm text-slate-900 truncate">{r.title}</span>
                  <span className="block text-xs text-slate-500">{zoneLabel(r.zone)}</span>
                </span>
                {r.feasibility !== undefined && (
                  <span className="hidden sm:block text-xs text-slate-500 w-24 text-right">{Math.round(r.feasibility * 100)}% feasible</span>
                )}
                <span className="text-sm text-[#0d7a5f] tabular-nums w-16 text-right">-{r.cooling.toFixed(1)} °C</span>
                <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} />
              </button>
              <div className="rc-collapse" data-open={isOpen}>
                <div>
                  <div className="px-5 pb-4 pl-14 flex flex-col gap-2 text-xs text-slate-600">
                    {r.reason && <p>{r.reason}</p>}
                    {r.placement && (
                      <p>
                        <span className="text-slate-900">Placement: </span>
                        {r.placement}
                      </p>
                    )}
                    {r.evidence?.length ? (
                      <ul className="list-disc pl-4 flex flex-col gap-0.5">
                        {r.evidence.map((e) => (
                          <li key={e}>{e}</li>
                        ))}
                      </ul>
                    ) : null}
                    {r.conditions?.map((c) => (
                      <Tag key={c} tone="amber" className="self-start">
                        {c}
                      </Tag>
                    ))}
                  </div>
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </Panel>
  );
};

export const ProposalSection: React.FC = () => (
  <section className="flex flex-col gap-5">
    <SectionHeading
      id="proposal"
      number="02"
      title="Proposal"
      description="Interventions that pass the dimension and site rules, with an illustrative image of the top priorities."
    />
    <RunControls />
    <div className="grid grid-cols-1 xl:grid-cols-[1.6fr_1fr] gap-5 items-start">
      <ComparisonViewer />
      <InterventionList />
    </div>
  </section>
);
