import React, { useEffect, useRef, useState } from 'react';
import { ImageUp } from 'lucide-react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import { SAMPLE_SCENES } from '../services/sampleData';
import { exposureLabel } from '../services/heatCalculator';
import { AnimatedNumber } from './AnimatedNumber';
import { Button, Panel, SectionHeading, Stat, Switch, Tag } from './ui';
import { framedAspect, sentenceCase } from './format';

const sampleName = (title: string) => title.replace(/^\d+\.\s*/, '');

const SiteViewer: React.FC = () => {
  const {
    rawImageUrl,
    segmentationMasks,
    visibleMaskIds,
    isOverlayActive,
    toggleOverlayActive,
    uploadCustomImage,
    isSegmenting,
    generationStage,
    currentFile,
  } = useResiliCityStore();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const aspect = size ? size.w / size.h : 3 / 2;

  // Real pixel size of the photo being shown (also drives the frame's aspect ratio).
  useEffect(() => {
    if (!rawImageUrl) return;
    const img = new Image();
    img.onload = () => img.naturalWidth && setSize({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = rawImageUrl;
  }, [rawImageUrl]);

  const onFiles = (files: FileList | null) => {
    const f = files?.[0];
    if (f && f.type.startsWith('image/')) void uploadCustomImage(f);
  };

  const visibleMasks = segmentationMasks.filter((m) => visibleMaskIds.includes(m.id));

  return (
    <Panel
      eyebrow="Street photograph"
      title={currentFile ? currentFile.name : 'Sample street'}
      actions={
        <>
          <Switch checked={isOverlayActive} onChange={toggleOverlayActive} label="Surface overlay" />
          <Button icon={<ImageUp className="w-4 h-4" />} onClick={() => inputRef.current?.click()}>
            Upload
          </Button>
        </>
      }
      bodyClassName="p-0 bg-slate-50"
    >
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={(e) => onFiles(e.target.files)} />
      <div
        className={`relative bg-slate-100 overflow-hidden transition-[outline-color] outline-2 -outline-offset-2 ${
          dragging ? 'outline outline-[#0d7a5f]' : 'outline-transparent'
        }`}
        style={framedAspect(aspect, 480)}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          onFiles(e.dataTransfer.files);
        }}
      >
        {rawImageUrl && (
          <img key={rawImageUrl} src={rawImageUrl} alt="Street photograph" className="rc-fade-in absolute inset-0 w-full h-full object-cover" />
        )}

        {/* Segmentation overlay: polygons are in percent-of-image coordinates */}
        <svg
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          className={`absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-500 ${isOverlayActive ? 'opacity-100' : 'opacity-0'}`}
        >
          {visibleMasks.map((m) =>
            (m.polygons?.length ? m.polygons : m.polygonPoints ? [m.polygonPoints] : []).map((poly, i) => (
              <polygon
                key={`${m.id}-${i}`}
                points={poly.map(([x, y]) => `${x},${y}`).join(' ')}
                fill={m.color}
                fillOpacity={0.35}
                stroke={m.color}
                strokeWidth={0.3}
                vectorEffect="non-scaling-stroke"
              />
            ))
          )}
        </svg>

        {dragging && (
          <div className="absolute inset-0 bg-white/80 flex items-center justify-center text-sm text-[#0d7a5f]">
            Drop the photo to analyse it
          </div>
        )}

        {isSegmenting && <div className="rc-scan" />}
        {isSegmenting && (
          <div className="absolute inset-x-0 bottom-0 bg-white/90 px-4 py-3 flex flex-col gap-2">
            <span className="text-xs text-slate-700">{sentenceCase(generationStage || 'Analysing site')}</span>
            <div className="rc-progress" />
          </div>
        )}
      </div>

      <div className="flex items-center justify-between px-5 py-3 border-t border-slate-200 text-xs text-slate-500">
        <span>{currentFile ? `${(currentFile.size / 1024 / 1024).toFixed(1)} MB` : 'Reference sample'}</span>
        <span>{size ? `${size.w} × ${size.h} px` : ''}</span>
      </div>
    </Panel>
  );
};

const SamplePicker: React.FC = () => {
  const { currentSceneId, loadSampleScene, isGenerating, isSegmenting } = useResiliCityStore();
  return (
    <div className="rc-stagger grid grid-cols-3 sm:grid-cols-5 gap-2">
      {SAMPLE_SCENES.map((s) => {
        const active = s.id === currentSceneId;
        return (
          <button
            key={s.id}
            type="button"
            disabled={isGenerating || isSegmenting}
            onClick={() => loadSampleScene(s.id)}
            title={sampleName(s.title)}
            className={`group relative aspect-[4/3] overflow-hidden border cursor-pointer disabled:cursor-not-allowed ${
              active ? 'border-[#0d7a5f] ring-1 ring-[#0d7a5f]' : 'border-slate-200 hover:border-[#0d7a5f]'
            }`}
          >
            <img src={s.rawImageUrl} alt="" className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" />
            <span className="absolute inset-x-0 bottom-0 bg-white/90 px-1.5 py-1 text-[10px] text-slate-700 truncate text-left">
              {sampleName(s.title)}
            </span>
          </button>
        );
      })}
    </div>
  );
};

const SurfaceComposition: React.FC = () => {
  const { segmentationMasks, visibleMaskIds, toggleMaskVisibility, segmentationSource, segmentationModelInfo } = useResiliCityStore();
  return (
    <Panel
      eyebrow="Surface composition"
      title="What the street is made of"
      actions={<Tag tone={segmentationSource === 'segformer' ? 'green' : 'neutral'}>{segmentationSource === 'segformer' ? 'Measured' : 'Sample data'}</Tag>}
    >
      <ul className="rc-stagger flex flex-col gap-3">
        {segmentationMasks.map((m, i) => {
          const on = visibleMaskIds.includes(m.id);
          return (
            <li key={m.id}>
              <button
                type="button"
                onClick={() => toggleMaskVisibility(m.id)}
                className={`w-full flex flex-col gap-1.5 text-left cursor-pointer transition-opacity ${on ? '' : 'opacity-40'}`}
                title="Show or hide on the overlay"
              >
                <span className="flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2 text-slate-700">
                    <span className="w-2.5 h-2.5" style={{ backgroundColor: m.color }} />
                    {m.label}
                  </span>
                  <span className="text-slate-900 tabular-nums">{m.areaPercentage}%</span>
                </span>
                <span className="h-1.5 bg-slate-100 w-full block">
                  <span
                    className="rc-bar block h-full"
                    style={{ width: `${m.areaPercentage}%`, backgroundColor: m.color, animationDelay: `${i * 60}ms` }}
                  />
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      <p className="mt-4 text-xs text-slate-500">
        {segmentationSource === 'segformer'
          ? `Segmentation: ${segmentationModelInfo?.name ?? 'SegFormer'}. Click a surface to toggle it on the overlay.`
          : 'Reference values for the sample street. Upload a photo for measured surfaces.'}
      </p>
    </Panel>
  );
};

const BaselineRisk: React.FC = () => {
  const { heatMetrics } = useResiliCityStore();
  const pct = Math.min(100, (heatMetrics.baseScore / 10) * 100);
  return (
    <Panel eyebrow="Baseline heat risk" title="Before any intervention" actions={<Tag tone="amber">Estimate</Tag>}>
      <div className="flex items-baseline gap-3">
        <span className="text-4xl font-semibold text-slate-900">
          <AnimatedNumber value={heatMetrics.baseScore} />
        </span>
        <span className="text-sm text-slate-500">/ 10</span>
        <span className="ml-auto text-sm text-amber-800">{exposureLabel(heatMetrics.baseScore)}</span>
      </div>
      <div className="relative mt-4 h-2 bg-gradient-to-r from-emerald-500 via-amber-400 to-rose-500">
        <span
          className="absolute -top-1.5 w-0.5 h-5 bg-slate-900 transition-[left] duration-700 ease-out"
          style={{ left: `calc(${pct}% - 1px)` }}
        />
      </div>
      <div className="mt-1.5 flex justify-between text-[11px] text-slate-400">
        <span>Low</span>
        <span>High</span>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2">
        <Stat label="Surface treatments" value={`-${heatMetrics.surfaceCooling.toFixed(1)}`} sub="points available" />
        <Stat label="Shade and canopy" value={`-${heatMetrics.shadeCooling.toFixed(1)}`} sub="points available" />
      </div>
      <p className="mt-4 text-xs text-slate-500">
        Calibrated against satellite land surface temperature. Not an air temperature measurement.
      </p>
    </Panel>
  );
};

export const SiteSection: React.FC = () => {
  const plan = useResiliCityStore((s) => s.spatialDesignPlan);
  const m = plan?.street_metrics;
  const fmt = (v?: number | null) => (v ? `${v.toFixed(1)} m` : 'Not measured');

  return (
    <section className="flex flex-col gap-5">
      <SectionHeading
        id="site"
        number="01"
        title="Site"
        description="Upload a street photo or pick a sample. Surfaces are segmented and the street is measured in metres."
      />
      <div className="grid grid-cols-1 lg:grid-cols-[1.45fr_1fr] gap-5">
        <div className="flex flex-col gap-3">
          <SiteViewer />
          <SamplePicker />
        </div>
        <div className="flex flex-col gap-5">
          <SurfaceComposition />
          <BaselineRisk />
        </div>
      </div>
      {m && (
        <div className="rc-stagger grid grid-cols-2 md:grid-cols-4 gap-2">
          <Stat label="Carriageway" value={`${m.road_width_m.toFixed(1)} m`} sub={m.scale_source === 'user' ? 'Entered' : 'Assumed scale'} />
          <Stat label="Left footpath" value={fmt(m.left_sidewalk_m)} sub={m.left_range_m ? `${m.left_range_m[0]}–${m.left_range_m[1]} m` : undefined} />
          <Stat label="Right footpath" value={fmt(m.right_sidewalk_m)} sub={m.right_range_m ? `${m.right_range_m[0]}–${m.right_range_m[1]} m` : undefined} />
          <Stat label="Uncertainty" value={`±${m.uncertainty_pct}%`} sub={m.reliable ? 'Usable for planning' : 'Survey before design'} />
        </div>
      )}
    </section>
  );
};
