import React from 'react';
import type { SourceTag } from '../types/planner';

const SOURCE_STYLE: Record<string, string> = {
  model: 'border-[#0d7a5f] text-[#0a6550]',
  measured: 'border-slate-500 text-slate-700',
  literature: 'border-sky-700 text-sky-800',
  rule: 'border-slate-400 text-slate-600',
  assumed: 'border-amber-500 text-amber-800',
};

const SOURCE_HELP: Record<string, string> = {
  model: 'Predicted by the site heat model (spatial cross-validated)',
  measured: 'Measured: satellite data or the street photo',
  literature: 'Indicative range from published studies, not a model prediction',
  rule: 'Derived by a design rule from measured inputs',
  assumed: 'Assumed or indicative value; replace with local data',
};

/** Small square tag naming where a number comes from. */
export const SourceChip: React.FC<{ source?: SourceTag | string | null }> = ({ source }) => {
  if (!source) return null;
  const key = String(source).split(' ')[0];
  return (
    <span
      title={SOURCE_HELP[key] ?? String(source)}
      className={`inline-block border px-1 text-[10px] uppercase tracking-wide leading-4 ${SOURCE_STYLE[key] ?? SOURCE_STYLE.rule}`}
    >
      {key}
    </span>
  );
};

interface FigureProps {
  label?: string;
  value: number | null | undefined;
  unit?: string;
  band?: [number, number] | null;
  source?: SourceTag | string | null;
  decimals?: number;
  format?: (n: number) => string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

/** A number with its uncertainty band and source tag. Shows an em dash when the value is absent. */
export const Figure: React.FC<FigureProps> = ({ label, value, unit = '', band, source, decimals = 1, format, size = 'md', className = '' }) => {
  const f = format ?? ((n: number) => n.toFixed(decimals));
  const sizeCls = size === 'lg' ? 'text-2xl' : size === 'sm' ? 'text-sm' : 'text-lg';
  return (
    <div className={`flex flex-col gap-0.5 ${className}`}>
      {label && <span className="rc-card-header-label">{label}</span>}
      <div className="flex items-baseline gap-2 flex-wrap">
        <span className={`${sizeCls} font-semibold text-slate-900 tabular-nums`}>
          {value === null || value === undefined ? '—' : `${f(value)}${unit}`}
        </span>
        <SourceChip source={source} />
      </div>
      {band && value !== null && value !== undefined && (
        <span className="text-xs text-slate-500 tabular-nums">
          range {f(band[0])} to {f(band[1])}
          {unit}
        </span>
      )}
    </div>
  );
};
