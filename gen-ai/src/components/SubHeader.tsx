import React from 'react';
import { RotateCcw, Download, Upload, Layers, Flame, Sparkles, SlidersHorizontal } from 'lucide-react';
import { useResiliCityStore } from '../store/useResiliCityStore';

interface SubHeaderProps {
  onExportClick: () => void;
}

export const SubHeader: React.FC<SubHeaderProps> = ({ onExportClick }) => {
  const { resetAll } = useResiliCityStore();

  return (
    <div className="flex flex-col gap-6 mb-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="text-[11px] font-mono font-semibold tracking-widest text-slate-400 uppercase mb-1">
            WORKSPACE <span className="text-slate-300">→</span> RUN / 0248
          </div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
            Heat-risk inspection
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Turn a street image into an explainable cooling plan. Every score is an estimate, not a measured temperature.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={resetAll}
            className="px-4 py-2 bg-white border border-slate-200 hover:border-slate-300 hover:bg-slate-50 text-slate-700 text-xs font-semibold rounded-xl flex items-center gap-2 transition-all shadow-sm font-mono"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
            Reset run
          </button>

          <button
            onClick={onExportClick}
            className="px-4 py-2 bg-white border border-slate-200 hover:border-slate-300 hover:bg-slate-50 text-slate-700 text-xs font-semibold rounded-xl flex items-center gap-2 transition-all shadow-sm font-mono"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" />
            Export notes
          </button>
        </div>
      </div>

      <div className="flex items-center gap-3 overflow-x-auto py-2 border-b border-slate-200/60 text-xs font-mono">
        <div className="flex items-center gap-2 text-[#0d7a5f] font-semibold bg-emerald-50 border border-emerald-200/80 px-3 py-1.5 rounded-full">
          <Upload className="w-3.5 h-3.5" />
          <span>Input</span>
        </div>
        <span className="text-slate-300">—</span>

        <div className="flex items-center gap-2 text-[#0d7a5f] font-semibold bg-emerald-50 border border-emerald-200/80 px-3 py-1.5 rounded-full">
          <Layers className="w-3.5 h-3.5" />
          <span>Segment</span>
        </div>
        <span className="text-slate-300">—</span>

        <div className="flex items-center gap-2 text-slate-400">
          <Flame className="w-3.5 h-3.5" />
          <span>Score</span>
        </div>
        <span className="text-slate-300">—</span>

        <div className="flex items-center gap-2 text-slate-400">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Plan</span>
        </div>
        <span className="text-slate-300">—</span>

        <div className="flex items-center gap-2 text-slate-400">
          <SlidersHorizontal className="w-3.5 h-3.5" />
          <span>Compare</span>
        </div>

        <div className="ml-auto flex items-center gap-1.5 text-emerald-600 text-xs font-semibold">
          <span className="w-2 h-2 rounded-full bg-emerald-500" />
          <span>Ready for analysis</span>
        </div>
      </div>
    </div>
  );
};
