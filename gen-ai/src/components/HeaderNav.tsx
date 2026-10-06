import React from 'react';
import { Layers } from 'lucide-react';
import { useResiliCityStore } from '../store/useResiliCityStore';

export const HeaderNav: React.FC = () => {
  const { activeTab, setActiveTab, backendStatus } = useResiliCityStore();

  return (
    <header className="h-16 bg-white border-b border-slate-200/80 px-6 flex items-center justify-between sticky top-0 z-50">
      <div className="flex items-center gap-3 cursor-pointer" onClick={() => setActiveTab('analysis')}>
        <div className="w-9 h-9 rounded-xl bg-[#0d7a5f] flex items-center justify-center text-white shadow-sm">
          <Layers className="w-5 h-5" />
        </div>
        <div className="flex flex-col">
          <span className="text-xl font-extrabold text-slate-900 tracking-tight font-sans">
            ResiliCity
          </span>
          <span className="text-[9px] font-mono font-semibold tracking-widest text-slate-400 uppercase">
            GENAI / URBAN HEAT LAB
          </span>
        </div>
      </div>

      <nav className="flex items-center gap-8 h-full">
        <button
          onClick={() => setActiveTab('analysis')}
          className={`h-full flex items-center text-sm font-semibold transition-colors px-1 border-b-2 ${
            activeTab === 'analysis'
              ? 'text-slate-900 border-[#0d7a5f]'
              : 'text-slate-500 border-transparent hover:text-slate-900'
          }`}
        >
          Analysis
        </button>

        <button
          onClick={() => setActiveTab('saved_runs')}
          className={`h-full flex items-center text-sm font-semibold transition-colors px-1 border-b-2 ${
            activeTab === 'saved_runs'
              ? 'text-slate-900 border-[#0d7a5f]'
              : 'text-slate-500 border-transparent hover:text-slate-900'
          }`}
        >
          Saved runs
        </button>

        <button
          onClick={() => setActiveTab('method_note')}
          className={`h-full flex items-center text-sm font-semibold transition-colors px-1 border-b-2 ${
            activeTab === 'method_note'
              ? 'text-slate-900 border-[#0d7a5f]'
              : 'text-slate-500 border-transparent hover:text-slate-900'
          }`}
        >
          Method note
        </button>
      </nav>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 px-3 py-1 rounded-full text-xs font-mono font-medium text-slate-700">
          <span
            className={`w-2 h-2 rounded-full ${
              backendStatus.isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
            }`}
          />
          <span>{backendStatus.isConnected ? 'FastAPI :8000' : 'Backend offline'}</span>
        </div>

        <div className="bg-emerald-50 border border-emerald-200/80 px-2.5 py-1 rounded-lg text-[10px] font-mono font-bold text-emerald-700 tracking-wider uppercase">
          {backendStatus.segmenterAvailable ? 'SegFormer Live' : 'Stage 1 Ready'}
        </div>
      </div>
    </header>
  );
};

