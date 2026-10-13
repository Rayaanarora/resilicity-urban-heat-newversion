import React, { useLayoutEffect, useRef, useState } from 'react';
import { useResiliCityStore, type NavTab } from '../store/useResiliCityStore';

const TABS: { id: NavTab; label: string }[] = [
  { id: 'analysis', label: 'Analysis' },
  { id: 'saved_runs', label: 'Saved runs' },
  { id: 'method_note', label: 'Method' },
];

export const TopBar: React.FC = () => {
  const { activeTab, setActiveTab, backendStatus } = useResiliCityStore();
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const [indicator, setIndicator] = useState({ left: 0, width: 0 });

  // Slide the underline to the active tab.
  useLayoutEffect(() => {
    const el = tabRefs.current[activeTab];
    if (el) setIndicator({ left: el.offsetLeft, width: el.offsetWidth });
  }, [activeTab]);

  return (
    <header className="sticky top-0 z-50 bg-white/95 backdrop-blur border-b border-slate-200">
      <div className="max-w-[1280px] mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-6">
        <button type="button" onClick={() => setActiveTab('analysis')} className="flex items-center gap-3 cursor-pointer">
          <span className="w-7 h-7 bg-[#0d7a5f] flex items-center justify-center text-white text-sm font-semibold">R</span>
          <span className="flex flex-col items-start leading-tight">
            <span className="text-[15px] font-semibold text-slate-900">ResiliCity</span>
            <span className="text-[10px] tracking-[0.12em] uppercase text-slate-500">Urban heat lab</span>
          </span>
        </button>

        <nav className="relative hidden sm:flex items-center gap-8 h-14">
          {TABS.map((t) => (
            <button
              key={t.id}
              ref={(el) => {
                tabRefs.current[t.id] = el;
              }}
              type="button"
              onClick={() => setActiveTab(t.id)}
              className={`h-full text-sm cursor-pointer ${activeTab === t.id ? 'text-slate-900' : 'text-slate-500 hover:text-slate-900'}`}
            >
              {t.label}
            </button>
          ))}
          <span
            className="absolute bottom-0 h-0.5 bg-[#0d7a5f] transition-all duration-300 ease-out"
            style={{ left: indicator.left, width: indicator.width }}
          />
        </nav>

        <div className="flex items-center gap-4 text-xs text-slate-600">
          <span className="flex items-center gap-2" title={backendStatus.isConnected ? 'Backend connected' : 'Backend offline'}>
            <span className={`w-2 h-2 ${backendStatus.isConnected ? 'bg-[#0d7a5f]' : 'bg-rose-500'}`} />
            <span className="hidden sm:inline">{backendStatus.isConnected ? 'Backend connected' : 'Backend offline'}</span>
          </span>
          <span className="hidden md:inline text-slate-400">
            Segmentation {backendStatus.segmenterAvailable ? 'live' : 'unavailable'}
          </span>
        </div>
      </div>

      {/* Mobile tabs */}
      <nav className="sm:hidden flex border-t border-slate-200">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setActiveTab(t.id)}
            className={`flex-1 py-2.5 text-sm border-b-2 ${activeTab === t.id ? 'border-[#0d7a5f] text-slate-900' : 'border-transparent text-slate-500'}`}
          >
            {t.label}
          </button>
        ))}
      </nav>
    </header>
  );
};
