import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="mt-12 py-6 border-t border-slate-200/80 text-xs font-mono text-slate-400 flex flex-col md:flex-row items-center justify-between gap-4">
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-emerald-500" />
        <span className="text-slate-600 font-semibold">ResiliCity / Colab boundary</span>
      </div>

      <div className="text-slate-400 text-center">
        Scores are transparent literature-based cooling coefficients. Not CFD or physical simulation.
      </div>

      <div className="text-slate-400 font-semibold">
        v0.1 · local demo
      </div>
    </footer>
  );
};
