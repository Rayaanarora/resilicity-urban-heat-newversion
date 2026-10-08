import React from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import { Play, Download, Calendar, MapPin, Sparkles, Layers } from 'lucide-react';

interface SavedRun {
  id: string;
  runCode: string;
  timestamp: string;
  location: string;
  imageUrl: string;
  baseScore: number;
  projectedScore: number;
  activeInterventionsCount: number;
  interventions: string[];
}

const SAVED_RUNS_DATA: SavedRun[] = [
  {
    id: 'run-0248',
    runCode: 'RUN / 0248',
    timestamp: '2026-10-02 18:42',
    location: 'North Loop / sample',
    imageUrl: '/samples/urban_street_after.png',
    baseScore: 6.9,
    projectedScore: 5.1,
    activeInterventionsCount: 2,
    interventions: ['Cool roof coating', 'Tree canopy'],
  },
  {
    id: 'run-0247',
    runCode: 'RUN / 0247',
    timestamp: '2026-10-01 14:15',
    location: 'Downtown Commercial Corridor',
    imageUrl: '/samples/urban_street_before.png',
    baseScore: 8.4,
    projectedScore: 5.4,
    activeInterventionsCount: 3,
    interventions: ['Cool roof coating', 'Tree canopy', 'Cool pavement'],
  },
  {
    id: 'run-0246',
    runCode: 'RUN / 0246',
    timestamp: '2026-09-28 11:30',
    location: 'Eastside Residential Courtyard',
    imageUrl: '/samples/urban_street_after.png',
    baseScore: 7.8,
    projectedScore: 6.0,
    activeInterventionsCount: 2,
    interventions: ['Courtyard Tree Cluster', 'Permeable Grass Pavers'],
  },
];

interface SavedRunsViewProps {
  onExportClick: () => void;
}

export const SavedRunsView: React.FC<SavedRunsViewProps> = ({ onExportClick }) => {
  const { loadSampleScene } = useResiliCityStore();

  return (
    <div className="flex flex-col gap-6 animate-fade-in">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="text-[11px] font-mono font-semibold tracking-widest text-slate-400 uppercase mb-1">
            WORKSPACE → SAVED INSPECTION RUNS
          </div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
            Saved inspection runs
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Illustrative sample runs for the interface concept. They are not stored results from the model; only the first run matches the live sample scene.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs text-slate-500 bg-white border border-slate-200 px-3 py-1.5 rounded-xl shadow-sm">
          <Layers className="w-4 h-4 text-[#0d7a5f]" />
          <span>{SAVED_RUNS_DATA.length} Saved Inspections</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {SAVED_RUNS_DATA.map((run) => {
          const delta = (run.baseScore - run.projectedScore).toFixed(1);
          return (
            <div key={run.id} className="rc-card p-5 flex flex-col justify-between gap-4 hover:border-slate-300 transition-all shadow-sm">
              <div className="relative rounded-xl overflow-hidden aspect-[16/10] bg-slate-900 border border-slate-200">
                <img
                  src={run.imageUrl}
                  alt={run.location}
                  className="w-full h-full object-cover"
                />
                <div className="absolute top-3 left-3 bg-slate-900/85 backdrop-blur text-white px-2.5 py-1 rounded-md text-[11px] font-mono font-bold">
                  {run.runCode}
                </div>
                <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur text-slate-800 px-2.5 py-0.5 rounded-md text-[11px] font-semibold flex items-center gap-1">
                  <MapPin className="w-3 h-3 text-slate-500" />
                  <span>{run.location}</span>
                </div>
              </div>

              <div className="flex items-center justify-between bg-slate-50 p-3 rounded-xl border border-slate-100">
                <div className="flex flex-col">
                  <span className="text-[10px] font-mono uppercase text-slate-400">Baseline</span>
                  <span className="text-lg font-black text-amber-700 font-sans">{run.baseScore.toFixed(1)}</span>
                </div>

                <div className="text-slate-300 font-mono text-sm">→</div>

                <div className="flex flex-col items-center">
                  <span className="text-[10px] font-mono uppercase text-slate-400">Projected</span>
                  <span className="text-lg font-black text-teal-700 font-sans">{run.projectedScore.toFixed(1)}</span>
                </div>

                <div className="flex flex-col items-end">
                  <span className="text-[10px] font-mono uppercase text-slate-400">Delta</span>
                  <span className="text-xs font-bold text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded font-mono">
                    -{delta} PTS
                  </span>
                </div>
              </div>

              <div className="flex flex-col gap-1.5 text-xs font-sans">
                <span className="text-[11px] font-mono uppercase text-slate-400 font-semibold">
                  Autonomous Interventions ({run.activeInterventionsCount})
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {run.interventions.map((intName, idx) => (
                    <span key={idx} className="bg-slate-100 text-slate-700 px-2 py-0.5 rounded text-[11px] font-medium flex items-center gap-1">
                      <Sparkles className="w-3 h-3 text-emerald-600" />
                      {intName}
                    </span>
                  ))}
                </div>
              </div>

              <div className="flex items-center justify-between border-t border-slate-100 pt-3 mt-1">
                <span className="text-[11px] font-mono text-slate-400 flex items-center gap-1">
                  <Calendar className="w-3 h-3 text-slate-400" />
                  {run.timestamp}
                </span>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => loadSampleScene('north-loop-sample')}
                    className="px-3 py-1.5 bg-[#0d7a5f] hover:bg-[#0b6b53] text-white text-xs font-semibold rounded-lg flex items-center gap-1 transition-colors shadow-sm font-mono"
                  >
                    <Play className="w-3 h-3 fill-current" />
                    Load
                  </button>

                  <button
                    onClick={onExportClick}
                    className="p-1.5 border border-slate-200 hover:bg-slate-50 text-slate-500 rounded-lg transition-colors"
                    title="Export PDF Report"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
