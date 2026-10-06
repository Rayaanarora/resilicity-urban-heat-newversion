import React, { useRef } from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import { FileText, Download, X, Sparkles } from 'lucide-react';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

interface ImpactReportModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ImpactReportModal: React.FC<ImpactReportModalProps> = ({ isOpen, onClose }) => {
  const reportRef = useRef<HTMLDivElement>(null);
  const {
    rawImageUrl,
    generatedImageUrl,
    heatMetrics,
    interventions,
    activeInterventionIds,
    currentSceneId,
    climateZone,
  } = useResiliCityStore();

  if (!isOpen) return null;

  const activeInterventionsList = interventions.filter((i) => activeInterventionIds.includes(i.id));

  const handleExportJSON = () => {
    const data = {
      timestamp: new Date().toISOString(),
      sceneId: currentSceneId,
      climateZone,
      heatMetrics,
      activeInterventions: activeInterventionsList,
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `resilicity-report-${currentSceneId}.json`;
    a.click();
  };

  const handleExportPDF = async () => {
    if (!reportRef.current) return;
    try {
      const canvas = await html2canvas(reportRef.current, { scale: 2 });
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save(`ResiliCity-Heat-Assessment-${currentSceneId}.pdf`);
    } catch (e) {
      console.error('Failed to generate PDF export', e);
    }
  };

  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white border border-slate-200 rounded-2xl max-w-3xl w-full max-h-[90dvh] flex flex-col overflow-hidden shadow-2xl">
        <div className="px-6 py-4 border-b border-slate-100 flex justify-between items-center bg-slate-50">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-teal-700" />
            <h2 className="text-base font-bold font-sans text-slate-900">
              ResiliCity Heat Resilience Impact Report
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-200 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div ref={reportRef} className="p-6 overflow-y-auto flex flex-col gap-6 bg-white font-sans">
          <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl flex items-center justify-between">
            <div className="flex flex-col gap-1">
              <span className="text-xs font-mono text-slate-400 uppercase">Assessment Summary</span>
              <h3 className="text-lg font-bold text-slate-900 font-sans">
                Projected Cooling: -{heatMetrics.estimatedTempReductionC}°C surface temp (
                {heatMetrics.tempSource === 'assumed'
                  ? 'assumption-based estimate'
                  : heatMetrics.tempSource === 'model'
                    ? `satellite-calibrated scenario estimate, typical model error ±${heatMetrics.modelRmseC?.toFixed(1)}°C`
                    : `${heatMetrics.modelCoolingC}°C from satellite-calibrated model (typical error ±${heatMetrics.modelRmseC?.toFixed(1)}°C) + ${heatMetrics.assumedCoolingC}°C literature/assumed`}
                )
              </h3>
              {heatMetrics.modelUnreliable && (
                <p className="text-xs text-amber-700">
                  The satellite regression predicts {heatMetrics.modelDeltaC}°C for the greening measures, which is smaller than its typical
                  error (±{heatMetrics.modelRmseC?.toFixed(1)}°C), so literature values are used instead.
                </p>
              )}
              <p className="text-xs text-slate-500">
                Calculated across {activeInterventionsList.length} active VLM cooling interventions in {climateZone.toUpperCase()} climate zone.
              </p>
            </div>
            <div className="text-right font-mono">
              <div className="text-xs text-slate-400">Heat Index Drop</div>
              <div className="text-2xl font-black text-teal-700">
                {heatMetrics.baseScore} → {heatMetrics.projectedScore}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-1.5">
              <span className="text-xs font-mono text-slate-400 uppercase font-semibold">Original Baseline</span>
              <img
                src={rawImageUrl || ''}
                alt="Original Scene"
                className="w-full h-40 object-cover rounded-lg border border-slate-200"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <span className="text-xs font-mono text-teal-700 uppercase font-semibold flex items-center gap-1">
                <Sparkles className="w-3.5 h-3.5" />
                Inprinted Resilient Scene
              </span>
              <img
                src={generatedImageUrl || ''}
                alt="Resilient Scene"
                className="w-full h-40 object-cover rounded-lg border border-teal-500/40"
              />
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <h4 className="text-xs font-mono uppercase text-slate-400 font-semibold">
              Selected Cooling Interventions
            </h4>
            <div className="border border-slate-200 rounded-xl overflow-hidden">
              <table className="w-full text-left text-xs font-sans">
                <thead className="bg-slate-50 text-slate-500 uppercase text-[10px] border-b border-slate-200 font-mono">
                  <tr>
                    <th className="p-3">Priority</th>
                    <th className="p-3">Intervention</th>
                    <th className="p-3">Target Region</th>
                    <th className="p-3">Cost Tier</th>
                    <th className="p-3 text-right">Cooling Delta</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white text-slate-800">
                  {activeInterventionsList.map((item) => (
                    <tr key={item.id}>
                      <td className="p-3 font-mono">#{item.priority}</td>
                      <td className="p-3 font-semibold">{item.title}</td>
                      <td className="p-3 uppercase text-slate-500 font-mono">{item.targetRegion}</td>
                      <td className="p-3 uppercase text-teal-700 font-bold font-mono">{item.estCostTier}</td>
                      <td className="p-3 text-right font-bold text-teal-700 font-mono">-{item.coolingImpact.toFixed(1)} PTS</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-50 border border-slate-200 p-3.5 rounded-xl flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-700 font-black font-mono">
                SDG 11
              </div>
              <div className="flex flex-col">
                <span className="text-xs font-bold text-slate-900 font-sans">Sustainable Cities</span>
                <span className="text-[11px] text-slate-500">Target 11.7: Green Public Urban Spaces</span>
              </div>
            </div>

            <div className="bg-slate-50 border border-slate-200 p-3.5 rounded-xl flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-teal-700 font-black font-mono">
                SDG 13
              </div>
              <div className="flex flex-col">
                <span className="text-xs font-bold text-slate-900 font-sans">Climate Action</span>
                <span className="text-[11px] text-slate-500">Target 13.1: Adaptive Resilience Capacity</span>
              </div>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-slate-100 bg-slate-50 flex justify-end gap-3">
          <button
            onClick={handleExportJSON}
            className="px-4 py-2 bg-white hover:bg-slate-100 text-slate-700 text-xs font-semibold rounded-lg border border-slate-200 font-mono flex items-center gap-2 transition-colors"
          >
            <Download className="w-4 h-4" />
            Export JSON Data
          </button>

          <button
            onClick={handleExportPDF}
            className="px-4 py-2 bg-[#0d7a5f] hover:bg-[#0b6b53] text-white text-xs font-bold rounded-lg font-mono flex items-center gap-2 transition-colors shadow-sm"
          >
            <FileText className="w-4 h-4" />
            Download PDF Report
          </button>
        </div>
      </div>
    </div>
  );
};
