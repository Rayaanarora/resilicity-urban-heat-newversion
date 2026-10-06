import React, { useRef } from 'react';
import { useResiliCityStore } from '../store/useResiliCityStore';
import { FileText, Download, X, Sparkles, ShieldCheck, Info } from 'lucide-react';
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
    designProfile,
    spatialDesignPlan,
    visualizationOutput,
  } = useResiliCityStore();

  if (!isOpen) return null;

  const activeInterventionsList = interventions.filter((i) => activeInterventionIds.includes(i.id));

  const handleExportJSON = () => {
    const data = {
      timestamp: new Date().toISOString(),
      sceneId: currentSceneId,
      climateZone,
      designProfile,
      heatMetrics,
      spatialDesignPlan,
      visualizationOutput,
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
      <div className="bg-white border border-slate-200 rounded-2xl max-w-4xl w-full max-h-[92dvh] flex flex-col overflow-hidden shadow-2xl">
        <div className="px-6 py-4 border-b border-slate-100 flex justify-between items-center bg-slate-50">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-teal-700" />
            <h2 className="text-base font-bold font-sans text-slate-900">
              ResiliCity Urban Resilience Assessment & Design Report
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
          {/* Header Summary */}
          <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl flex items-center justify-between">
            <div className="flex flex-col gap-1 max-w-xl">
              <span className="text-xs font-mono text-slate-400 uppercase">Assessment Summary · Profile: {designProfile.toUpperCase()}</span>
              <h3 className="text-lg font-bold text-slate-900 font-sans">
                Projected Cooling: -{heatMetrics.estimatedTempReductionC}°C Surface Temperature
              </h3>
              <p className="text-xs text-slate-600">
                {spatialDesignPlan?.overall_design_intent ||
                  'AI urban resilience intervention plan balancing canopy shading, high-albedo surfaces, and architectural tensile structures.'}
              </p>
            </div>
            <div className="text-right font-mono">
              <div className="text-xs text-slate-400">Heat Risk Score</div>
              <div className="text-2xl font-black text-teal-700">
                {heatMetrics.baseScore} → {heatMetrics.projectedScore}
              </div>
            </div>
          </div>

          {/* Hero Visual Comparison */}
          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-1.5">
              <span className="text-xs font-mono text-slate-400 uppercase font-semibold">
                Original Street Photograph
              </span>
              <div className="relative aspect-[3/2] bg-slate-900 rounded-xl overflow-hidden border border-slate-200">
                <img
                  src={rawImageUrl || '/samples/sample_1_dense_urban.jpg'}
                  alt="Original Scene"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <span className="text-xs font-mono text-teal-700 uppercase font-semibold flex items-center gap-1">
                <Sparkles className="w-3.5 h-3.5" />
                AI-Generated Design Visualization
              </span>
              <div className="relative aspect-[3/2] bg-slate-900 rounded-xl overflow-hidden border border-teal-500/40">
                <img
                  src={generatedImageUrl || rawImageUrl || '/samples/sample_1_dense_urban.jpg'}
                  alt="AI-Generated Design Visualization"
                  className="w-full h-full object-contain"
                />
                <span className="absolute bottom-2 right-2 bg-slate-900/80 text-[10px] font-mono text-white px-2 py-0.5 rounded">
                  AI-generated design visualization
                </span>
              </div>
            </div>
          </div>

          {/* Selected Cooling Interventions Table with "Why" */}
          <div className="flex flex-col gap-2">
            <h4 className="text-xs font-mono uppercase text-slate-400 font-semibold">
              Selected Cooling Interventions & Physical Rationale
            </h4>
            <div className="border border-slate-200 rounded-xl overflow-hidden">
              <table className="w-full text-left text-xs font-sans">
                <thead className="bg-slate-50 text-slate-500 uppercase text-[10px] border-b border-slate-200 font-mono">
                  <tr>
                    <th className="p-3">Priority</th>
                    <th className="p-3">Intervention</th>
                    <th className="p-3">Target Region</th>
                    <th className="p-3">Why Selected (Physical Rationale)</th>
                    <th className="p-3 text-right">Cooling Impact</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white text-slate-800">
                  {activeInterventionsList.map((item) => {
                    const anyItem = item as any;
                    const reason = anyItem.reason || item.description;
                    return (
                      <tr key={item.id}>
                        <td className="p-3 font-mono font-bold">#{item.priority}</td>
                        <td className="p-3 font-semibold">{item.title}</td>
                        <td className="p-3 uppercase text-slate-500 font-mono text-[10px]">
                          {item.targetRegion}
                        </td>
                        <td className="p-3 text-slate-600 text-[11px] leading-relaxed max-w-xs">
                          {reason}
                        </td>
                        <td className="p-3 text-right font-bold text-teal-700 font-mono">
                          -{item.coolingImpact.toFixed(1)}°C
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Methodology, Confidence & Assumptions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs bg-slate-50 p-4 rounded-xl border border-slate-200">
            <div className="flex flex-col gap-1.5">
              <span className="font-bold text-slate-800 flex items-center gap-1.5 font-mono uppercase text-[11px]">
                <ShieldCheck className="w-4 h-4 text-emerald-600" />
                Confidence & Methodology
              </span>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                <strong>Methodology:</strong> Multi-stage perception via SegFormer computer vision constraints $\to$ Gemini spatial planning reasoning $\to$ Gemini multimodal architectural generative image editing. Thermal impact is calculated via satellite-calibrated Landsat regression with spatial cross-validation error bounds (±{heatMetrics.modelRmseC?.toFixed(1) || '0.8'}°C).
              </p>
            </div>

            <div className="flex flex-col gap-1.5">
              <span className="font-bold text-slate-800 flex items-center gap-1.5 font-mono uppercase text-[11px]">
                <Info className="w-4 h-4 text-teal-600" />
                Assumptions & Disclaimer
              </span>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                <strong>Assumptions:</strong> Mature tree canopies assumed at 5–10 years post-planting. Pavement coatings assume solar reflectance index (SRI) ≥ 78. The generative redesign image is an architectural concept visualization; the separate numerical thermal model determines calculated temperature reductions.
              </p>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
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
