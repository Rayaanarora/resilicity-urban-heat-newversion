import React, { useState } from 'react';
import { HeaderNav } from './components/HeaderNav';
import { SubHeader } from './components/SubHeader';
import { SiteContextCard } from './components/SiteContextCard';
import { SurfaceMasksCard } from './components/SurfaceMasksCard';
import { HeatRiskScoreCard } from './components/HeatRiskScoreCard';
import { BeforeAfterPreview } from './components/BeforeAfterPreview';
import { CoolingPlanCard } from './components/CoolingPlanCard';
import { ProjectedImpactCard } from './components/ProjectedImpactCard';
import { Footer } from './components/Footer';
import { ImpactReportModal } from './components/ImpactReportModal';
import { SavedRunsView } from './components/SavedRunsView';
import { MethodNoteView } from './components/MethodNoteView';
import { useResiliCityStore } from './store/useResiliCityStore';

export const App: React.FC = () => {
  const [isReportOpen, setIsReportOpen] = useState(false);
  const { activeTab, isUploadedPlaceholder, apiError, segmentationSource, segmentationModelInfo } = useResiliCityStore();

  return (
    <div className="min-h-dvh flex flex-col bg-[#f6f8fa] text-slate-800 font-sans">
      {/* Top Header */}
      <HeaderNav />

      {/* Main Content Workspace Container */}
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 py-6 flex flex-col gap-6">
        {activeTab === 'analysis' && (
          <>
            {/* Workspace Title & Stepper Header */}
            <SubHeader onExportClick={() => setIsReportOpen(true)} />

            {apiError && (
              <div role="alert" className="text-sm text-rose-900 bg-rose-50 border border-rose-300 rounded-xl px-4 py-3 flex items-center justify-between">
                <span>{apiError}</span>
              </div>
            )}

            {segmentationSource === 'segformer' && !apiError && (
              <div role="status" className="text-sm text-teal-900 bg-emerald-50 border border-emerald-300 rounded-xl px-4 py-2.5 flex items-center justify-between">
                <span>
                  <strong>Stage 1 Live:</strong> SegFormer semantic segmentation active ({segmentationModelInfo?.name ?? 'pretrained-b0'}). Surface area percentages directly feed Stage 2 satellite heat regression.
                </span>
                <span className="text-[11px] font-mono text-emerald-700 bg-white/80 px-2 py-0.5 rounded border border-emerald-200">
                  {segmentationModelInfo?.device ? `device: ${segmentationModelInfo.device}` : 'live'}
                </span>
              </div>
            )}

            {isUploadedPlaceholder && !apiError && (
              <div role="status" className="text-sm text-amber-900 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3">
                Demo data: segmentation, scores and the after image for uploaded photos are placeholders until the
                model backend is connected.
              </div>
            )}

            {/* Row 1: Site Context & Surface Masks */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
              <SiteContextCard />
              <SurfaceMasksCard />
            </div>

            {/* Row 2: Heat Risk Score */}
            <HeatRiskScoreCard />

            {/* Row 3: Before / After Intervention Preview */}
            <BeforeAfterPreview />

            {/* Row 4: Cooling Plan & Projected Impact */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-stretch">
              <CoolingPlanCard />
              <ProjectedImpactCard />
            </div>
          </>
        )}

        {activeTab === 'saved_runs' && (
          <SavedRunsView onExportClick={() => setIsReportOpen(true)} />
        )}

        {activeTab === 'method_note' && (
          <MethodNoteView />
        )}

        {/* Footer */}
        <Footer />
      </main>

      {/* Export Report Modal */}
      <ImpactReportModal
        isOpen={isReportOpen}
        onClose={() => setIsReportOpen(false)}
      />
    </div>
  );
};

export default App;
