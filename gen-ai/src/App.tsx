import React, { useState } from 'react';
import { TopBar } from './components/TopBar';
import { PageHeader } from './components/PageHeader';
import { SiteSection } from './components/SiteSection';
import { ProposalSection } from './components/ProposalSection';
import { SpecSection } from './components/SpecSection';
import { ImpactSection } from './components/ImpactSection';
import { Footer } from './components/Footer';
import { ImpactReportModal } from './components/ImpactReportModal';
import { SavedRunsView } from './components/SavedRunsView';
import { MethodNoteView } from './components/MethodNoteView';
import { useResiliCityStore } from './store/useResiliCityStore';

export const App: React.FC = () => {
  const [isReportOpen, setIsReportOpen] = useState(false);
  const { activeTab, apiError } = useResiliCityStore();

  return (
    <div className="min-h-dvh flex flex-col bg-white text-slate-800">
      <TopBar />

      <main className="flex-1 w-full max-w-[1280px] mx-auto px-4 sm:px-6 py-8 flex flex-col gap-12">
        {activeTab === 'analysis' && (
          <>
            <PageHeader onExport={() => setIsReportOpen(true)} />

            {apiError && (
              <div role="alert" className="rc-fade-in sticky top-16 z-40 border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">
                {apiError}
              </div>
            )}

            <SiteSection />
            <ProposalSection />
            <SpecSection />
            <ImpactSection />
          </>
        )}

        {activeTab === 'saved_runs' && <SavedRunsView onExportClick={() => setIsReportOpen(true)} />}
        {activeTab === 'method_note' && <MethodNoteView />}
      </main>

      <Footer />
      <ImpactReportModal isOpen={isReportOpen} onClose={() => setIsReportOpen(false)} />
    </div>
  );
};

export default App;
