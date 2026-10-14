import React, { useEffect, useState } from 'react';
import { TopBar } from './components/TopBar';
import { PageHeader } from './components/PageHeader';
import { SiteSection } from './components/SiteSection';
import { ProposalSection } from './components/ProposalSection';
import { SpecSection } from './components/SpecSection';
import { ImpactSection } from './components/ImpactSection';
import { Footer } from './components/Footer';
import { ImpactReportModal } from './components/ImpactReportModal';
import { MethodNoteView } from './components/MethodNoteView';
import { MapPage } from './pages/MapPage';
import { ProjectsPage } from './pages/ProjectsPage';
import { ProjectPage } from './pages/ProjectPage';
import { PortfolioPage } from './pages/PortfolioPage';
import { useResiliCityStore, type NavTab } from './store/useResiliCityStore';

const HASH_TABS: NavTab[] = ['map', 'projects', 'portfolio', 'analysis', 'method_note'];

/** Keep the URL hash and the active page in sync (#map, #projects, #project/<id>, ...). */
function useHashRoute() {
  const { activeTab, activeProjectId, setActiveTab, openProject } = useResiliCityStore();
  useEffect(() => {
    // The initial hash is read when the store is created; this follows later changes (links, back button).
    const read = () => {
      const [tab, id] = window.location.hash.replace(/^#/, '').split('/');
      if (tab === 'project' && id) openProject(id);
      else if (HASH_TABS.includes(tab as NavTab)) setActiveTab(tab as NavTab);
    };
    window.addEventListener('hashchange', read);
    return () => window.removeEventListener('hashchange', read);
  }, [openProject, setActiveTab]);
  useEffect(() => {
    const next = activeTab === 'project' && activeProjectId ? `#project/${activeProjectId}` : `#${activeTab}`;
    if (window.location.hash !== next) window.history.replaceState(null, '', next);
  }, [activeTab, activeProjectId]);
}

export const App: React.FC = () => {
  const [isReportOpen, setIsReportOpen] = useState(false);
  const { activeTab, apiError } = useResiliCityStore();
  useHashRoute();

  return (
    <div className="min-h-dvh flex flex-col bg-white text-slate-800">
      <TopBar />

      <main className="flex-1 w-full max-w-[1280px] mx-auto px-4 sm:px-6 py-8 flex flex-col gap-12">
        {activeTab === 'map' && <MapPage />}
        {activeTab === 'projects' && <ProjectsPage />}
        {activeTab === 'project' && <ProjectPage />}
        {activeTab === 'portfolio' && <PortfolioPage />}

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

        {activeTab === 'method_note' && <MethodNoteView />}
      </main>

      <Footer />
      <ImpactReportModal isOpen={isReportOpen} onClose={() => setIsReportOpen(false)} />
    </div>
  );
};

export default App;
