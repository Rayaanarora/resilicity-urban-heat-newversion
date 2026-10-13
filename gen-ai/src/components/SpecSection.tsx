import React from 'react';
import { ImplementationPlanCard } from './ImplementationPlanCard';
import { StreetCrossSection } from './StreetCrossSection';
import { SectionHeading } from './ui';

export const SpecSection: React.FC = () => (
  <section className="flex flex-col gap-5">
    <SectionHeading
      id="specification"
      number="03"
      title="Specification"
      description="The deliverable: a dimensioned section and a bill of quantities with costs, phasing and permits."
    />
    <StreetCrossSection />
    <ImplementationPlanCard />
  </section>
);
