import React from 'react';

export const Footer: React.FC = () => (
  <footer className="border-t border-slate-200 mt-16">
    <div className="max-w-[1280px] mx-auto px-4 sm:px-6 py-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-2 text-xs text-slate-500">
      <span>ResiliCity, urban heat lab</span>
      <span>Costs and coefficients are indicative. Replace with the local Schedule of Rates before use.</span>
    </div>
  </footer>
);
