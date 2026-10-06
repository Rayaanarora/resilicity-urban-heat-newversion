import React, { useRef } from 'react';
import { HelpCircle, MapPin, Upload } from 'lucide-react';
import { useResiliCityStore } from '../store/useResiliCityStore';

export const SiteContextCard: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { rawImageUrl, uploadCustomImage } = useResiliCityStore();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      uploadCustomImage(e.target.files[0]);
    }
  };

  return (
    <div className="rc-card p-5 flex flex-col justify-between gap-4 h-full">
      <div className="flex items-center justify-between">
        <div>
          <span className="rc-card-header-label">SOURCE IMAGE</span>
          <h3 className="text-lg font-bold text-slate-900 font-sans">Site context</h3>
        </div>
        <HelpCircle className="w-4 h-4 text-slate-400 cursor-pointer hover:text-slate-600" />
      </div>

      <div className="relative rounded-xl overflow-hidden border border-slate-200 bg-slate-900 group aspect-[16/10]">
        <img
          src={rawImageUrl || '/samples/urban_street_before.png'}
          alt="Site context"
          className="w-full h-full object-cover"
        />

        <div className="absolute top-3 left-3 bg-white/90 backdrop-blur-md border border-slate-200/80 px-3 py-1 rounded-lg text-xs font-semibold text-slate-800 flex items-center gap-1.5 shadow-sm">
          <MapPin className="w-3.5 h-3.5 text-slate-600" />
          <span>North Loop / sample</span>
        </div>

        <button
          onClick={() => fileInputRef.current?.click()}
          className="absolute top-3 right-3 bg-slate-900/80 hover:bg-slate-900 text-white p-2 rounded-lg backdrop-blur border border-slate-700 transition-colors shadow"
          title="Upload new image"
        >
          <Upload className="w-4 h-4 text-emerald-400" />
        </button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
      />

      <div
        onClick={() => fileInputRef.current?.click()}
        className="border-2 border-dashed border-slate-200 hover:border-emerald-500/60 bg-emerald-50/30 p-4 rounded-xl text-center cursor-pointer transition-all flex flex-col items-center justify-center gap-1"
      >
        <div className="flex items-center gap-2 text-emerald-700 font-semibold text-xs font-sans">
          <Upload className="w-4 h-4 text-emerald-600" />
          <span>Drop a new site image</span>
        </div>
        <span className="text-[11px] text-slate-400 font-mono">
          JPG, PNG, or WebP · max 10 MB
        </span>
      </div>

      <div className="flex items-center justify-between text-xs font-mono text-slate-400 border-t border-slate-100 pt-3">
        <span>north-loop-courtyard.jpg</span>
        <span>2048 × 1365</span>
      </div>
    </div>
  );
};
