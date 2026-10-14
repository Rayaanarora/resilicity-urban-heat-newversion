import type React from 'react';

/** Turn backend stage codes like 'PLANNING RESILIENCE STRATEGY' into sentence case. */
export const sentenceCase = (s: string) => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();

export const zoneLabel = (zone?: string | null) =>
  zone === 'left_sidewalk' ? 'Left side' : zone === 'right_sidewalk' ? 'Right side' : zone === 'roadway' ? 'Carriageway' : zone ? sentenceCase(zone.replace(/_/g, ' ')) : 'Site';

/** Frame style that keeps an exact aspect ratio but never exceeds maxHeight (width shrinks instead). */
export const framedAspect = (aspect: number, maxHeight: number): React.CSSProperties => ({
  aspectRatio: `${aspect}`,
  width: `min(100%, ${Math.round(maxHeight * aspect)}px)`,
  marginInline: 'auto',
});

export const inrLakh = (n: number) => (n >= 100_000 ? `₹${(n / 100_000).toFixed(1)} L` : `₹${Math.round(n).toLocaleString('en-IN')}`);
