import React from 'react';

/* Shared building blocks: square panels, quiet labels, flat buttons. */

export const Eyebrow: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = '' }) => (
  <span className={`rc-card-header-label ${className}`}>{children}</span>
);

interface PanelProps {
  eyebrow?: string;
  title?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}

export const Panel: React.FC<PanelProps> = ({ eyebrow, title, actions, children, className = '', bodyClassName = 'p-5' }) => (
  <section className={`rc-card flex flex-col ${className}`}>
    {(eyebrow || title || actions) && (
      <header className="flex flex-wrap items-center justify-between gap-3 px-5 py-4 border-b border-slate-200">
        <div className="flex flex-col gap-0.5 min-w-0">
          {eyebrow && <Eyebrow>{eyebrow}</Eyebrow>}
          {title && <h3 className="text-base font-semibold text-slate-900 truncate">{title}</h3>}
        </div>
        {actions && <div className="flex items-center gap-2 flex-wrap">{actions}</div>}
      </header>
    )}
    <div className={`flex-1 ${bodyClassName}`}>{children}</div>
  </section>
);

type ButtonVariant = 'primary' | 'secondary' | 'ghost';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  icon?: React.ReactNode;
}

const BUTTON_STYLES: Record<ButtonVariant, string> = {
  primary: 'rc-btn-primary border border-[#0d7a5f]',
  secondary: 'bg-white border border-slate-300 text-slate-700 hover:border-[#0d7a5f] hover:text-[#0d7a5f]',
  ghost: 'border border-transparent text-slate-600 hover:text-[#0d7a5f] hover:bg-slate-50',
};

export const Button: React.FC<ButtonProps> = ({ variant = 'secondary', icon, children, className = '', ...rest }) => (
  <button
    type="button"
    {...rest}
    className={`inline-flex items-center justify-center gap-2 px-4 h-9 text-sm font-medium cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${BUTTON_STYLES[variant]} ${className}`}
  >
    {icon}
    {children}
  </button>
);

type TagTone = 'neutral' | 'green' | 'amber' | 'red';

const TAG_STYLES: Record<TagTone, string> = {
  neutral: 'bg-slate-50 border-slate-200 text-slate-600',
  green: 'bg-[#e8f3ef] border-[#b7d9cc] text-[#0a6550]',
  amber: 'bg-amber-50 border-amber-200 text-amber-800',
  red: 'bg-rose-50 border-rose-200 text-rose-800',
};

export const Tag: React.FC<{ tone?: TagTone; children: React.ReactNode; className?: string }> = ({
  tone = 'neutral',
  children,
  className = '',
}) => (
  <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 text-[11px] border ${TAG_STYLES[tone]} ${className}`}>
    {children}
  </span>
);

export const Stat: React.FC<{ label: string; value: React.ReactNode; sub?: React.ReactNode; tone?: 'default' | 'green' }> = ({
  label,
  value,
  sub,
  tone = 'default',
}) => (
  <div className="flex flex-col gap-1 border border-slate-200 bg-white px-4 py-3 transition-colors hover:border-[#b7d9cc]">
    <Eyebrow>{label}</Eyebrow>
    <div className={`text-xl font-semibold ${tone === 'green' ? 'text-[#0d7a5f]' : 'text-slate-900'}`}>{value}</div>
    {sub && <div className="text-xs text-slate-500">{sub}</div>}
  </div>
);

export const SectionHeading: React.FC<{ id: string; number: string; title: string; description?: string; actions?: React.ReactNode }> = ({
  id,
  number,
  title,
  description,
  actions,
}) => (
  <div id={id} className="scroll-mt-28 flex flex-wrap items-end justify-between gap-4 border-b border-slate-200 pb-3">
    <div className="flex items-baseline gap-4">
      <span className="text-sm text-[#0d7a5f] tabular-nums">{number}</span>
      <div>
        <h2 className="text-xl font-semibold text-slate-900">{title}</h2>
        {description && <p className="text-sm text-slate-500 mt-0.5 max-w-2xl">{description}</p>}
      </div>
    </div>
    {actions && <div className="flex items-center gap-2">{actions}</div>}
  </div>
);

export const Switch: React.FC<{ checked: boolean; onChange: () => void; label: string }> = ({ checked, onChange, label }) => (
  <button
    type="button"
    role="switch"
    aria-checked={checked}
    onClick={onChange}
    className="inline-flex items-center gap-2 text-xs text-slate-600 cursor-pointer"
  >
    <span className={`relative w-9 h-5 border transition-colors ${checked ? 'bg-[#0d7a5f] border-[#0d7a5f]' : 'bg-slate-100 border-slate-300'}`}>
      <span
        className={`absolute top-0.5 w-3.5 h-3.5 bg-white border border-slate-300 transition-transform duration-200 ${
          checked ? 'translate-x-[18px]' : 'translate-x-0.5'
        }`}
      />
    </span>
    {label}
  </button>
);
