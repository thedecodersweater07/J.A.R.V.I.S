import React from 'react';

interface ProgressBarProps {
  value: number;
  max?: number;
  gradient?: string;
  className?: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 100,
  gradient = 'from-blue-500 to-cyan-500',
  className = ''
}) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));

  return (
    <div className={`w-full bg-slate-700/50 rounded-full h-1.5 overflow-hidden ${className}`}>
      <div 
        className={`h-full rounded-full bg-gradient-to-r ${gradient}`}
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
};

export default ProgressBar;
