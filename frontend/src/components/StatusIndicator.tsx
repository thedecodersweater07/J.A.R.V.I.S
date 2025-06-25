import React from 'react';

interface StatusIndicatorProps {
  label: string;
  status: string;
  color: 'green' | 'red' | 'yellow' | 'blue';
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ label, status, color }) => {
  const colorMap = {
    green: 'bg-green-500',
    red: 'bg-red-500',
    yellow: 'bg-yellow-500',
    blue: 'bg-blue-500',
  };

  return (
    <div className="flex items-center p-3 bg-slate-800/30 rounded-lg">
      <div className={`w-2 h-2 ${colorMap[color]} rounded-full mr-3`}></div>
      <div>
        <p className="text-sm text-slate-300">{label}</p>
        <p className="text-xs text-slate-400">{status}</p>
      </div>
    </div>
  );
};

export default StatusIndicator;
