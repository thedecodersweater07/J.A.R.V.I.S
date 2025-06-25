import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string;
  icon: LucideIcon;
  color: string;
  percentage?: number;
  gradient?: string;
}

const StatCard: React.FC<StatCardProps> = ({ 
  title, 
  value, 
  icon: Icon, 
  color, 
  percentage, 
  gradient 
}) => {
  return (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-lg p-4 hover:bg-slate-800/70 transition-all duration-300">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-slate-400 text-sm">{title}</p>
          <p className={`text-2xl font-bold text-${color}-400`}>{value}</p>
        </div>
        <div className={`w-12 h-12 bg-${color}-500/20 rounded-full flex items-center justify-center`}>
          <Icon className={`w-6 h-6 text-${color}-400`} />
        </div>
      </div>
      {percentage !== undefined && gradient && (
        <div className="mt-2 w-full bg-slate-700 rounded-full h-2">
          <div 
            className={`bg-gradient-to-r ${gradient} h-2 rounded-full transition-all duration-500`}
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
      )}
    </div>
  );
};

export default StatCard;