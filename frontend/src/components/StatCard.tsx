import React from 'react';
import { IconType } from 'react-icons';

interface StatCardProps {
  title: string;
  value: string;
  icon: IconType;
  color: string;
  percentage?: number;
  gradient: string;
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  icon: Icon,
  gradient,
  percentage
}) => {
  return (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-lg p-4">
      <div className="flex justify-between items-start">
        <div>
          <p className="text-slate-400 text-sm">{title}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
        </div>
        <div className={`p-2 rounded-lg bg-gradient-to-br ${gradient}`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
      </div>
      {percentage !== undefined && (
        <div className="mt-4">
          <div className="w-full bg-slate-700 rounded-full h-1.5">
            <div 
              className={`h-full rounded-full ${gradient}`}
              style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default StatCard;
