import React from 'react';
import { Activity, HardDrive, Cpu, WifiHigh } from 'lucide-react';
import StatusIndicator from '../components/StatusIndicator';
import ProgressBar from '../components/ProgressBar';

interface SystemStats {
  cpu: number;
  memory: number;
  network: number;
  uptime: string;
}

interface SystemStatusProps {
  systemStats?: Partial<SystemStats>;
}

const defaultSystemStats: SystemStats = {
  cpu: 0,
  memory: 0,
  network: 0,
  uptime: '00:00:00'
};

const SystemStatus: React.FC<SystemStatusProps> = ({ 
  systemStats = defaultSystemStats 
}) => {
  const stats = { ...defaultSystemStats, ...systemStats };

  const performanceMetrics = [
    {
      label: 'CPU Usage',
      value: stats.cpu,
      color: 'blue',
      gradient: 'from-blue-500 to-cyan-500',
      icon: Cpu
    },
    {
      label: 'Memory Usage',
      value: stats.memory,
      color: 'green',
      gradient: 'from-green-500 to-emerald-500',
      icon: HardDrive
    },
    {
      label: 'Network',
      value: stats.network,
      color: 'purple',
      gradient: 'from-purple-500 to-indigo-500',
      icon: WifiHigh
    }
  ];

  return (
    <div className="space-y-6">
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-white mb-4">System Performance</h3>
        <div className="space-y-4">
          {performanceMetrics.map((metric, index) => (
            <div key={index} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className={`p-2 rounded-lg bg-${metric.color}-500/20`}>
                    <metric.icon className={`w-5 h-5 text-${metric.color}-400`} />
                  </div>
                  <span className="ml-3 text-sm text-slate-400">{metric.label}</span>
                </div>
                <span className={`text-sm font-medium text-${metric.color}-400`}>
                  {metric.value}{metric.label === 'Network' ? ' MB/s' : '%'}
                </span>
              </div>
              <ProgressBar
                value={metric.value}
                gradient={metric.gradient}
                max={metric.label === 'Network' ? 1000 : 100}
              />
            </div>
          ))}
        </div>
      </div>

      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-white mb-4">Additional Metrics</h3>
        <div className="grid grid-cols-1 gap-4">
          <div className="flex items-center p-4 bg-slate-700/50 rounded-lg">
            <div className="p-2 rounded-lg bg-orange-500/20">
              <Activity className="w-5 h-5 text-orange-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm text-slate-400">System Uptime</p>
              <p className="text-xl font-semibold text-white">{stats.uptime}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemStatus;