import React from 'react';
import { Activity, Wifi, WifiOff } from 'lucide-react';
import { StatCard, StatusIndicator } from '../components';

interface SystemStats {
  cpu: number;
  memory: number;
  network: number;
  uptime: string;
}

interface DashboardProps {
  systemStats: SystemStats;
  isConnected: boolean;
}

const Dashboard: React.FC<Partial<DashboardProps>> = ({ 
  systemStats = {
    cpu: 0,
    memory: 0,
    network: 0,
    uptime: '00:00:00'
  }, 
  isConnected = false 
}) => {
  const statCards = [
    {
      title: 'CPU Usage',
      value: `${systemStats.cpu}%`,
      icon: Activity,
      color: 'cyan',
      percentage: systemStats.cpu,
      gradient: 'from-cyan-500 to-blue-500'
    },
    {
      title: 'Memory',
      value: `${systemStats.memory}%`,
      icon: Activity,
      color: 'green',
      percentage: systemStats.memory,
      gradient: 'from-green-500 to-emerald-500'
    },
    {
      title: 'Network',
      value: `${systemStats.network} MB/s`,
      icon: isConnected ? Wifi : WifiOff,
      color: isConnected ? 'purple' : 'red',
      gradient: 'from-purple-500 to-indigo-500'
    },
    {
      title: 'Uptime',
      value: systemStats.uptime,
      icon: Activity,
      color: 'orange',
      gradient: 'from-orange-500 to-red-500'
    }
  ];

  const statusItems: Array<{
    label: string;
    status: string;
    color: 'green' | 'red' | 'yellow' | 'blue';
  }> = [
    {
      label: 'WebSocket',
      status: isConnected ? 'Connected' : 'Disconnected',
      color: isConnected ? 'green' : 'red'
    },
    {
      label: 'Services',
      status: 'Active',
      color: 'green'
    },
    {
      label: 'Health',
      status: 'Good',
      color: 'yellow'
    }
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {statCards.map((card, index) => (
          <StatCard key={index} {...card} />
        ))}
      </div>

      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-white mb-4">System Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {statusItems.map((item, index) => (
            <StatusIndicator key={index} {...item} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;