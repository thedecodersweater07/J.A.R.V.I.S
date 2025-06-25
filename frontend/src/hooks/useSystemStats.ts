import { useState, useEffect } from 'react';

interface SystemStats {
  cpu: number;
  memory: number;
  network: number;
  uptime: string;
}

const useSystemStats = () => {
  const [systemStats, setSystemStats] = useState<SystemStats>({
    cpu: 0,
    memory: 0,
    network: 0,
    uptime: '0h 0m 0s'
  });

  useEffect(() => {
    // Simulate system stats updates
    const interval = setInterval(() => {
      setSystemStats({
        cpu: Math.min(100, Math.max(0, systemStats.cpu + (Math.random() * 10 - 5))),
        memory: Math.min(100, Math.max(0, systemStats.memory + (Math.random() * 5 - 2.5))),
        network: Math.max(0, systemStats.network + (Math.random() * 2 - 1)),
        uptime: formatUptime(systemStats.uptime)
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [systemStats]);

  const formatUptime = (currentUptime: string): string => {
    // Simple uptime formatter - in a real app, this would come from the system
    const parts = currentUptime.split(' ');
    let h = parseInt(parts[0]) || 0;
    let m = parseInt(parts[1]) || 0;
    let s = parseInt(parts[2]) || 0;
    
    s += 2; // Add 2 seconds for each interval
    if (s >= 60) {
      m += Math.floor(s / 60);
      s = s % 60;
    }
    if (m >= 60) {
      h += Math.floor(m / 60);
      m = m % 60;
    }
    
    return `${h}h ${m}m ${s}s`;
  };

  return { systemStats };
};

export default useSystemStats;
