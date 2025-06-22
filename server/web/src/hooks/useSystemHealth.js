import { useState, useEffect } from 'react';

const HEALTH_CHECK_INTERVAL = 30000; // 30 seconds

export const useSystemHealth = () => {
  const [health, setHealth] = useState({
    status: 'unknown',
    version: null,
    timestamp: null,
  });
  const [error, setError] = useState(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch('/api/health');
        if (!response.ok) {
          throw new Error(`Health check failed: ${response.statusText}`);
        }
        const data = await response.json();
        setHealth(data);
        setError(null);
      } catch (err) {
        setError(err.message);
      }
    };

    // Initial check
    checkHealth();

    // Set up periodic health checks
    const interval = setInterval(checkHealth, HEALTH_CHECK_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  return { health, error };
}; 