import React from 'react';
import { useSystemHealth } from '../hooks/useSystemHealth';
import './StatusPanel.css';

const StatusPanel = () => {
  const { health, error } = useSystemHealth();

  return (
    <div className="status-panel">
      <h2>System Status</h2>
      
      <div className="status-grid">
        <div className="status-item">
          <span className="status-label">Status:</span>
          <span className={`status-value ${health.status}`}>
            {health.status || 'Unknown'}
          </span>
        </div>

        <div className="status-item">
          <span className="status-label">Version:</span>
          <span className="status-value">
            {health.version || 'Unknown'}
          </span>
        </div>

        <div className="status-item">
          <span className="status-label">Last Update:</span>
          <span className="status-value">
            {health.timestamp ? new Date(health.timestamp).toLocaleString() : 'Never'}
          </span>
        </div>

        {error && (
          <div className="status-error">
            <span className="error-icon">⚠️</span>
            <span className="error-message">{error}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default StatusPanel;