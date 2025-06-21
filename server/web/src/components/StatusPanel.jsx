import React from 'react';
import './StatusPanel.css';

const StatusPanel = ({ systemHealth, responseTime, connectionStatus }) => {
    // Determine status classes
    const getHealthStatus = (health) => {
        return health === 'Optimal' ? 'online' : 'offline';
    };

    const getConnectionStatus = (status) => {
        return status ? 'connected' : 'disconnected';
    };

    return (
        <div className="status-panel" role="region" aria-label="System Status">
            <div className="status-grid">
                <div className="status-item">
                    <div className="status-label">Connection Status</div>
                    <div className={`status-value ${getConnectionStatus(connectionStatus)}`}>
                        <span 
                            className={`power-indicator ${connectionStatus ? 'online' : 'offline'}`}
                            aria-hidden="true"
                        ></span>
                        {connectionStatus ? 'Connected' : 'Disconnected'}
                    </div>
                </div>
                
                <div className="status-item">
                    <div className="status-label">System Health</div>
                    <div className={`status-value ${getHealthStatus(systemHealth)}`}>
                        <span 
                            className={`power-indicator ${getHealthStatus(systemHealth)}`}
                            aria-hidden="true"
                        ></span>
                        {systemHealth}
                    </div>
                </div>
                
                <div className="status-item">
                    <div className="status-label">Response Time</div>
                    <div className="status-value online">
                        <span 
                            className="power-indicator online"
                            aria-hidden="true"
                        ></span>
                        {responseTime}s
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StatusPanel;