import React, { useState, useEffect } from 'react';
import StatusPanel from './components/StatusPanel';
import Chat from './components/Chat';
import './App.css';

const App = () => {
    const [systemStatus, setSystemStatus] = useState({
        systemHealth: 'Initializing...',
        responseTime: '0.00',
        connectionStatus: false
    });

    // Create background particles
    useEffect(() => {
        const createParticles = () => {
            const bgAnimation = document.getElementById('bgAnimation');
            if (!bgAnimation) return;
            
            // Clear existing particles
            bgAnimation.innerHTML = '';
            
            const particleCount = 30;
            
            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 6 + 's';
                particle.style.animationDuration = (8 + Math.random() * 4) + 's';
                bgAnimation.appendChild(particle);
            }
        };

        // Create particles after component mounts
        const timer = setTimeout(createParticles, 100);
        return () => clearTimeout(timer);
    }, []);

    // Health check system
    useEffect(() => {
        const checkHealth = async () => {
            try {
                const response = await fetch('/health');
                if (response.ok) {
                    const data = await response.json();
                    setSystemStatus(prev => ({
                        ...prev,
                        systemHealth: data.status === 'healthy' ? 'Optimal' : 'Degraded',
                        connectionStatus: data.status === 'healthy'
                    }));
                } else {
                    throw new Error('Health check failed');
                }
            } catch (error) {
                console.warn('Health check failed:', error);
                setSystemStatus(prev => ({
                    ...prev,
                    systemHealth: 'Error',
                    connectionStatus: false
                }));
            }
        };

        // Initial health check
        checkHealth();
        
        // Periodic health checks
        const interval = setInterval(checkHealth, 30000);
        return () => clearInterval(interval);
    }, []);

    // Response time simulation
    useEffect(() => {
        const updateResponseTime = () => {
            setSystemStatus(prev => ({
                ...prev,
                responseTime: (Math.random() * 0.4 + 0.1).toFixed(2)
            }));
        };

        const interval = setInterval(updateResponseTime, 3000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="app">
            <div className="bg-animation" id="bgAnimation"></div>
            <div className="circuit-overlay"></div>

            <div className="container">
                <header className="header">
                    <h1>J.A.R.V.I.S</h1>
                    <div className="subtitle">Just A Rather Very Intelligent System</div>
                </header>

                <StatusPanel 
                    systemHealth={systemStatus.systemHealth}
                    responseTime={systemStatus.responseTime}
                    connectionStatus={systemStatus.connectionStatus}
                />
                
                <Chat />
            </div>
        </div>
    );
};

export default App;