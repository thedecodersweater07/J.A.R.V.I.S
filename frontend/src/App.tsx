import React, { useEffect, useState } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { useTheme } from '@/hooks/useTheme';
import { TitleBar, Sidebar } from '@/components';
import Dashboard from '@/pages/Dashboard';
import Chat from '@/pages/Chat';
import Settings from '@/pages/Settings';
import SystemStatus from '@/pages/SystemStatus';
import './App.css';

function App() {
  const { theme } = useTheme();
  const [isConnected, setIsConnected] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  
  // Get active tab from current path
  const activeTab = location.pathname === '/' ? 'dashboard' : location.pathname.substring(1);

  // Handle tab change
  const handleTabChange = (tab: string) => {
    navigate(tab === 'dashboard' ? '/' : `/${tab}`);
  };

  // Simulate connection status for now
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsConnected(true);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={`app ${theme}`}>
      <TitleBar />
      <div className="app-container">
        <Sidebar activeTab={activeTab} onTabChange={handleTabChange} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/system" element={<SystemStatus />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
      
      {/* Connection status indicator */}
      <div 
        className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}
        title={isConnected ? 'Verbonden' : 'Niet verbonden'}
      />
    </div>
  );
}

export default App;