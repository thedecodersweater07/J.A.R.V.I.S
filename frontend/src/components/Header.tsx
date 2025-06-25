import React from 'react';
import { Menu } from 'lucide-react';

interface HeaderProps {
  isConnected: boolean;
  onMenuClick?: () => void;
}

const Header: React.FC<HeaderProps> = ({ isConnected, onMenuClick }) => {
  return (
    <header className="sticky top-0 z-50 bg-slate-900/80 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <button
              onClick={onMenuClick}
              className="text-slate-400 hover:text-white p-2 rounded-md focus:outline-none focus:ring-2 focus:ring-cyan-500"
            >
              <Menu className="h-6 w-6" />
            </button>
            <div className="flex-shrink-0 ml-4">
              <h1 className="text-xl font-bold text-white">JARVIS</h1>
            </div>
          </div>
          <div className="flex items-center">
            <div className="flex items-center">
              <div className={`w-2 h-2 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-slate-300">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
