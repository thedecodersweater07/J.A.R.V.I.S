import React from 'react';
import { IconType } from 'react-icons';

interface NavigationItem {
  id: string;
  label: string;
  icon: IconType;
}

interface NavigationProps {
  items: NavigationItem[];
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const Navigation: React.FC<NavigationProps> = ({ items, activeTab, onTabChange }) => {
  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-slate-900/80 backdrop-blur-md border-t border-slate-800 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-around">
          {items.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`flex flex-col items-center py-3 px-4 text-sm font-medium w-full transition-colors ${
                  isActive ? 'text-cyan-400' : 'text-slate-400 hover:text-white'
                }`}
              >
                <Icon className={`w-6 h-6 mb-1 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                <span className="text-xs">{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
