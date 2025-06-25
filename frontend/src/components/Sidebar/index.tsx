import React from 'react';
import { NavLink } from 'react-router-dom';
import { Home, MessageSquare, Settings, Activity } from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Home },
    { id: 'chat', label: 'Chat', icon: MessageSquare },
    { id: 'system', label: 'System', icon: Activity },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <div className="w-64 h-screen bg-slate-800 text-white p-4 fixed left-0 top-0 pt-16">
      <div className="space-y-2">
        {navItems.map((item) => (
          <NavLink
            key={item.id}
            to={item.id === 'dashboard' ? '/' : `/${item.id}`}
            className={({ isActive }) =>
              `flex items-center p-3 rounded-lg transition-colors ${
                isActive ? 'bg-slate-700' : 'hover:bg-slate-700/50'
              }`
            }
            onClick={() => onTabChange(item.id)}
          >
            <item.icon className="w-5 h-5 mr-3" />
            <span>{item.label}</span>
          </NavLink>
        ))}
      </div>
    </div>
  );
};

export default Sidebar;
