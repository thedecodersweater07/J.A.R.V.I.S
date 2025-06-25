import React from 'react';
import { FiHome, FiMessageSquare, FiCpu, FiSettings } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';

type SidebarProps = {
  activeTab: string;
  onTabChange: (tab: string) => void;
};

const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange }) => {
  const navigate = useNavigate();

  const menuItems = [
    { id: 'dashboard', icon: <FiHome size={20} />, label: 'Dashboard' },
    { id: 'chat', icon: <FiMessageSquare size={20} />, label: 'Chat' },
    { id: 'system', icon: <FiCpu size={20} />, label: 'Systeem' },
    { id: 'settings', icon: <FiSettings size={20} />, label: 'Instellingen' },
  ];

  const handleClick = (id: string) => {
    onTabChange(id);
    navigate(`/${id === 'dashboard' ? '' : id}`);
  };

  return (
    <aside className="w-20 md:w-64 bg-gray-800 text-white flex flex-col h-[calc(100vh-2rem)]">
      <div className="p-4 flex items-center justify-center md:justify-start md:px-6">
        <div className="w-10 h-10 rounded-lg bg-blue-600 flex items-center justify-center">
          <FiCpu size={24} />
        </div>
        <span className="hidden md:block ml-3 text-xl font-semibold">JARVIS</span>
      </div>
      
      <nav className="flex-1 mt-8">
        <ul className="space-y-2 px-2">
          {menuItems.map((item) => (
            <li key={item.id}>
              <button
                onClick={() => handleClick(item.id)}
                className={`w-full flex items-center p-3 rounded-lg transition-colors ${
                  activeTab === item.id
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-gray-700'
                }`}
              >
                <span className="flex items-center justify-center w-8">
                  {item.icon}
                </span>
                <span className="hidden md:block ml-3">{item.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>
      
      <div className="p-4 border-t border-gray-700">
        <div className="flex items-center">
          <div className="w-8 h-8 rounded-full bg-gray-600"></div>
          <div className="hidden md:block ml-3">
            <p className="text-sm font-medium">Gebruiker</p>
            <p className="text-xs text-gray-400">Beheerder</p>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
