import React from 'react';
import { FiMinus, FiMaximize2, FiX } from 'react-icons/fi';

const { ipcRenderer } = window.require('electron');

const TitleBar = () => {
  return (
    <div className="bg-gray-800 text-white flex items-center justify-between px-4 h-8 select-none">
      <div className="flex items-center">
        <img 
          src="/icon.png" 
          alt="JARVIS" 
          className="w-5 h-5 mr-2"
        />
        <span className="text-sm font-medium">JARVIS</span>
      </div>
      <div className="flex items-center space-x-2">
        <button 
          onClick={() => ipcRenderer.send('minimize-window')}
          className="p-1 rounded hover:bg-gray-700 transition-colors"
          title="Minimaliseren"
        >
          <FiMinus size={16} />
        </button>
        <button 
          onClick={() => ipcRenderer.send('maximize-window')}
          className="p-1 rounded hover:bg-gray-700 transition-colors"
          title="Maximaliseren"
        >
          <FiMaximize2 size={14} />
        </button>
        <button 
          onClick={() => ipcRenderer.send('close-window')}
          className="p-1 rounded hover:bg-red-600 transition-colors"
          title="Afsluiten"
        >
          <FiX size={16} />
        </button>
      </div>
    </div>
  );
};

export default TitleBar;
