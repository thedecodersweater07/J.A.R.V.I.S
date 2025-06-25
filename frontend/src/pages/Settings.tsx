import React, { useState } from 'react';
import { ToggleSwitch } from '../components';

interface SettingsState {
  darkMode: boolean;
  notifications: boolean;
  autoStart: boolean;
}

const Settings: React.FC = () => {
  const [settings, setSettings] = useState<SettingsState>({
    darkMode: true,
    notifications: true,
    autoStart: false,
  });

  const handleSettingChange = (key: keyof SettingsState) => {
    setSettings(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const formatLabel = (key: string): string => {
    return key.replace(/([A-Z])/g, ' $1').trim()
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-lg p-6 max-w-md">
      <h3 className="text-xl font-semibold text-white mb-6">Settings</h3>
      <div className="space-y-4">
        {Object.entries(settings).map(([key, value]) => (
          <div key={key} className="flex items-center justify-between">
            <label className="text-slate-300">
              {formatLabel(key)}
            </label>
            <ToggleSwitch
              checked={value}
              onChange={() => handleSettingChange(key as keyof SettingsState)}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default Settings;