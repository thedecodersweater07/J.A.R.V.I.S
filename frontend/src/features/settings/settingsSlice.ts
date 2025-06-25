import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../../store/store';

export interface SettingsState {
  theme: 'light' | 'dark' | 'system';
  language: string;
  fontSize: number;
  autoStart: boolean;
  notifications: boolean;
  showTimestamps: boolean;
  developerMode: boolean;
  apiKey: string | null;
  apiEndpoint: string;
}

const initialState: SettingsState = {
  theme: 'dark',
  language: 'en-US',
  fontSize: 14,
  autoStart: false,
  notifications: true,
  showTimestamps: true,
  developerMode: false,
  apiKey: null,
  apiEndpoint: 'http://localhost:8000/api',
};

const settingsSlice = createSlice({
  name: 'settings',
  initialState,
  reducers: {
    setTheme: (state, action: PayloadAction<SettingsState['theme']>) => {
      state.theme = action.payload;
      // You might want to persist this to localStorage
      localStorage.setItem('theme', action.payload);
    },
    setLanguage: (state, action: PayloadAction<string>) => {
      state.language = action.payload;
    },
    setFontSize: (state, action: PayloadAction<number>) => {
      state.fontSize = Math.max(10, Math.min(24, action.payload));
    },
    setAutoStart: (state, action: PayloadAction<boolean>) => {
      state.autoStart = action.payload;
    },
    setNotifications: (state, action: PayloadAction<boolean>) => {
      state.notifications = action.payload;
    },
    setShowTimestamps: (state, action: PayloadAction<boolean>) => {
      state.showTimestamps = action.payload;
    },
    setDeveloperMode: (state, action: PayloadAction<boolean>) => {
      state.developerMode = action.payload;
    },
    setApiKey: (state, action: PayloadAction<string | null>) => {
      state.apiKey = action.payload;
      // You might want to encrypt this before saving
      if (action.payload) {
        localStorage.setItem('apiKey', action.payload);
      } else {
        localStorage.removeItem('apiKey');
      }
    },
    setApiEndpoint: (state, action: PayloadAction<string>) => {
      state.apiEndpoint = action.payload;
      localStorage.setItem('apiEndpoint', action.payload);
    },
    loadSettings: (state) => {
      // Load settings from localStorage
      const savedTheme = localStorage.getItem('theme');
      if (savedTheme && ['light', 'dark', 'system'].includes(savedTheme)) {
        state.theme = savedTheme as SettingsState['theme'];
      }
      
      const savedApiKey = localStorage.getItem('apiKey');
      if (savedApiKey) {
        state.apiKey = savedApiKey;
      }
      
      const savedEndpoint = localStorage.getItem('apiEndpoint');
      if (savedEndpoint) {
        state.apiEndpoint = savedEndpoint;
      }
    },
  },
});

// Selectors
export const selectTheme = (state: RootState) => state.settings.theme;
export const selectLanguage = (state: RootState) => state.settings.language;
export const selectFontSize = (state: RootState) => state.settings.fontSize;
export const selectAutoStart = (state: RootState) => state.settings.autoStart;
export const selectNotifications = (state: RootState) => state.settings.notifications;
export const selectShowTimestamps = (state: RootState) => state.settings.showTimestamps;
export const selectDeveloperMode = (state: RootState) => state.settings.developerMode;
export const selectApiKey = (state: RootState) => state.settings.apiKey;
export const selectApiEndpoint = (state: RootState) => state.settings.apiEndpoint;

// Actions
export const {
  setTheme,
  setLanguage,
  setFontSize,
  setAutoStart,
  setNotifications,
  setShowTimestamps,
  setDeveloperMode,
  setApiKey,
  setApiEndpoint,
  loadSettings,
} = settingsSlice.actions;

export default settingsSlice.reducer;
