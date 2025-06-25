import React, { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import App from './App';
import './index.css';

// Service Worker Registration
const registerServiceWorker = async () => {
  if ('serviceWorker' in navigator) {
    try {
      await navigator.serviceWorker.register('/service-worker.js', { scope: '/' });
      console.log('Service Worker registered');
    } catch (error) {
      console.error('Service Worker registration failed:', error);
    }
  }
};

// Check if the app is installed
const isAppInstalled = () => {
  return (
    window.matchMedia('(display-mode: standalone)').matches ||
    (window.navigator as any).standalone ||
    document.referrer.includes('android-app://')
  );
};

// Initialize service worker when DOM is loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', registerServiceWorker);
} else {
  registerServiceWorker();
}

// Create root element
const container = document.getElementById('root');
if (!container) throw new Error('Failed to find the root element');

// Create root
const root = createRoot(container);

// Render app
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </BrowserRouter>
  </React.StrictMode>
);