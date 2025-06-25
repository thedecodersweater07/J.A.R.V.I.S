// Extend the Window interface to include electron API
declare global {
  interface Window {
    electron?: {
      window: {
        minimize: () => void;
        maximize: () => void;
        close: () => void;
        isMaximized: () => Promise<boolean>;
      };
      on: (event: 'maximized' | 'unmaximized', callback: () => void) => void;
      off: (event: 'maximized' | 'unmaximized', callback: () => void) => void;
    };
  }
}

export {}; // This file needs to be a module
