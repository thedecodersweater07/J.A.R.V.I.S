const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const isDev = require('electron-is-dev');

let mainWindow;

function createWindow() {
  // Maak het browser venster.
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true,
    },
    frame: false,
    titleBarStyle: 'hidden',
    titleBarOverlay: {
      color: '#1e293b',
      symbolColor: '#ffffff',
      height: 30
    },
    icon: path.join(__dirname, '../assets/icon.ico')
  });

  // Laad de index.html van de app.
  mainWindow.loadURL(
    isDev
      ? 'http://localhost:3000'
      : `file://${path.join(__dirname, '../dist/index.html')}`
  );

  // Open de DevTools in development mode.
  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  // Wordt aangeroepen wanneer het venster wordt gesloten.
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Deze methode wordt aangeroepen wanneer Electron klaar is met initialiseren
app.whenReady().then(createWindow);

// Sluit de app als alle vensters gesloten zijn, behalve op macOS
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  // Op macOS is het gebruikelijk om opnieuw een venster te maken als er op het dock-icoon wordt geklikt
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// IPC handlers
ipcMain.on('minimize-window', () => {
  mainWindow.minimize();
});

ipcMain.on('maximize-window', () => {
  if (mainWindow.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow.maximize();
  }
});

ipcMain.on('close-window', () => {
  mainWindow.close();
});
