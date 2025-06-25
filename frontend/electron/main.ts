import { app, BrowserWindow, ipcMain, shell, WebContents, IpcMainEvent } from 'electron';
import * as path from 'path';
import { spawn, ChildProcess, execSync } from 'child_process';
import * as isDev from 'electron-is-dev';
import * as fs from 'fs';

// Global references
let mainWindow: BrowserWindow | null = null;
let jarvisProcess: ChildProcess | null = null;
let requestId = 0;
const pendingRequests = new Map<number, (response: any) => void>();

// Path to the app directory
const appDir = path.resolve(__dirname, '..');
const isPackaged = app.isPackaged || process.env.NODE_ENV === 'production';
const resourcesPath = isPackaged ? process.resourcesPath : appDir;

// Python executable path
const getPythonPath = (): string => {
  if (process.platform === 'win32') {
    const pythonExe = path.join(resourcesPath, 'python', 'python.exe');
    if (fs.existsSync(pythonExe)) {
      return pythonExe;
    }
    // Fallback to system Python in development
    return 'python';
  } else {
    // For macOS/Linux
    const pythonExe = path.join(resourcesPath, 'python', 'bin', 'python3');
    if (fs.existsSync(pythonExe)) {
      return pythonExe;
    }
    return 'python3';
  }
};

/**
 * Create the browser window
 */
function createWindow(): Promise<void> {
  return new Promise((resolve, reject) => {
    mainWindow = new BrowserWindow({
      width: 1280,
      height: 800,
      minWidth: 1024,
      minHeight: 768,
      frame: false,
      titleBarStyle: 'hidden',
      webPreferences: {
        nodeIntegration: true,
        contextIsolation: false,
        webSecurity: !isDev,
        preload: path.join(__dirname, 'preload.js'),
      },
      icon: path.join(__dirname, '../public/icon.png'),
    });

    // Load the index.html file
    const startUrl = isDev 
      ? 'http://localhost:3000'
      : `file://${path.join(appDir, 'dist/index.html')}`;

    mainWindow.loadURL(startUrl)
      .then(() => {
        if (isDev) {
          mainWindow?.webContents.openDevTools();
        }
        
        // Handle window controls
        ipcMain.on('minimize-window', () => {
          if (mainWindow) mainWindow.minimize();
        });

        ipcMain.on('maximize-window', () => {
          if (!mainWindow) return;
          
          if (mainWindow.isMaximized()) {
            mainWindow.unmaximize();
          } else {
            mainWindow.maximize();
          }
        });

        ipcMain.on('close-window', () => {
          if (mainWindow) mainWindow.close();
        });

        mainWindow?.on('closed', () => {
          if (jarvisProcess) {
            jarvisProcess.kill();
            jarvisProcess = null;
          }
          mainWindow = null;
        });

        // Start Jarvis service
        return startJarvisService()
          .then(() => {
            console.log('Jarvis service started successfully');
            
            // Test the connection
            return sendToJarvis({
              type: 'ping',
              data: {}
            });
          })
          .then((testResponse) => {
            console.log('Jarvis service test response:', testResponse);
            resolve();
          });
      })
      .catch((error) => {
        console.error('Failed to load app:', error);
        reject(error);
      });
  });
}

/**
 * Start the Jarvis service
 * @returns {Promise<void>}
 */
function startJarvisService(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (jarvisProcess) {
      console.log('Jarvis service already running');
      return resolve();
    }

    try {
      const pythonPath = getPythonPath();
      const scriptPath = path.join(__dirname, 'jarvis_service.py');
      
      console.log(`Starting Jarvis service with Python: ${pythonPath}`);
      console.log(`Script path: ${scriptPath}`);
      
      jarvisProcess = spawn(pythonPath, [scriptPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        shell: true,
        env: {
          ...process.env,
          PYTHONPATH: path.join(appDir, '..'), // Add project root to PYTHONPATH
          PYTHONIOENCODING: 'utf-8',
          PYTHONUNBUFFERED: '1'
        }
      });

      if (!jarvisProcess || !jarvisProcess.pid) {
        throw new Error('Failed to spawn Jarvis process');
      }

      let buffer = '';
      
      // Handle stdout data
      if (jarvisProcess.stdout) {
        const onData = (data: Buffer) => {
          try {
            const text = data.toString();
            buffer += text;
            
            // Process complete messages (separated by double newlines)
            const messages = buffer.split('\n\n');
            buffer = messages.pop() || ''; // Keep incomplete message in buffer
            
            for (const msg of messages) {
              if (msg.trim()) {
                try {
                  const response = JSON.parse(msg);
                  console.log('Jarvis response:', response);
                  
                  // Handle the response if it has a request_id
                  if (response.request_id !== undefined) {
                    const callback = pendingRequests.get(response.request_id);
                    if (callback) {
                      callback(response);
                      pendingRequests.delete(response.request_id);
                    }
                  }
                } catch (e) {
                  console.error('Error parsing Jarvis response:', e, 'Raw:', msg);
                }
              }
            }
          } catch (error) {
            console.error('Error processing stdout data:', error);
          }
        };
        
        jarvisProcess.stdout.on('data', onData);
      }

      // Handle stderr
      if (jarvisProcess.stderr) {
        jarvisProcess.stderr.on('data', (data: Buffer) => {
          console.error(`Jarvis Error: ${data.toString()}`);
        });
      }

      // Handle process error
      jarvisProcess.on('error', (error: Error) => {
        console.error('Failed to start Jarvis process:', error);
        jarvisProcess = null;
        reject(error);
      });

      // Handle process exit
      jarvisProcess.on('close', (code: number | null) => {
        console.log(`Jarvis process exited with code ${code}`);
        jarvisProcess = null;
        if (code !== 0) {
          console.error(`Jarvis process exited with code ${code}`);
          // Don't reject here as it might be a graceful shutdown
        }
      });

      // Consider the service started successfully if no error after a short delay
      setTimeout(resolve, 1000);
    } catch (error) {
      console.error('Failed to start Jarvis service:', error);
      jarvisProcess = null;
      reject(error);
    }
  });
}

/**
 * Send a message to the Jarvis service
 * @param message The message to send
 * @returns A promise that resolves with the response
 */
function sendToJarvis(message: any): Promise<any> {
  return new Promise((resolve, reject) => {
    if (!jarvisProcess) {
      return reject(new Error('Jarvis service is not running'));
    }
    
    const request_id = requestId++;
    const messageWithId = { ...message, request_id };
    
    // Store the callback
    pendingRequests.set(request_id, (response: any) => {
      if (response.type === 'error') {
        reject(new Error(response.error || 'Unknown error from Jarvis service'));
      } else {
        resolve(response);
      }
    });
    
    // Send the message
    const messageStr = JSON.stringify(messageWithId) + '\n\n';
    
    // Create a promise to handle the write operation
    const writePromise = new Promise<void>((writeResolve, writeReject) => {
      if (!jarvisProcess || !jarvisProcess.stdin) {
        writeReject(new Error('Jarvis process or stdin is not available'));
        return;
      }
      
      const onDrain = () => {
        jarvisProcess!.stdin?.removeListener('error', onError);
        writeResolve();
      };
      
      const onError = (error: Error) => {
        jarvisProcess!.stdin?.removeListener('drain', onDrain);
        writeReject(error);
      };
      
      jarvisProcess.stdin.once('drain', onDrain);
      jarvisProcess.stdin.once('error', onError);
      
      const writeResult = jarvisProcess.stdin.write(messageStr, (error) => {
        jarvisProcess!.stdin?.removeListener('drain', onDrain);
        jarvisProcess!.stdin?.removeListener('error', onError);
        
        if (error) {
          writeReject(error);
        } else {
          writeResolve();
        }
      });
      
      // If write returns false, we need to wait for the 'drain' event
      if (writeResult === false) {
        // The 'drain' event will resolve the promise
      } else {
        // The write completed immediately, clean up listeners
        jarvisProcess.stdin?.removeListener('drain', onDrain);
        jarvisProcess.stdin?.removeListener('error', onError);
        writeResolve();
      }
    });
    
    // Handle the write operation
    writePromise
      .then(() => {
        // Write completed successfully
      })
      .catch((error) => {
        pendingRequests.delete(request_id);
        reject(error);
      });
    
    // Set a timeout for the request
    setTimeout(() => {
      if (pendingRequests.has(request_id)) {
        pendingRequests.delete(request_id);
        reject(new Error('Request to Jarvis service timed out'));
      }
    }, 30000); // 30 second timeout
  });
}

// Expose IPC handlers for the renderer process
ipcMain.handle('jarvis:processInput', async (event, text: string) => {
  try {
    const response = await sendToJarvis({
      type: 'process_input',
      data: { text }
    });
    return response.data || { success: false, error: 'No response data' };
  } catch (error) {
    console.error('Error processing input:', error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error',
      response: "I'm sorry, but I'm having trouble processing your request."
    };
  }
});

// When Electron has finished initialization
app.whenReady().then(() => {
  createWindow().catch(console.error);
  
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow().catch(console.error);
    }
  });
});

// Quit when all windows are closed
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Clean up on quit
app.on('will-quit', () => {
  if (jarvisProcess) {
    jarvisProcess.kill();
    jarvisProcess = null;
  }
  
  // Clear any pending requests
  pendingRequests.clear();
});

// Handle external links
app.on('web-contents-created', (event: Electron.Event, contents: WebContents) => {
  if (contents.getType() === 'window') {
    contents.setWindowOpenHandler(({ url }) => {
      // Allow localhost and file protocols
      if (url.startsWith('http://localhost:') || url.startsWith('file://')) {
        return { action: 'allow' };
      }
      
      // Open external links in default browser
      try {
        const result = shell.openExternal(url);
        if (result && typeof result === 'object' && 'catch' in result) {
          (result as Promise<void>).catch((error: Error) => {
            console.error('Failed to open external URL:', error);
          });
        }
      } catch (error) {
        console.error('Error opening external URL:', error);
      }
      return { action: 'deny' };
    });
  }
});
