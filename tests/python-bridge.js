// WebSocket client voor communicatie met de Python server
export class PythonBridge {
    constructor(host = '127.0.0.1', port = 8765) {
        this.host = host;
        this.port = port;
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        this.pendingMessages = new Map();
        this.messageCounter = 0;
        this.debug = true;
        
        // Callbacks
        this.onConnect = null;
        this.onDisconnect = null;
        this.onError = null;
    }
    
    debugLog(...args) {
        if (this.debug) {
            console.log('[PythonBridge]', ...args);
        }
    }
    
    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
            this.debugLog('Already connected or connecting');
            return;
        }
        
        this.debugLog(`Connecting to ws://${this.host}:${this.port}`);
        
        try {
            this.ws = new WebSocket(`ws://${this.host}:${this.port}`);
            
            this.ws.onopen = () => {
                this.debugLog('Connection established');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.reconnectDelay = 1000;
                if (this.onConnect) this.onConnect();
            };
            
            this.ws.onclose = (event) => {
                this.debugLog('Connection closed:', event);
                this.isConnected = false;
                
                if (this.onDisconnect) this.onDisconnect(event);
                
                // Attempt to reconnect if not manually closed
                if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.debugLog(`Attempting to reconnect (${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);
                    setTimeout(() => this.connect(), this.reconnectDelay);
                    this.reconnectAttempts++;
                    this.reconnectDelay *= 2; // Exponential backoff
                }
            };
            
            this.ws.onmessage = (event) => {
                this.debugLog('Received message:', event.data);
                try {
                    const response = JSON.parse(event.data);
                    
                    // Check if this is a response to a pending message
                    if (response.id && this.pendingMessages.has(response.id)) {
                        const { resolve, reject, timeout } = this.pendingMessages.get(response.id);
                        clearTimeout(timeout);
                        this.pendingMessages.delete(response.id);
                        
                        if (response.error) {
                            this.debugLog('Error in response:', response.error);
                            reject(new Error(response.error));
                        } else {
                            this.debugLog('Processing successful response');
                            resolve(response);
                        }
                    }
                } catch (error) {
                    this.debugLog('Error parsing message:', error);
                    if (this.onError) this.onError(error);
                }
            };
            
            this.ws.onerror = (error) => {
                this.debugLog('WebSocket error:', error);
                if (this.onError) this.onError(error);
            };
            
        } catch (error) {
            this.debugLog('Connection error:', error);
            if (this.onError) this.onError(error);
        }
    }
    
    disconnect() {
        this.debugLog('Disconnecting...');
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
    }
    
    async sendMessage(message, callback, timeout = 30000) {
        if (!this.isConnected) {
            this.debugLog('Not connected, attempting to connect...');
            this.connect();
            await new Promise(resolve => setTimeout(resolve, 1000));
            if (!this.isConnected) {
                throw new Error('Failed to connect to server');
            }
        }
        
        const messageId = ++this.messageCounter;
        const messageWithId = { ...message, id: messageId };
        
        this.debugLog('Sending message:', messageWithId);
        
        return new Promise((resolve, reject) => {
            // Set up timeout
            const timeoutId = setTimeout(() => {
                this.debugLog('Message timeout:', messageId);
                this.pendingMessages.delete(messageId);
                reject(new Error('Message timeout'));
            }, timeout);
            
            // Store the promise handlers
            this.pendingMessages.set(messageId, {
                resolve: (response) => {
                    if (callback) callback(null, response);
                    resolve(response);
                },
                reject: (error) => {
                    if (callback) callback(error);
                    reject(error);
                },
                timeout: timeoutId
            });
            
            try {
                this.ws.send(JSON.stringify(messageWithId));
            } catch (error) {
                this.debugLog('Error sending message:', error);
                clearTimeout(timeoutId);
                this.pendingMessages.delete(messageId);
                if (callback) callback(error);
                reject(error);
            }
        });
    }
    
    async executeCode(code) {
        return new Promise((resolve, reject) => {
            this.sendMessage({
                type: 'execute',
                code: code
            }, (error, result) => {
                if (error) {
                    reject(error);
                } else {
                    resolve(result);
                }
            });
        });
    }
    
    async evaluateExpression(expression) {
        return new Promise((resolve, reject) => {
            this.sendMessage({
                type: 'evaluate',
                expression: expression
            }, (error, result) => {
                if (error) {
                    reject(error);
                } else {
                    resolve(result);
                }
            });
        });
    }
}

// Maak een singleton instance
const pythonBridge = new PythonBridge();

// Exporteer de PythonBridge klasse en een standaard instantie
export { PythonBridge };
export default pythonBridge;
