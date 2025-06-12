// WebSocket client voor communicatie met de Python server
export class PythonBridge {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.messageQueue = [];
        this.callbacks = new Map();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        this.port = 18865;
        
        this.connect();
    }
    
    connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            return;
        }
        
        this.ws = new WebSocket(`ws://localhost:${this.port}`);
        
        this.ws.onopen = () => {
            console.log('Connected to Python WebSocket server');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            
            // Process any queued messages
            while (this.messageQueue.length > 0) {
                const { message, callback } = this.messageQueue.shift();
                this.sendMessage(message, callback);
            }
        };
        
        this.ws.onclose = () => {
            console.log('Disconnected from Python WebSocket server');
            this.isConnected = false;
            
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                setTimeout(() => {
                    this.reconnectAttempts++;
                    this.connect();
                }, this.reconnectDelay);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.ws.onmessage = (event) => {
            try {
                const response = JSON.parse(event.data);
                
                if (response.id && this.callbacks.has(response.id)) {
                    const callback = this.callbacks.get(response.id);
                    this.callbacks.delete(response.id);
                    callback(response.error, response.result);
                }
            } catch (error) {
                console.error('Error processing message:', error);
            }
        };
    }
    
    sendMessage(message, callback) {
        if (!this.isConnected) {
            this.messageQueue.push({ message, callback });
            return;
        }
        
        const id = Math.random().toString(36).substr(2, 9);
        message.id = id;
        
        if (callback) {
            this.callbacks.set(id, callback);
        }
        
        this.ws.send(JSON.stringify(message));
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
