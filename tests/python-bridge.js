// WebSocket client voor communicatie met de Python server
class PythonBridge {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.messageQueue = [];
        this.callbacks = new Map();
        this.heartbeatInterval = null;
        this.heartbeatTimeout = null;
        this.heartbeatIntervalMs = 30000; // 30 seconden
        this.heartbeatTimeoutMs = 5000;   // 5 seconden timeout
        
        // Status elementen
        this.statusElement = document.getElementById('statusIndicator');
        this.statusDot = this.statusElement ? this.statusElement.querySelector('.pulse-dot') : null;
        this.statusText = this.statusElement || { textContent: '' };
        
        // Event callbacks
        this.onConnect = () => this.updateStatus('connected');
        this.onDisconnect = () => this.updateStatus('disconnected');
        this.onReconnectAttempt = () => this.updateStatus('connecting');
        this.onMaxReconnectAttempts = () => this.updateStatus('error');
        this.onError = () => this.updateStatus('error');
        
        this.connect();
    }
    
    // Maak verbinding met de WebSocket server
    connect() {
        try {
            // Sluit bestaande verbinding indien aanwezig
            if (this.socket) {
                this.socket.close();
            }
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            // Haal het poortnummer uit de URL of gebruik de standaard poort 18866
            const wsPort = window.location.port ? `:${window.location.port}`.replace('18085', '18866') : ':18866';
            const wsUrl = `${protocol}//${window.location.hostname}${wsPort}`;
            
            console.log(`Verbinding maken met ${wsUrl}...`);
            this.socket = new WebSocket(wsUrl);
            this.updateStatus('connecting');
            
            this.socket.onopen = () => {
                console.log('Verbonden met Python WebSocket server');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                
                // Start heartbeat
                this.startHeartbeat();
                
                // Verwerk eventuele wachtrijberichten
                this.processQueue();
                
                // Roep connect callback aan indien gedefinieerd
                if (typeof this.onConnect === 'function') {
                    this.onConnect();
                }
            };
            
            this.socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    // Verwerk heartbeat antwoord
                    if (data.type === 'pong') {
                        this.handlePong();
                        return;
                    }
                    
                    // Verwerk reguliere berichten
                    if (data.requestId && this.callbacks.has(data.requestId)) {
                        const { resolve, reject } = this.callbacks.get(data.requestId);
                        this.callbacks.delete(data.requestId);
                        
                        if (data.error) {
                            reject(new Error(data.error));
                        } else {
                            resolve(data);
                        }
                    }
                } catch (e) {
                    console.error('Fout bij verwerken bericht:', e);
                    if (this.onError) {
                        this.onError(e);
                    }
                }
            };
            
            this.socket.onclose = (event) => {
                this.updateStatus('disconnected');
                console.log('WebSocket verbinding gesloten', event.code, event.reason);
                this.isConnected = false;
                this.stopHeartbeat();
                
                // Roep disconnect callback aan indien gedefinieerd
                if (typeof this.onDisconnect === 'function') {
                    this.onDisconnect(event);
                }
                
                // Probeer opnieuw verbinding te maken tenzij het een bewuste sluiting was
                if (event.code !== 1000) { // 1000 = normale sluiting
                    this.attemptReconnect();
                }
            };
            
            this.socket.onerror = (error) => {
                this.updateStatus('error');
                console.error('WebSocket fout:', error);
                this.stopHeartbeat();
                
                if (this.onError) {
                    this.onError(error);
                }
                
                // Sluit de verbinding om een reconnect te forceren
                if (this.socket) {
                    this.socket.close();
                }
            };
        } catch (error) {
            console.error('Fout bij opzetten WebSocket verbinding:', error);
            this.attemptReconnect();
        }
    }
    
    // Start het heartbeat mechanisme
    startHeartbeat() {
        // Stop eventuele bestaande heartbeat
        this.stopHeartbeat();
        
        console.log('Start heartbeat interval');
        
        // Stuur periodiek een heartbeat
        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected) {
                this.sendHeartbeat();
            }
        }, this.heartbeatIntervalMs);
        
        // Stuur direct een eerste heartbeat
        if (this.isConnected) {
            this.sendHeartbeat();
        }
    }
    
    // Stop het heartbeat mechanisme
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
        
        if (this.heartbeatTimeout) {
            clearTimeout(this.heartbeatTimeout);
            this.heartbeatTimeout = null;
        }
    }
    
    // Verstuur een heartbeat naar de server
    sendHeartbeat() {
        if (!this.isConnected) return;
        
        // Stel een timeout in voor het heartbeat antwoord
        this.heartbeatTimeout = setTimeout(() => {
            console.warn('Geen heartbeat antwoord ontvangen, verbinding opnieuw opzetten...');
            if (this.socket) {
                this.socket.close();
            }
        }, this.heartbeatTimeoutMs);
        
        // Verstuur de heartbeat
        try {
            this.socket.send(JSON.stringify({
                type: 'ping',
                timestamp: Date.now()
            }));
        } catch (error) {
            console.error('Fout bij versturen heartbeat:', error);
            this.stopHeartbeat();
        }
    }
    
    // Verwerk een binnenkomend pong bericht
    handlePong() {
        if (this.heartbeatTimeout) {
            clearTimeout(this.heartbeatTimeout);
            this.heartbeatTimeout = null;
        }
    }
    
    // Probeer opnieuw verbinding te maken
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(
                this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 
                30000
            );
            
            const message = `Poging ${this.reconnectAttempts}/${this.maxReconnectAttempts} om opnieuw te verbinden over ${delay}ms...`;
            console.log(message);
            
            // Roep de reconnect callback aan indien gedefinieerd
            if (typeof this.onReconnectAttempt === 'function') {
                this.onReconnectAttempt(this.reconnectAttempts, this.maxReconnectAttempts, delay);
            }
            
            // Plan een nieuwe verbindingspoging
            setTimeout(() => {
                if (!this.isConnected) {
                    this.connect();
                }
            }, delay);
        } else {
            const message = 'Maximaal aantal herpogingen bereikt';
            console.error(message);
            
            // Roep de max reconnect callback aan indien gedefinieerd
            if (typeof this.onMaxReconnectAttempts === 'function') {
                this.onMaxReconnectAttempts();
            }
        }
    }
    
    // Verstuur een bericht naar de server
    sendMessage(message, timeoutMs = 30000) {
        return new Promise((resolve, reject) => {
            if (!message || typeof message !== 'object') {
                reject(new Error('Ongeldig bericht formaat'));
                return;
            }
            
            const requestId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            const messageWithId = { 
                ...message, 
                requestId,
                timestamp: new Date().toISOString()
            };
            
            // Timeout voor dit specifieke bericht
            let timeout = null;
            
            // Callback voor het verwerken van het antwoord
            const callback = (error, response) => {
                // Verwijder de timeout
                if (timeout) {
                    clearTimeout(timeout);
                    timeout = null;
                }
                
                // Verwijder de callback uit de map
                this.callbacks.delete(requestId);
                
                // Roep de juiste callback aan
                if (error) {
                    reject(error);
                } else {
                    resolve(response);
                }
            };
            
            // Voeg de callback toe aan de map
            this.callbacks.set(requestId, { 
                resolve: (response) => callback(null, response),
                reject: (error) => callback(error || new Error('Onbekende fout'))
            });
            
            // Stel een timeout in voor dit bericht
            timeout = setTimeout(() => {
                if (this.callbacks.has(requestId)) {
                    this.callbacks.delete(requestId);
                    reject(new Error(`Timeout: Geen antwoord ontvangen na ${timeoutMs}ms`));
                }
            }, timeoutMs);
            
            // Verstuur het bericht of zet het in de wachtrij
            try {
                if (this.isConnected && this.socket && this.socket.readyState === WebSocket.OPEN) {
                    this.socket.send(JSON.stringify(messageWithId));
                } else {
                    console.log('Bericht in wachtrij gezet (geen verbinding)');
                    this.messageQueue.push({ 
                        message: messageWithId, 
                        resolve: (response) => callback(null, response),
                        reject: (error) => callback(error || new Error('Verbindingsfout'))
                    });
                }
            } catch (error) {
                console.error('Fout bij verzenden bericht:', error);
                this.callbacks.delete(requestId);
                reject(new Error(`Kon bericht niet verzenden: ${error.message}`));
            }
        });
    }
    
    // Verwerk de wachtrij met berichten
    processQueue() {
        const failedMessages = [];
        
        while (this.messageQueue.length > 0 && this.isConnected) {
            const { message, resolve, reject } = this.messageQueue.shift();
            
            try {
                if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                    this.socket.send(JSON.stringify(message));
                } else {
                    // Bewaar het bericht voor een volgende poging
                    failedMessages.push({ message, resolve, reject });
                }
            } catch (error) {
                console.error('Fout bij verwerken wachtrijbericht:', error);
                reject(new Error(`Fout bij verzenden: ${error.message}`));
            }
        }
        
        // Zet gefaalde berichten terug in de wachtrij
        if (failedMessages.length > 0) {
            this.messageQueue.unshift(...failedMessages);
            
            // Probeer opnieuw verbinding te maken als dat nodig is
            if (this.isConnected) {
                console.log('Opnieuw proberen verbinding te maken...');
                this.connect();
            }
        }
    }
    
    // Verwerk tekst met de Python backend
    async processText(text, context = {}) {
        if (!text || typeof text !== 'string') {
            throw new Error('Ongeldige tekstinvoer');
        }
        
        try {
            const response = await this.sendMessage({
                type: 'process_text',
                text: text,
                context: context,
                timestamp: new Date().toISOString()
            });
            
            // Controleer of het antwoord geldig is
            if (!response || typeof response !== 'object') {
                throw new Error('Ongeldig antwoord ontvangen van de server');
            }
            
            return response;
        } catch (error) {
            console.error('Fout bij verwerken tekst:', error);
            throw new Error(`Kon tekst niet verwerken: ${error.message}`);
        }
    }
    
    // Controleer of de verbinding actief is
    isConnected() {
        return this.isConnected && 
               this.socket && 
               this.socket.readyState === WebSocket.OPEN;
    }
    
    // Sluit de WebSocket verbinding
    disconnect() {
        if (this.socket) {
            try {
                this.socket.close(1000, 'Gebruiker heeft de verbinding gesloten');
            } catch (error) {
                console.error('Fout bij sluiten verbinding:', error);
            } finally {
                this.isConnected = false;
                this.socket = null;
                this.stopHeartbeat();
            }
        }
    }
    
    // Vernieuw de verbinding
    reconnect() {
        this.disconnect();
        this.reconnectAttempts = 0;
        return new Promise((resolve, reject) => {
            const checkConnection = () => {
                if (this.isConnected()) {
                    resolve(true);
                } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                    reject(new Error('Kon geen verbinding maken met de server'));
                } else {
                    setTimeout(checkConnection, 100);
                }
            };
            
            this.connect();
            checkConnection();
        });
    }
}

// Maak een singleton instance
const pythonBridge = new PythonBridge();

// Exporteer de PythonBridge klasse en een standaard instantie
export { PythonBridge };
export default pythonBridge;
