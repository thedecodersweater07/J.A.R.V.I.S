// Importeer de Python bridge
import { PythonBridge } from './python-bridge.js';

// Globale variabelen
let isProcessing = false;
let conversationHistory = [];
let messageCount = 0;
let pythonBridge = new PythonBridge();

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const statusIndicator = document.getElementById('statusIndicator');
const connectionStatus = document.getElementById('connectionStatus');
const messageCountElement = document.getElementById('messageCount');
const latencyDisplay = document.getElementById('latency');

// Debug mode
const DEBUG = true;

function debugLog(...args) {
    if (DEBUG) {
        console.log('[DEBUG]', ...args);
    }
}

let messageCounter = 0;
let lastPingTime = null;

// Configureer de Python bridge
function setupPythonBridge() {
    debugLog('Setting up Python bridge...');
    
    // Stel event handlers in
    pythonBridge.onConnect = () => {
        debugLog('Connected to Python server');
        updateConnectionStatus('connected');
        
        // Toon welkomstbericht als dit de eerste keer is
        if (conversationHistory.length === 0) {
            addMessage('assistant', 'Hallo! Ik ben J.A.R.V.I.S., je persoonlijke assistent. Hoe kan ik je vandaag helpen?');
        }
    };
    
    pythonBridge.onDisconnect = (event) => {
        debugLog('Disconnected from Python server', event);
        updateConnectionStatus('disconnected');
        showError('Verbinding met de server verbroken. Er wordt geprobeerd opnieuw verbinding te maken...');
    };
    
    pythonBridge.onError = (error) => {
        console.error('Python bridge error:', error);
        showError(`Verbindingsfout: ${error.message}`);
    };
    
    // Start de verbinding
    debugLog('Initializing Python bridge...');
    pythonBridge.connect();
}

// Functie om een bericht naar de AI te sturen
async function sendToAI(message) {
    if (!message || !message.trim() || isProcessing) {
        debugLog('Invalid message or already processing');
        return;
    }
    
    debugLog('Sending message to AI:', message);
    
    // Markeer dat we bezig zijn met verwerken
    isProcessing = true;
    updateUI();
    
    // Voeg het bericht toe aan de chat
    addMessage('user', message);
    
    // Voeg het bericht toe aan de gespreksgeschiedenis
    conversationHistory.push({ 
        role: 'user', 
        content: message,
        timestamp: new Date().toISOString()
    });
    
    // Verhoog berichtenteller
    messageCount++;
    
    // Update statistieken
    updateStats();
    
    // Toon een 'typing' indicator
    const typingIndicator = addMessage('assistant', 'Denk na...', true);
    
    try {
        debugLog('Preparing request');
        // Bereid het verzoek voor
        const request = {
            type: 'chat',
            message: message
        };
        
        debugLog('Sending request:', request);
        // Verstuur het bericht via de Python bridge
        const response = await new Promise((resolve, reject) => {
            pythonBridge.sendMessage(request, (error, result) => {
                if (error) {
                    debugLog('Error from Python bridge:', error);
                    reject(error);
                } else {
                    debugLog('Response from Python bridge:', result);
                    resolve(result);
                }
            });
        });
        
        // Verwijder de 'typing' indicator
        if (typingIndicator) typingIndicator.remove();
        
        debugLog('Processing response:', response);
        // Voeg het antwoord toe aan de chat
        if (response && response.message) {
            addMessage('assistant', response.message);
            
            // Voeg toe aan gespreksgeschiedenis
            conversationHistory.push({ 
                role: 'assistant', 
                content: response.message,
                timestamp: new Date().toISOString()
            });
        } else {
            throw new Error('Invalid response format from server');
        }
    } catch (error) {
        console.error('Error sending message:', error);
        debugLog('Error details:', error);
        
        // Verwijder de 'typing' indicator
        if (typingIndicator) typingIndicator.remove();
        
        // Toon foutmelding
        addMessage('error', 'Er is een fout opgetreden bij het verwerken van je bericht. Probeer het opnieuw.');
    } finally {
        isProcessing = false;
        updateUI();
    }
}

// Functie om een bericht aan de chat toe te voegen
function addMessage(type, text, isTyping = false) {
    debugLog('Adding message:', { type, text, isTyping });
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = text;
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    // Scroll naar beneden
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return isTyping ? messageDiv : null;
}

// Functie om de chat te wissen
function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    // Bevestig of de gebruiker de chat echt wil wissen
    if (conversationHistory.length > 0 && !confirm('Weet je zeker dat je het gesprek wilt wissen?')) {
        return;
    }
    
    // Wis de chat UI
    chatMessages.innerHTML = '';
    
    // Wis de gespreksgeschiedenis
    conversationHistory = [];
    
    // Reset de berichtenteller
    messageCount = 0;
    updateStats();
    
    // Voeg een welkomstbericht toe
    addMessage('assistant', 'Het gesprek is gewist. Hoe kan ik je vandaag helpen?');
}

// Functie om de UI te updaten op basis van de huidige staat
function updateUI() {
    debugLog('Updating UI');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const clearButton = document.getElementById('clearButton');
    
    if (!messageInput || !sendButton || !clearButton) return;
    
    // Schakel invoer in/op basis van de verwerkingsstatus en verbindingsstatus
    const isConnected = pythonBridge.isConnected ? pythonBridge.isConnected() : false;
    const hasText = messageInput.value.trim().length > 0;
    
    messageInput.disabled = isProcessing || !isConnected;
    sendButton.disabled = isProcessing || !hasText || !isConnected;
    clearButton.disabled = isProcessing || conversationHistory.length === 0;
    
    // Update de knoptekst en stijlen
    if (isProcessing) {
        sendButton.innerHTML = '<div class="spinner"></div>';
        sendButton.title = 'Bezig met verwerken...';
    } else if (!isConnected) {
        sendButton.innerHTML = '<i class="fas fa-unlink"></i>';
        sendButton.title = 'Geen verbinding met de server';
    } else {
        sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
        sendButton.title = 'Verstuur bericht';
    }
    
    // Update de placeholder op basis van de verbindingsstatus
    messageInput.placeholder = isConnected ? 
        'Typ je bericht hier...' : 
        'Verbinden met server...';
    
    // Update de verbindingsstatus indicator
    updateConnectionStatus(isConnected ? 'connected' : 'disconnected');
}

// Functie om de verbindingsstatus in de UI bij te werken
function updateConnectionStatus(status) {
    debugLog('Updating connection status:', status);
    statusIndicator.className = `pulse-dot ${status}`;
    connectionStatus.textContent = status.charAt(0).toUpperCase() + status.slice(1);
}

// Functie om een foutmelding te tonen
function showError(message) {
    debugLog('Showing error:', message);
    addMessage('error', message);
}

// Functie om de tijd te formatteren
function formatTime(date) {
    if (!(date instanceof Date)) {
        date = new Date(date);
    }
    
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    
    return `${hours}:${minutes}`;
}

// Functie om de statistieken bij te werken
function updateStats() {
    debugLog('Updating stats');
    const statsElement = document.getElementById('stats');
    if (!statsElement) return;
    
    const now = new Date();
    const timeString = now.toLocaleTimeString('nl-NL', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false
    });
    
    const dateString = now.toLocaleDateString('nl-NL', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    
    statsElement.innerHTML = `
        <div class="stat-item">
            <i class="fas fa-comment"></i>
            <span>${messageCount} berichten</span>
        </div>
        <div class="stat-item">
            <i class="fas fa-clock"></i>
            <span>${timeString}</span>
        </div>
        <div class="stat-item">
            <i class="fas fa-calendar"></i>
            <span>${dateString}</span>
        </div>
    `;
    
    // Update de tijd elke minuut
    setTimeout(updateStats, 60000 - (now.getSeconds() * 1000 + now.getMilliseconds()));
}

// Functie om event listeners toe te voegen
function setupEventListeners() {
    debugLog('Setting up event listeners');
    
    // Zoek de chat formulier en input elementen
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    
    if (!chatForm || !messageInput || !sendButton) {
        console.error('Chat elementen niet gevonden!');
        return;
    }
    
    // Voeg submit handler toe aan het formulier
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const message = messageInput.value.trim();
        debugLog('Form submitted with message:', message);
        
        if (message) {
            sendToAI(message);
            messageInput.value = '';
        }
    });
    
    // Voeg click handler toe aan de verzend knop
    sendButton.addEventListener('click', async () => {
        const message = messageInput.value.trim();
        if (message) {
            await sendToAI(message);
            messageInput.value = '';
            messageInput.focus();
        }
    });
    
    // Voeg enter key handler toe aan het input veld
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });
    
    // Voeg clear chat handler toe
    const clearButton = document.getElementById('clearChat');
    if (clearButton) {
        clearButton.addEventListener('click', clearChat);
    }
}

function updateStatus(status, message) {
    const dot = statusIndicator.querySelector('.pulse-dot');
    const text = statusIndicator.querySelector('.status-text');
    
    dot.className = 'pulse-dot ' + status;
    text.textContent = message;
    connectionStatus.textContent = message;
}

// Initialiseer de applicatie
document.addEventListener('DOMContentLoaded', () => {
    debugLog('DOM loaded, initializing...');
    setupPythonBridge();
    setupEventListeners();
    
    // Focus op het input veld
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.focus();
    }
    
    // Initialize connection
    updateConnectionStatus('disconnected');
});

// Export for use in React component
window.WS_PORT = WS_PORT;