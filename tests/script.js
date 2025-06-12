// Importeer de Python bridge
import pythonBridge from './python-bridge.js';

// Globale variabelen
let isProcessing = false;
let conversationHistory = [];
let messageCount = 0;

// Configureer de Python bridge
function setupPythonBridge() {
    // Stel event handlers in
    pythonBridge.onConnect = () => {
        console.log('Verbonden met Python server');
        updateConnectionStatus('connected');
        
        // Verstuur een testbericht om de verbinding te verifiëren
        pythonBridge.sendMessage({
            type: 'handshake',
            client: 'J.A.R.V.I.S. Web Client',
            version: '1.0.0',
            timestamp: new Date().toISOString()
        })
        .then(response => {
            console.log('Handshake voltooid:', response);
            
            // Toon welkomstbericht als dit de eerste keer is
            if (conversationHistory.length === 0) {
                addMessage('assistant', 'Hallo! Ik ben J.A.R.V.I.S., je persoonlijke assistent. Hoe kan ik je vandaag helpen?');
            }
        })
        .catch(error => {
            console.error('Handshake mislukt:', error);
            showError('Kon geen verbinding maken met de server. Probeer het later opnieuw.');
        });
    };
    
    pythonBridge.onDisconnect = (event) => {
        console.log('Verbinding met Python server verbroken', event);
        updateConnectionStatus('disconnected');
        showError('Verbinding met de server verbroken. Er wordt geprobeerd opnieuw verbinding te maken...');
    };
    
    pythonBridge.onReconnectAttempt = (attempt, maxAttempts, delay) => {
        console.log(`Poging ${attempt}/${maxAttempts} om opnieuw te verbinden over ${delay}ms...`);
        updateConnectionStatus('reconnecting', attempt, maxAttempts);
    };
    
    pythonBridge.onMaxReconnectAttempts = () => {
        console.error('Maximaal aantal herpogingen bereikt');
        updateConnectionStatus('failed');
        showError('Kon geen verbinding maken met de server. Ververs de pagina om het opnieuw te proberen.');
    };
    
    pythonBridge.onError = (error) => {
        console.error('Python bridge fout:', error);
        showError(`Verbindingsfout: ${error.message}`);
    };
    
    // Start de verbinding
    console.log('Initialiseer Python bridge...');
    pythonBridge.connect();
}

// Controleer de verbindingsstatus
function checkConnection() {
    if (!pythonBridge.isConnected()) {
        pythonBridge.reconnect();
    }
}

// Functie om een bericht naar de AI te sturen
async function sendToAI(message) {
    if (!message || !message.trim() || isProcessing) return;
    
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
        // Bereid het verzoek voor met contextuele informatie
        const request = {
            type: 'chat',
            timestamp: new Date().toISOString(),
            context: {
                user_agent: navigator.userAgent,
                language: navigator.language,
                timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                previous_messages: conversationHistory
                    .slice(-4) // Neem de laatste 4 berichten als context
                    .map(msg => ({
                        role: msg.role,
                        content: msg.content,
                        timestamp: new Date().toISOString()
                    }))
            },
            message: message
        };
        
        console.log('Verstuur verzoek:', request);
        
        // Verstuur het bericht naar de Python bridge
        const response = await pythonBridge.sendMessage(request);
        
        console.log('Ontvangen antwoord:', response);
        
        // Verwijder de 'typing' indicator
        if (typingIndicator) typingIndicator.remove();
        
        // Verwerk het antwoord
        if (response && (response.result || response.text)) {
            const responseText = response.result || response.text;
            
            // Voeg AI antwoord toe aan de chat
            addMessage('assistant', responseText);
            
            // Voeg toe aan gespreksgeschiedenis
            conversationHistory.push({ 
                role: 'assistant', 
                content: responseText,
                timestamp: new Date().toISOString()
            });
            
            // Voeg eventuele extra acties toe (bijv. knoppen, afbeeldingen, etc.)
            if (response.actions && Array.isArray(response.actions)) {
                handleResponseActions(response.actions);
            }
        } else {
            throw new Error('Ongeldig antwoord ontvangen van de server');
        }
    } catch (error) {
        console.error('Fout bij versturen bericht:', error);
        
        // Verwijder de 'typing' indicator
        if (typingIndicator) typingIndicator.remove();
        
        // Toon een passende foutmelding
        let errorMessage = 'Er is een fout opgetreden bij het verwerken van je bericht. ';
        
        if (error.message.includes('timeout') || error.message.includes('time-out')) {
            errorMessage += 'De server reageert niet op tijd. Probeer het later opnieuw.';
        } else if (error.message.includes('connection') || error.message.includes('verbinding')) {
            errorMessage += 'Er is een probleem met de verbinding. Controleer je internetverbinding.';
        } else {
            errorMessage += 'Probeer het later opnieuw.';
        }
        
        addMessage('error', errorMessage);
        
        // Probeer opnieuw verbinding te maken
        if (pythonBridge.reconnect) {
            pythonBridge.reconnect();
        }
    } finally {
        // Zorg ervoor dat de invoer weer actief is
        isProcessing = false;
        
        // Update de UI
        updateUI();
    }
}

// Functie om extra acties uit het antwoord te verwerken
function handleResponseActions(actions) {
    if (!Array.isArray(actions)) return;
    
    actions.forEach(action => {
        if (action.type === 'show_image' && action.url) {
            // Toon een afbeelding in de chat
            const img = document.createElement('img');
            img.src = action.url;
            img.alt = action.alt || 'Afbeelding';
            img.style.maxWidth = '100%';
            img.style.borderRadius = '8px';
            img.style.marginTop = '8px';
            
            const messageElement = document.querySelector('.message:last-child .message-content');
            if (messageElement) {
                messageElement.appendChild(document.createElement('br'));
                messageElement.appendChild(img);
            }
        }
        // Voeg hier meer actietypen toe indien nodig
    });
}

// Functie om een bericht aan de chat toe te voegen
function addMessage(sender, text, isTyping = false) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return null;
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${sender}${isTyping ? ' typing' : ''}`;
    
    // Voeg avatar toe
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    
    if (sender === 'assistant') {
        avatar.innerHTML = '🤖'; // Of gebruik een afbeelding: <img src="path/to/avatar.png" alt="J.A.R.V.I.S.">
    } else if (sender === 'user') {
        avatar.innerHTML = '👤'; // Of gebruik een gebruikersafbeelding
    } else {
        avatar.innerHTML = '⚠️'; // Voor foutmeldingen
    }
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    if (isTyping) {
        // Voeg een leuke typing indicator toe met CSS-animaties
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = `
            <span></span>
            <span></span>
            <span></span>
        `;
        messageContent.appendChild(typingIndicator);
    } else {
        // Voeg de berichttekst toe
        messageContent.textContent = text;
    }
    
    // Voeg timestamp toe
    const timestamp = document.createElement('div');
    timestamp.className = 'message-timestamp';
    timestamp.textContent = formatTime(new Date());
    
    // Voeg alle elementen samen
    messageElement.appendChild(avatar);
    messageElement.appendChild(messageContent);
    messageElement.appendChild(timestamp);
    
    // Voeg het bericht toe aan de chat
    chatMessages.appendChild(messageElement);
    
    // Scroll naar beneden
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Voeg een subtiele animatie toe voor nieuwe berichten
    messageElement.style.opacity = '0';
    messageElement.style.transform = 'translateY(10px)';
    messageElement.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    
    setTimeout(() => {
        messageElement.style.opacity = '1';
        messageElement.style.transform = 'translateY(0)';
    }, 10);
    
    return messageElement;
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
function updateConnectionStatus(status, attempt = 0, maxAttempts = 0) {
    const statusElement = document.getElementById('statusIndicator');
    if (!statusElement) return;
    
    // Verwijder bestaande statusklassen
    statusElement.className = 'status-indicator';
    
    switch (status) {
        case 'connected':
            statusElement.innerHTML = `
                <div class="pulse-dot connected"></div>
                <span>Verbonden</span>
            `;
            statusElement.classList.add('status-connected');
            break;
            
        case 'reconnecting':
            statusElement.innerHTML = `
                <div class="pulse-dot reconnecting"></div>
                <span>Verbinding maken... (${attempt}/${maxAttempts})</span>
            `;
            statusElement.classList.add('status-reconnecting');
            break;
            
        case 'disconnected':
            statusElement.innerHTML = `
                <div class="pulse-dot disconnected"></div>
                <span>Verbinding verbroken</span>
            `;
            statusElement.classList.add('status-disconnected');
            break;
            
        case 'failed':
            statusElement.innerHTML = `
                <div class="pulse-dot failed"></div>
                <span>Verbinding mislukt</span>
            `;
            statusElement.classList.add('status-failed');
            break;
            
        default:
            statusElement.innerHTML = `
                <div class="pulse-dot unknown"></div>
                <span>Status onbekend</span>
            `;
            statusElement.classList.add('status-unknown');
    }
}

// Functie om een foutmelding te tonen
function showError(message) {
    console.error('Fout:', message);
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
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const clearButton = document.getElementById('clearButton');
    
    if (!messageInput || !sendButton || !clearButton) return;
    
    // Verstuur bericht op Enter (maar niet Shift+Enter)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendButton.click();
        }
    });
    
    // Update de UI wanneer de gebruiker typt
    messageInput.addEventListener('input', () => {
        updateUI();
    });
    
    // Verstuur knop
    sendButton.addEventListener('click', () => {
        const message = messageInput.value.trim();
        if (message) {
            sendToAI(message);
            messageInput.value = '';
            updateUI();
        }
    });
    
    // Wis chat knop
    clearButton.addEventListener('click', clearChat);
}

// Initialiseer de applicatie wanneer de DOM geladen is
document.addEventListener('DOMContentLoaded', () => {
    // Initialiseer de Python bridge
    setupPythonBridge();
    
    // Voeg event listeners toe
    setupEventListeners();
    
    // Update de UI
    updateUI();
    
    // Initialiseer de statistieken
    updateStats();
    
    // Controleer periodiek de verbinding
    setInterval(checkConnection, 30000);
    
    // Focus op het invoerveld
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.focus();
    }
});
