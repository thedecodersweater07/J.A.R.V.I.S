// --- Particle Background ---
function createParticles() {
    const bgAnimation = document.getElementById('bgAnimation');
    const particleCount = 50;
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 6 + 's';
        particle.style.animationDuration = (6 + Math.random() * 4) + 's';
        bgAnimation.appendChild(particle);
    }
}
createParticles();

// --- DOM Elements ---
const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/ws';
let ws;
const messageContainer = document.getElementById('messageContainer');
const connectionStatus = document.getElementById('connectionStatus');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const loadingIndicator = document.getElementById('loadingIndicator');
const themeToggle = document.getElementById('themeToggle');
const micButton = document.getElementById('micButton');

// --- WebSocket Logic ---
function connectWebSocket() {
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
        setStatus('Connected', 'connected');
        addMessage('system', 'J.A.R.V.I.S online. All systems operational.');
    };
    ws.onclose = () => {
        setStatus('Disconnected', 'disconnected');
        addMessage('system', 'Connection lost. Attempting to reconnect...');
        setTimeout(connectWebSocket, 2000);
    };
    ws.onerror = () => {
        setStatus('Error', 'disconnected');
    };
    ws.onmessage = (event) => {
        hideLoading();
        try {
            const message = JSON.parse(event.data);
            addMessage(message.type, message.content);
        } catch (e) {
            addMessage('system', 'Received invalid message.');
        }
    };
}
connectWebSocket();

function setStatus(text, statusClass) {
    connectionStatus.innerHTML = `<span class="power-indicator ${statusClass === 'connected' ? 'online' : 'offline'}"></span>${text}`;
    connectionStatus.className = `status-value ${statusClass}`;
}

// --- Message Handling ---
function addMessage(type, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = content;
    messageContainer.appendChild(messageDiv);
    messageContainer.scrollTop = messageContainer.scrollHeight;
}

function sendMessage() {
    const message = userInput.value.trim();
    if (message && ws.readyState === WebSocket.OPEN) {
        showLoading();
        ws.send(JSON.stringify({ type: 'user', content: message }));
        addMessage('user', message);
        userInput.value = '';
    }
}

sendButton.onclick = sendMessage;
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// --- Loading Indicator ---
function showLoading() {
    loadingIndicator.style.display = 'flex';
}
function hideLoading() {
    loadingIndicator.style.display = 'none';
}
hideLoading();

// --- Theme Toggle ---
themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-theme');
    themeToggle.textContent = document.body.classList.contains('dark-theme') ? '☀️' : '🌙';
});

// --- Voice Input (Stub) ---
micButton.addEventListener('click', () => {
    addMessage('system', 'Voice input is not yet implemented.');
});

// --- Initial System Message ---
setTimeout(() => {
    addMessage('system', 'Backup systems initialized. Ready for input.');
}, 1000);

// --- Response Time Simulation ---
setInterval(() => {
    const responseTime = document.getElementById('responseTime');
    const time = (Math.random() * 0.5 + 0.1).toFixed(2);
    responseTime.innerHTML = `<span class="power-indicator online"></span>${time}s`;
}, 3000);