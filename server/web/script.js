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

// Initialize particles
createParticles();

// WebSocket connection
const ws = new WebSocket('ws://' + window.location.host + '/ws');
const messageContainer = document.getElementById('messageContainer');
const connectionStatus = document.getElementById('connectionStatus');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');

ws.onopen = () => {
    connectionStatus.innerHTML = '<span class="power-indicator online"></span>Connected';
    connectionStatus.className = 'status-value connected';
    addMessage('system', 'J.A.R.V.I.S online. All systems operational.');
};

ws.onclose = () => {
    connectionStatus.innerHTML = '<span class="power-indicator offline"></span>Disconnected';
    connectionStatus.className = 'status-value disconnected';
    addMessage('system', 'Connection lost. Attempting to reconnect...');
};

ws.onerror = () => {
    connectionStatus.innerHTML = '<span class="power-indicator offline"></span>Error';
    connectionStatus.className = 'status-value disconnected';
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    addMessage(message.type, message.content);
};

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
        ws.send(JSON.stringify({
            type: 'user',
            content: message
        }));
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

// Add initial system message
setTimeout(() => {
    addMessage('system', 'Backup systems initialized. Ready for input.');
}, 1000);

// Update response time periodically
setInterval(() => {
    const responseTime = document.getElementById('responseTime');
    const time = (Math.random() * 0.5 + 0.1).toFixed(2);
    responseTime.innerHTML = `<span class="power-indicator online"></span>${time}s`;
}, 3000);