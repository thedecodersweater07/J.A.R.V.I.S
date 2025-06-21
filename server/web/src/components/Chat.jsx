import React, { useState, useEffect, useRef } from 'react';
import './Chat.css';

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [isConnected, setIsConnected] = useState(false);
    const [ws, setWs] = useState(null);
    const [toast, setToast] = useState(null);
    const [isTyping, setIsTyping] = useState(false);
    const messageContainerRef = useRef(null);

    useEffect(() => {
        const websocket = new WebSocket(`ws://${window.location.host}/ws`);

        websocket.onopen = () => {
            setIsConnected(true);
            addMessage('system', 'Connected to JARVIS');
        };

        websocket.onclose = () => {
            setIsConnected(false);
            addMessage('system', 'Disconnected from JARVIS');
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'typing') {
                setIsTyping(true);
                setTimeout(() => setIsTyping(false), 1200);
            } else {
                setIsTyping(false);
                addMessage(data.type, data.content);
            }
        };

        setWs(websocket);

        return () => {
            websocket.close();
        };
    }, []);

    const addMessage = (type, content) => {
        setMessages(prev => [...prev, { type, content, timestamp: new Date() }]);
        setTimeout(() => {
            if (messageContainerRef.current) {
                messageContainerRef.current.scrollTop = messageContainerRef.current.scrollHeight;
            }
        }, 100);
    };

    const showToast = (msg) => {
        setToast(msg);
        setTimeout(() => setToast(null), 2500);
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!inputValue.trim()) return;
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            showToast('Cannot send: Not connected to JARVIS');
            return;
        }
        ws.send(JSON.stringify({
            type: 'user',
            content: inputValue
        }));
        setInputValue('');
    };

    return (
        <div className="chat-interface" aria-label="AI chat interface">
            <div className="status-bar">
                <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
                    {isConnected ? 'Connected' : 'Disconnected'}
                </div>
            </div>

            <div className="message-container" ref={messageContainerRef} id="messageContainer" tabIndex={0} aria-live="polite">
                {messages.length === 0 && (
                    <div className="message system empty-state">Start a conversation with JARVIS!</div>
                )}
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.type}`} tabIndex={0} aria-label={`${msg.type} message`}>
                        <div className="message-content">{msg.content}</div>
                        <div className="message-timestamp">
                            {msg.timestamp.toLocaleTimeString()}
                        </div>
                    </div>
                ))}
                {isTyping && (
                    <div className="message assistant typing-indicator" aria-live="polite">
                        <span className="typing-dot"></span>
                        <span className="typing-dot"></span>
                        <span className="typing-dot"></span>
                    </div>
                )}
            </div>

            <form onSubmit={handleSubmit} className="input-form" autoComplete="off" role="search">
                <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder={isConnected ? "Type your message..." : "Connecting to JARVIS..."}
                    autoFocus
                    aria-label="Type your message"
                />
                <button type="submit" aria-label="Send message">
                    Send
                </button>
            </form>
            {toast && <div className="chat-toast" role="alert">{toast}</div>}
        </div>
    );
};

export default Chat;
