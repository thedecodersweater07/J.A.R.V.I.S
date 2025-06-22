import React, { useState, useCallback, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import './Chat.css';

const Chat = ({ clientId }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const { socket, isConnected, error, sendMessage } = useWebSocket(clientId);

  // Koppel WebSocket event voor inkomende berichten
  useEffect(() => {
    if (!socket) return;
    const handler = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'chat') {
          setMessages((prev) => [...prev, data.payload]);
        }
      } catch (e) {
        // ignore
      }
    };
    socket.addEventListener('message', handler);
    return () => socket.removeEventListener('message', handler);
  }, [socket]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    sendMessage('chat', {
      message: inputValue,
      timestamp: new Date().toISOString(),
      sender: clientId,
    });
    setInputValue('');
  };

  return (
    <div className="chat-container">
      <div className="chat-status">
        {isConnected ? (
          <span className="status-connected">Connected</span>
        ) : (
          <span className="status-disconnected">Disconnected</span>
        )}
        {error && <span className="status-error">{error}</span>}
      </div>
      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.sender === clientId ? 'sent' : 'received'}`}>
            <span className="message-sender">{msg.sender}</span>
            <p className="message-content">{msg.message}</p>
            <span className="message-time">
              {new Date(msg.timestamp).toLocaleTimeString()}
            </span>
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="chat-input">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type a message..."
          disabled={!isConnected}
        />
        <button type="submit" disabled={!isConnected}>
          Send
        </button>
      </form>
    </div>
  );
};

export default Chat;
