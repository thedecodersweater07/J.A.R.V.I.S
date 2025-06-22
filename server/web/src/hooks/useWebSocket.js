import { useState, useEffect, useCallback } from 'react';

const RECONNECT_DELAY = 3000;
const MAX_RECONNECT_ATTEMPTS = 10;

export const useWebSocket = (clientId) => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const connect = useCallback(() => {
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      setError('Kan geen verbinding maken met de server. Probeer later opnieuw.');
      return;
    }
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/${clientId}`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
        setReconnectAttempts(0);
      };

      ws.onclose = () => {
        setIsConnected(false);
        setReconnectAttempts((prev) => prev + 1);
        setTimeout(connect, RECONNECT_DELAY);
      };

      ws.onerror = (event) => {
        setError('WebSocket error occurred');
        console.error('WebSocket error:', event);
      };

      setSocket(ws);
    } catch (err) {
      setError(err.message);
      setReconnectAttempts((prev) => prev + 1);
      setTimeout(connect, RECONNECT_DELAY);
    }
  }, [clientId, reconnectAttempts]);

  useEffect(() => {
    connect();
    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, [connect]);

  const sendMessage = useCallback((type, payload) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type, payload }));
    }
  }, [socket]);

  return {
    socket,
    isConnected,
    error,
    sendMessage
  };
};
