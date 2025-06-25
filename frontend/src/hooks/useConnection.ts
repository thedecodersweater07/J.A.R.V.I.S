import { useState, useEffect } from 'react';

const useConnection = (url: string) => {
  const [isConnected, setIsConnected] = useState(false);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      // Attempt to reconnect after a delay
      setTimeout(() => {
        setSocket(new WebSocket(url));
      }, 3000);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
    
    setSocket(ws);
    
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [url]);
  
  const sendMessage = (message: any) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
      return true;
    }
    return false;
  };
  
  return { isConnected, sendMessage };
};

export default useConnection;
