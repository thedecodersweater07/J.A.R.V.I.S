import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle, Send, Bot, User } from 'lucide-react';
import useConnection from '../hooks/useConnection';

interface Message {
  id: string;
  text: string;
  timestamp: string;
  sender: 'user' | 'jarvis';
  status: 'sending' | 'sent' | 'error';
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Initialize WebSocket connection
  const { isConnected, sendMessage } = useConnection('ws://localhost:8000/ws/chat');

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = () => {
    if (!newMessage.trim()) return;

    const messageId = Date.now().toString();
    const userMessage: Message = {
      id: messageId,
      text: newMessage,
      timestamp: new Date().toISOString(),
      sender: 'user',
      status: 'sending'
    };

    // Optimistically add user message
    setMessages(prev => [...prev, userMessage]);
    setNewMessage('');

    // Send message to WebSocket
    const messageSent = sendMessage({
      type: 'chat_message',
      message: newMessage,
      message_id: messageId
    });

    if (!messageSent) {
      // Update message status if sending failed
      setMessages(prev => 
        prev.map(msg => 
          msg.id === messageId 
            ? { ...msg, status: 'error' } 
            : msg
        )
      );
    } else {
      // Update message status to sent
      setMessages(prev => 
        prev.map(msg => 
          msg.id === messageId 
            ? { ...msg, status: 'sent' } 
            : msg
        )
      );
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="fixed top-0 right-0 h-full w-96 bg-slate-900/80 backdrop-blur-sm border-l border-slate-700/50 flex flex-col z-10 shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-700/50">
        <div className="flex items-center">
          <Bot size={20} className="mr-2 text-cyan-400" />
          <h2 className="text-lg font-medium text-white">JARVIS Chat</h2>
        </div>
        <div className={`flex items-center px-2 py-1 rounded text-xs ${
          isConnected ? 'bg-green-900/20 text-green-400' : 'bg-red-900/20 text-red-400'
        }`}>
          <div className={`w-2 h-2 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          {isConnected ? 'Verbonden' : 'Verbinding verbroken'}
        </div>
      </div>

      {/* Messages container */}
      <div className="flex-1 overflow-y-auto mb-4 pr-2 custom-scrollbar">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-slate-400">
            <MessageCircle className="w-16 h-16 mb-4 opacity-30" />
            <p className="text-lg">Begin een gesprek met JARVIS</p>
            <p className="text-sm mt-2 text-slate-500">Stel een vraag of geef een opdracht</p>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <div 
                key={message.id} 
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex max-w-[80%] ${message.sender === 'user' ? 'flex-row-reverse' : ''}`}>
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    message.sender === 'user' 
                      ? 'bg-blue-500 ml-3' 
                      : 'bg-gradient-to-r from-cyan-500 to-blue-500 mr-3'
                  }`}>
                    {message.sender === 'user' ? (
                      <User size={16} className="text-white" />
                    ) : (
                      <Bot size={16} className="text-white" />
                    )}
                  </div>
                  <div className={`px-4 py-2 rounded-xl ${
                    message.sender === 'user'
                      ? 'bg-blue-600/80 text-white rounded-br-none'
                      : 'bg-slate-700/80 text-white rounded-bl-none'
                  }`}>
                    <p className="whitespace-pre-wrap break-words">{message.text}</p>
                    <div className="flex items-center justify-end mt-1 space-x-1">
                      <span className="text-xs opacity-60">
                        {formatTime(message.timestamp)}
                      </span>
                      {message.sender === 'user' && (
                        <span className="text-xs">
                          {message.status === 'sending' && '🕒'}
                          {message.status === 'sent' && '✓'}
                          {message.status === 'error' && '⚠️'}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
      
      {/* Message input */}
      <div className="p-3 border-t border-slate-700/50 bg-slate-900/70">
        <div className="relative">
          <div className="flex items-end bg-slate-800/70 border border-slate-700/50 rounded-lg overflow-hidden transition-colors focus-within:border-cyan-500/50">
            <textarea
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Typ een bericht..."
              rows={Math.min(3, newMessage.split('\n').length)}
              className="flex-1 bg-transparent border-0 focus:ring-0 text-white placeholder-slate-400 resize-none max-h-32 overflow-y-auto py-2.5 px-3 text-sm focus:outline-none"
              style={{ minHeight: '42px' }}
            />
            <button
              onClick={handleSendMessage}
              disabled={!newMessage.trim() || !isConnected}
              className={`p-2.5 transition-colors ${
                newMessage.trim() && isConnected
                  ? 'text-cyan-400 hover:bg-slate-700/50'
                  : 'text-slate-600 cursor-not-allowed'
              }`}
              title="Verstuur bericht"
            >
              <Send size={16} />
            </button>
          </div>
          <div className="text-xs text-slate-500 mt-1.5 px-1 text-right">
            {isConnected ? 'Druk op Enter om te verzenden' : 'Verbinden met server...'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;