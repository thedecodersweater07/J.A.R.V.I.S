import React, { useRef, useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import { 
  sendMessage, 
  selectMessages, 
  selectIsTyping, 
  setInput, 
  selectInput 
} from '../../features/chat/chatSlice';

interface ChatInterfaceProps {
  className?: string;
  onMinimize?: () => void;
  isMobile?: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ 
  className = '', 
  onMinimize, 
  isMobile = false 
}) => {
  const dispatch = useAppDispatch();
  const messages = useAppSelector(selectMessages);
  const isTyping = useAppSelector(selectIsTyping);
  const input = useAppSelector(selectInput);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    const content = input.trim();
    if (!content) return;
    
    dispatch(sendMessage({ 
      content, 
      conversationId: 'default' 
    }));
    dispatch(setInput(''));
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    dispatch(setInput(e.target.value));
  };

  return (
    <div 
      className={`flex flex-col h-full bg-slate-900/90 backdrop-blur-lg border border-slate-700/50 rounded-xl overflow-hidden ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-3 bg-slate-800/50 border-b border-slate-700/50">
        <div className="flex items-center">
          <div className="w-8 h-8 rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 flex items-center justify-center">
            <span className="text-white font-bold">J</span>
          </div>
          <h2 className="ml-2 text-lg font-semibold text-white">JARVIS</h2>
        </div>
        {onMinimize && (
          <button 
            onClick={onMinimize}
            className="p-1 text-slate-400 hover:text-white transition-colors"
            aria-label="Minimize chat"
          >
            <span className="text-xl">×</span>
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-8">
            <div className="w-16 h-16 rounded-full bg-slate-800/50 flex items-center justify-center mb-4">
              <span className="text-2xl">🤖</span>
            </div>
            <h3 className="text-xl font-semibold text-slate-200 mb-2">Hallo, ik ben JARVIS</h3>
            <p className="text-slate-400 max-w-md">
              Hoe kan ik je vandaag helpen? Stel me een vraag of geef me een opdracht.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <div 
                key={message.id}
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div 
                  className={`max-w-[80%] px-4 py-2 rounded-lg ${
                    message.sender === 'user'
                      ? 'bg-blue-600 text-white rounded-br-none'
                      : 'bg-slate-800 text-white rounded-bl-none'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="flex justify-start">
                <div className="bg-slate-800 text-white px-4 py-2 rounded-lg rounded-bl-none">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" />
                    <div 
                      className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" 
                      style={{ animationDelay: '0.2s' }} 
                    />
                    <div 
                      className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" 
                      style={{ animationDelay: '0.4s' }} 
                    />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-3 border-t border-slate-700/50 bg-slate-900/50">
        <div className="relative">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg py-2.5 px-3 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-cyan-500/50 focus:border-transparent"
            disabled={isTyping}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isTyping}
            className={`absolute right-2 top-1/2 transform -translate-y-1/2 p-1 rounded-full transition-colors ${
              input.trim() && !isTyping
                ? 'text-cyan-400 hover:text-cyan-300'
                : 'text-slate-500 cursor-not-allowed'
            }`}
            aria-label="Send message"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={isTyping ? 'animate-pulse' : ''}
            >
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
