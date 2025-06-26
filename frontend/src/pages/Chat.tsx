import React, { useState, useEffect } from 'react';
import { MessageCircle } from 'lucide-react';
import ChatInterface from '../components/chat/ChatInterface';

const Chat: React.FC = () => {
  const [isMinimized, setIsMinimized] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 640);

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 640;
      setIsMobile(mobile);
      if (!mobile) {
        setIsMinimized(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  if (isMinimized && isMobile) {
    return (
      <div className="fixed bottom-4 right-4 z-50">
        <button
          onClick={() => setIsMinimized(false)}
          className="p-3 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-full shadow-lg hover:shadow-xl transition-all hover:scale-105"
          aria-label="Open chat"
        >
          <MessageCircle className="w-6 h-6 text-white" />
        </button>
      </div>
    );
  }

  return (
    <div>
      <ChatInterface />
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Typ een bericht..."
            className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl py-3 pl-12 pr-16 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/30 focus:border-cyan-500/50 resize-none max-h-32 min-h-[44px]"
            rows={1}
            style={{ scrollbarWidth: 'thin' }}
          />
          
          <div className="absolute right-2 top-1.5 flex space-x-1">
            {input && (
              <button
                onClick={() => dispatch(setInput(''))}
                className="p-1 text-slate-500 hover:text-slate-300 transition-colors"
                aria-label="Clear input"
              >
                <X size={18} />
              </button>
            )}
            <button
              onClick={handleSendMessage}
              disabled={!input.trim() || isTyping}
              className={`p-1.5 rounded-full transition-all ${
                input.trim() && !isTyping
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30 hover:scale-105'
                  : 'bg-slate-700 text-slate-500 cursor-not-allowed'
              }`}
              aria-label="Send message"
            >
              {isTyping ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Send size={18} />
              )}
            </button>
          </div>
        </div>
        
        <div className="mt-2 text-xs text-slate-500 text-center">
          J.A.R.V.I.S. kan fouten maken. Controleer belangrijke informatie.
        </div>
      </div>
    </div>
  );
};

export default Chat;