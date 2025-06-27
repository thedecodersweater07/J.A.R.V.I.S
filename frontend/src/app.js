// Eenvoudige versie van de app zonder JSX
const { useState, useEffect, useRef } = React;

function App() {
  const [messages, setMessages] = useState([
    { id: 1, sender: 'ai', text: 'Hallo! Ik ben JARVIS. Hoe kan ik je vandaag helpen?', timestamp: new Date() }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = () => {
    if (!inputValue.trim()) return;

    // Voeg gebruikersbericht toe
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    // Simuleer een AI-antwoord na een korte vertraging
    setTimeout(() => {
      const responses = [
        `Ik heb je bericht ontvangen: "${inputValue}"`,
        `Interessante vraag over: "${inputValue}"`,
        `Ik verwerk je verzoek over: "${inputValue}"`
      ];
      const randomResponse = responses[Math.floor(Math.random() * responses.length)];
      
      const aiMessage = {
        id: Date.now() + 1,
        text: randomResponse,
        sender: 'ai',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, aiMessage]);
      setIsTyping(false);
    }, 1000);
  };

  // Maak React elementen aan zonder JSX
  return React.createElement('div', { className: 'container' },
    // Header
    React.createElement('header', { className: 'header' },
      React.createElement('div', { className: 'logo' },
        React.createElement('h1', { className: 'title' }, 'JARVIS AI')
      )
    ),

    // Chat container
    React.createElement('main', { className: 'chat-container' },
      // Berichten gebied
      React.createElement('div', { className: 'messages-area' },
        messages.map(message =>
          React.createElement('div', {
            key: message.id,
            className: `message-container ${message.sender}-message`
          },
            React.createElement('div', { className: 'message-content' },
              message.sender === 'ai' &&
                React.createElement('div', { className: 'message-avatar' }, 'AI'),
              
              React.createElement('div', { className: 'message-bubble' },
                React.createElement('p', { className: 'message-text' }, message.text),
                React.createElement('div', { className: 'message-footer' },
                  React.createElement('span', { className: 'timestamp' },
                    message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                  )
                )
              ),
              
              message.sender === 'user' &&
                React.createElement('div', { className: 'message-avatar user' }, 'JIJ')
            )
          )
        ),
        
        isTyping &&
          React.createElement('div', { className: 'message-container ai-message' },
            React.createElement('div', { className: 'message-content' },
              React.createElement('div', { className: 'message-avatar' }, 'AI'),
              React.createElement('div', { className: 'typing-indicator' },
                'JARVIS is aan het typen...'
              )
            )
          ),
          
        React.createElement('div', { ref: messagesEndRef })
      ),
      
      // Invoerveld
      React.createElement('div', { className: 'input-form' },
        React.createElement('div', { className: 'input-container' },
          React.createElement('input', {
            type: 'text',
            value: inputValue,
            onChange: (e) => setInputValue(e.target.value),
            onKeyDown: (e) => e.key === 'Enter' && handleSend(),
            placeholder: 'Typ je bericht...',
            className: 'input',
            'aria-label': 'Typ je bericht'
          }),
          React.createElement('button', {
            onClick: handleSend,
            className: 'send-button',
            disabled: !inputValue.trim()
          }, 'Verstuur')
        )
      )
    )
  );
}

// Initialiseer de app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(React.createElement(React.StrictMode, null, React.createElement(App)));
