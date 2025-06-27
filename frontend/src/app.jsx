// React is loaded via CDN
const { useState, useRef, useEffect, useCallback } = React;

// Lucide Icons are loaded via CDN
const {
  Settings, Monitor, Cpu, HardDrive, Wifi, X, Menu,
  Activity, Zap, Shield, Palette, Sun, Moon, Send,
  Paperclip, Terminal, Code, ChevronDown, ChevronUp
} = lucide.icons;
import './style.css';

// Add Inter and Fira Code fonts
const fontLink = document.createElement('link');
fontLink.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap';
fontLink.rel = 'stylesheet';
document.head.appendChild(fontLink);

const EnhancedJarvisAI = () => {
  // State management
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      text: "JARVIS AI System Online. How may I assist you today? I'm here to help with any task or question you might have.", 
      sender: 'ai', 
      timestamp: new Date(),
      status: 'delivered'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showPerformance, setShowPerformance] = useState(false);
  const [currentTheme, setCurrentTheme] = useState('neon');
  const [darkMode, setDarkMode] = useState(true);
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [autoScroll, setAutoScroll] = useState(true);
  const [soundEffects, setSoundEffects] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [inputFocused, setInputFocused] = useState(false);
  const [performance, setPerformance] = useState({
    cpu: 45, 
    gpu: 38, 
    memory: 62, 
    disk: 75,
    network: 'Good', 
    ping: 23, 
    temperature: 42, 
    powerUsage: 150
  });
  
  const messagesEndRef = useRef(null);
  const performanceInterval = useRef(null);

  // Theme configuration
  const themes = {
    neon: { 
      name: 'Neon Blue', 
      primary: '#00a8ff', 
      secondary: '#9c27b0', 
      accent: '#00e5ff',
      background: 'linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 100%)', 
      backgroundLight: '#f0f5ff',
      text: '#f0f0ff',
      textDark: '#1a1a2e',
      cardBg: 'rgba(20, 20, 40, 0.7)'
    },
    stark: { 
      name: 'Stark Gold', 
      primary: '#ffb300', 
      secondary: '#ff3d00', 
      accent: '#ff9100',
      background: 'linear-gradient(135deg, #0d1421 0%, #1a252f 100%)',
      backgroundLight: '#fff8e1',
      text: '#e6f3ff',
      textDark: '#2c2c2c',
      cardBg: 'rgba(30, 30, 40, 0.7)'
    },
    matrix: { 
      name: 'Matrix', 
      primary: '#00c853', 
      secondary: '#00b248', 
      accent: '#69f0ae',
      background: 'linear-gradient(135deg, #001a00 0%, #003300 100%)',
      backgroundLight: '#e8f5e9',
      text: '#00ff41',
      textDark: '#003300',
      cardBg: 'rgba(0, 30, 0, 0.6)'
    },
    cyber: { 
      name: 'Cyberpunk', 
      primary: '#ff4081', 
      secondary: '#7c4dff', 
      accent: '#00bcd4',
      background: 'linear-gradient(135deg, #0f0015 0%, #1a0033 100%)',
      backgroundLight: '#f3e5f5',
      text: '#ff69b4',
      textDark: '#4a148c',
      cardBg: 'rgba(40, 0, 80, 0.6)'
    }
  };

  // Set CSS variables based on theme
  useEffect(() => {
    const root = document.documentElement;
    const theme = themes[currentTheme];
    
    root.style.setProperty('--primary-color', theme.primary);
    root.style.setProperty('--secondary-color', theme.secondary);
    root.style.setProperty('--accent-color', theme.accent);
    root.style.setProperty('--primary-rgb', 
      `${parseInt(theme.primary.slice(1, 3), 16)}, ` +
      `${parseInt(theme.primary.slice(3, 5), 16)}, ` +
      `${parseInt(theme.primary.slice(5, 7), 16)}`
    );
    
    // Set dark/light mode
    document.body.className = darkMode ? 'dark' : 'light';
  }, [currentTheme, darkMode]);

  const currentThemeData = themes[currentTheme];

  const scrollToBottom = () => {
    if (autoScroll) messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => scrollToBottom(), [messages]);

  useEffect(() => {
    if (showPerformance) {
      performanceInterval.current = setInterval(() => {
        setPerformance(prev => ({
          cpu: Math.max(15, Math.min(95, prev.cpu + (Math.random() - 0.5) * 12)),
          gpu: Math.max(10, Math.min(90, prev.gpu + (Math.random() - 0.5) * 10)),
          memory: Math.max(25, Math.min(95, prev.memory + (Math.random() - 0.5) * 8)),
          disk: Math.max(40, Math.min(98, prev.disk + (Math.random() - 0.5) * 3)),
          network: Math.random() > 0.85 ? 'Excellent' : Math.random() > 0.6 ? 'Good' : 
                   Math.random() > 0.3 ? 'Fair' : 'Poor',
          ping: Math.max(8, Math.min(150, prev.ping + (Math.random() - 0.5) * 25)),
          temperature: Math.max(35, Math.min(80, prev.temperature + (Math.random() - 0.5) * 6)),
          powerUsage: Math.max(80, Math.min(300, prev.powerUsage + (Math.random() - 0.5) * 20))
        }));
      }, 1500);
    } else {
      clearInterval(performanceInterval.current);
    }
    return () => clearInterval(performanceInterval.current);
  }, [showPerformance]);

  const handleSend = useCallback(async () => {
    if (!inputValue.trim()) return;

    const userMessage = { 
      id: Date.now(), 
      text: inputValue, 
      sender: 'user', 
      timestamp: new Date(),
      status: 'sending'
    };
    
    setMessages(prev => [...prev, { ...userMessage, status: 'sent' }]);
    const query = inputValue;
    setInputValue('');
    setIsTyping(true);

    // Simulate AI processing time
    const processingTime = 800 + Math.random() * 1200;
    
    setTimeout(() => {
      const responses = [
        `I've analyzed your query: "${query}". Here's what I found...`,
        `Based on my analysis of "${query}", I can provide the following information...`,
        `Processing complete. Here's the response to "${query}":`,
        `After reviewing your request about "${query}", I've compiled the following...`
      ];
      
      const aiResponse = { 
        id: Date.now() + 1, 
        text: responses[Math.floor(Math.random() * responses.length)],
        sender: 'ai', 
        timestamp: new Date(),
        status: 'delivered'
      };
      
      setMessages(prev => [...prev, aiResponse]);
      setIsTyping(false);
      
      // Auto-scroll to bottom after new message
      setTimeout(scrollToBottom, 100);
    }, processingTime);
  }, [inputValue]);

  const getPerformanceColor = (value, type = 'default') => {
    if (type === 'temperature') {
      if (value < 45) return currentThemeData.accent;
      if (value < 65) return '#ffff00';
      return '#ff4500';
    }
    if (value < 40) return currentThemeData.accent;
    if (value < 70) return '#ffff00';
    return currentThemeData.secondary;
  };

  const getNetworkColor = (status) => {
    const colors = { 'Excellent': currentThemeData.accent, 'Good': currentThemeData.primary,
                    'Fair': '#ffff00', default: '#ff4500' };
    return colors[status] || colors.default;
  };

  const JarvisLogo = () => (
    <div className="logo-container">
      <div className="logo-circle" style={{ borderColor: currentThemeData.primary,
           boxShadow: `0 0 20px ${currentThemeData.primary}` }}>
        <div className="logo-inner" style={{ background: `radial-gradient(circle, ${currentThemeData.primary}20, transparent)` }}>
          <div className="logo-core" style={{ backgroundColor: currentThemeData.primary,
               boxShadow: `0 0 15px ${currentThemeData.primary}` }}></div>
          <div className="logo-ring" style={{ borderColor: currentThemeData.accent,
               boxShadow: `0 0 10px ${currentThemeData.accent}` }}></div>
        </div>
      </div>
    </div>
  );

  const PerformancePanel = () => (
    <div className="performance-panel" style={{ borderColor: currentThemeData.primary,
         boxShadow: `0 0 30px ${currentThemeData.primary}30` }}>
      <div className="panel-header">
        <h3 className="panel-title" style={{ color: currentThemeData.text,
            textShadow: `0 0 10px ${currentThemeData.primary}` }}>System Performance Monitor</h3>
        <button onClick={() => setShowPerformance(false)} className="close-button"
                style={{ color: currentThemeData.secondary }}>
          <X size={20} />
        </button>
      </div>
      
      <div className="performance-grid">
        <div className="performance-row">
          {[{icon: Cpu, label: 'CPU Usage', value: performance.cpu},
            {icon: Monitor, label: 'GPU Usage', value: performance.gpu}].map((item, i) => (
            <div key={i} className="performance-item">
              <div className="performance-label" style={{ color: currentThemeData.primary }}>
                <item.icon size={16} />
                <span>{item.label}</span>
              </div>
              <div className="progress-bar" style={{ borderColor: `${currentThemeData.primary}30` }}>
                <div className="progress-fill" style={{
                  width: `${item.value}%`, backgroundColor: getPerformanceColor(item.value),
                  boxShadow: `0 0 10px ${getPerformanceColor(item.value)}`
                }}/>
              </div>
              <span className="performance-value" style={{ color: currentThemeData.text }}>
                {Math.round(item.value)}%
              </span>
            </div>
          ))}
        </div>

        <div className="performance-row">
          {[{icon: HardDrive, label: 'Memory', value: performance.memory},
            {icon: Activity, label: 'Disk Usage', value: performance.disk}].map((item, i) => (
            <div key={i} className="performance-item">
              <div className="performance-label" style={{ color: currentThemeData.primary }}>
                <item.icon size={16} />
                <span>{item.label}</span>
              </div>
              <div className="progress-bar" style={{ borderColor: `${currentThemeData.primary}30` }}>
                <div className="progress-fill" style={{
                  width: `${item.value}%`, backgroundColor: getPerformanceColor(item.value),
                  boxShadow: `0 0 10px ${getPerformanceColor(item.value)}`
                }}/>
              </div>
              <span className="performance-value" style={{ color: currentThemeData.text }}>
                {Math.round(item.value)}%
              </span>
            </div>
          ))}
        </div>

        <div className="network-row">
          <div className="network-item">
            <div className="performance-label" style={{ color: currentThemeData.primary }}>
              <Wifi size={16} />
              <span>Network Status</span>
            </div>
            <div className="network-status">
              <span style={{ color: getNetworkColor(performance.network),
                           textShadow: `0 0 10px ${getNetworkColor(performance.network)}`,
                           fontWeight: 'bold' }}>
                {performance.network}
              </span>
              <span className="ping-value" style={{ color: currentThemeData.text }}>
                {Math.round(performance.ping)}ms
              </span>
            </div>
          </div>

          <div className="network-item">
            <div className="performance-label" style={{ color: currentThemeData.primary }}>
              <Zap size={16} />
              <span>Temperature</span>
            </div>
            <div className="network-status">
              <span style={{ color: getPerformanceColor(performance.temperature, 'temperature'),
                           textShadow: `0 0 10px ${getPerformanceColor(performance.temperature, 'temperature')}`,
                           fontWeight: 'bold' }}>
                {Math.round(performance.temperature)}°C
              </span>
              <span className="ping-value" style={{ color: currentThemeData.text }}>
                {Math.round(performance.powerUsage)}W
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="container">
      <header className="header">
        <div className="logo">
          <div className="logo-container">
            <div className="logo-circle" style={{ borderColor: darkMode ? '#4CAF50' : '#34C759', boxShadow: `0 0 20px ${darkMode ? '#4CAF50' : '#34C759'}` }}>
              <div className="logo-inner" style={{ background: `radial-gradient(circle, ${darkMode ? '#4CAF50' : '#34C759'}20, transparent)` }}>
                <div className="logo-core" style={{ backgroundColor: darkMode ? '#4CAF50' : '#34C759', boxShadow: `0 0 15px ${darkMode ? '#4CAF50' : '#34C759'}` }}></div>
                <div className="logo-ring" style={{ borderColor: darkMode ? '#4CAF50' : '#34C759', boxShadow: `0 0 10px ${darkMode ? '#4CAF50' : '#34C759'}` }}></div>
              </div>
            </div>
          </div>
          <h1 className="title">JARVIS AI</h1>
        </div>
        
        <div className="header-controls">
          <div className="status-indicator">
            <div className="status-dot" style={{ backgroundColor: darkMode ? '#4CAF50' : '#34C759', boxShadow: `0 0 15px ${darkMode ? '#4CAF50' : '#34C759'}` }}></div>
            <span className="status-text" style={{ color: darkMode ? '#4CAF50' : '#34C759', textShadow: `0 0 5px ${darkMode ? '#4CAF50' : '#34C759'}` }}>ONLINE</span>
          </div>
          
          <button 
            onClick={toggleDarkMode} 
            className="header-button theme-toggle"
            aria-label={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
            title={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
          >
            {darkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>
          
          <button 
            onClick={() => setShowPerformance(!showPerformance)} 
            className="header-button"
            aria-label="Performance Monitor"
            title="Performance Monitor"
          >
            <Monitor size={18} />
          </button>
          
          <button 
            onClick={() => setShowSettings(!showSettings)} 
            className="header-button"
            aria-label="Settings"
            title="Settings"
          >
            <Settings size={18} />
          </button>
          
          <button 
            className="mobile-menu-button"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            aria-label="Menu"
            aria-expanded={isMobileMenuOpen}
          >
            {isMobileMenuOpen ? <ChevronUp size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </header>

      {showPerformance && (
        <div className="performance-panel">
          <div className="panel-header">
            <h2 className="panel-title">System Status</h2>
            <button 
              onClick={() => setShowPerformance(false)} 
              className="close-button"
              aria-label="Close performance monitor"
            >
              <X size={20} />
            </button>
          </div>
          
          <div className="performance-grid">
            <div className="performance-row">
              <div className="performance-item">
                <div className="performance-label">
                  <Cpu size={16} />
                  <span>CPU Usage</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{
                      width: `${performance.cpu}%`,
                      backgroundColor: getPerformanceColor(performance.cpu)
                    }}
                  />
                </div>
                <span className="performance-value">{Math.round(performance.cpu)}%</span>
              </div>
              
              <div className="performance-item">
                <div className="performance-label">
                  <Monitor size={16} />
                  <span>GPU Usage</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{
                      width: `${performance.gpu}%`,
                      backgroundColor: getPerformanceColor(performance.gpu)
                    }}
                  />
                </div>
                <span className="performance-value">{Math.round(performance.gpu)}%</span>
              </div>
            </div>
            
            <div className="performance-row">
              <div className="performance-item">
                <div className="performance-label">
                  <HardDrive size={16} />
                  <span>Memory</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{
                      width: `${performance.memory}%`,
                      backgroundColor: getPerformanceColor(performance.memory)
                    }}
                  />
                </div>
                <span className="performance-value">{Math.round(performance.memory)}%</span>
              </div>
              
              <div className="performance-item">
                <div className="performance-label">
                  <Activity size={16} />
                  <span>Disk Usage</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{
                      width: `${performance.disk}%`,
                      backgroundColor: getPerformanceColor(performance.disk)
                    }}
                  />
                </div>
                <span className="performance-value">{Math.round(performance.disk)}%</span>
              </div>
            </div>
            
            <div className="performance-row">
              <div className="performance-item">
                <div className="performance-label">
                  <Wifi size={16} />
                  <span>Network</span>
                </div>
                <div className="network-status">
                  <span className="network-value">{performance.network}</span>
                  <span className="ping-value">{Math.round(performance.ping)}ms</span>
                </div>
              </div>
              
              <div className="performance-item">
                <div className="performance-label">
                  <Zap size={16} />
                  <span>Temperature</span>
                </div>
                <div className="temperature-status">
                  <span className="temperature-value">
                    {Math.round(performance.temperature)}°C
                  </span>
                  <span className="power-usage">{Math.round(performance.powerUsage)}W</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {showSettings && (
        <div className="settings-panel" style={{ 
          borderColor: currentThemeData.secondary,
          boxShadow: `0 0 30px ${currentThemeData.secondary}30` 
        }}>
          <div className="panel-header">
            <h2 className="panel-title" style={{ 
              color: currentThemeData.text,
              textShadow: `0 0 10px ${currentThemeData.secondary}` 
            }}>
              Settings
            </h2>
            <button 
              onClick={() => setShowSettings(false)} 
              className="close-button"
              style={{ color: currentThemeData.secondary }}
              aria-label="Close settings"
            >
              <X size={20} />
            </button>
          </div>
          
          <div className="settings-grid">
            <div className="setting-item">
              <label className="setting-label" style={{ color: currentThemeData.secondary }}>
                <Palette size={16} />
                Theme Selection
              </label>
              <select 
                className="setting-select"
                value={currentTheme}
                onChange={(e) => setCurrentTheme(e.target.value)}
                style={{ 
                  borderColor: currentThemeData.secondary, 
                  color: currentThemeData.text 
                }}
                aria-label="Select theme"
              >
                {Object.entries(themes).map(([key, theme]) => (
                  <option key={key} value={key}>
                    {theme.name}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="setting-item">
              <label className="setting-label" style={{ color: currentThemeData.secondary }}>
                <Activity size={16} />
                Animation Speed
              </label>
              <div className="slider-container">
                <input 
                  type="range" 
                  min="0.5" 
                  max="2" 
                  step="0.1" 
                  value={animationSpeed}
                  onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
                  className="setting-slider"
                  style={{ accentColor: currentThemeData.secondary }}
                  aria-label="Animation speed"
                />
                <span className="slider-value" style={{ color: currentThemeData.text }}>
                  {animationSpeed.toFixed(1)}x
                </span>
              </div>
            </div>
            
            {[
              {icon: Monitor, label: 'Auto-scroll Messages', checked: autoScroll, onChange: setAutoScroll},
              {icon: Zap, label: 'Sound Effects', checked: soundEffects, onChange: setSoundEffects},
              {icon: Shield, label: 'Privacy Mode', checked: false, onChange: () => {}}
            ].map((item, i) => (
              <div key={i} className="setting-item">
                <label className="setting-label" style={{ color: currentThemeData.secondary }}>
                  <item.icon size={16} />
                  {item.label}
                </label>
                <input 
                  type="checkbox"
                  checked={item.checked}
                  onChange={(e) => item.onChange(e.target.checked)}
                  className="setting-checkbox"
                  style={{ accentColor: currentThemeData.secondary }}
                  aria-label={item.label}
                />
              </div>
            ))}
          </div>
        </div>
      )}
      
      <main className="chat-container">
        <div className="messages-area">
          {messages.map((message, index) => (
            <div 
              key={message.id} 
              className={`message-container ${message.sender === 'user' ? 'user-message' : 'ai-message'}`}
              style={{
                animationDelay: `${index * 0.05}s`,
                opacity: message.status === 'sending' ? 0.7 : 1
              }}
            >
              <div className="message-content">
                {message.sender === 'ai' && (
                  <div className="message-avatar">
                    <Terminal size={16} />
                  </div>
                )}
                <div className="message-bubble">
                  <p className="message-text">{message.text}</p>
                  <div className="message-footer">
                    <span className="timestamp">
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    {message.sender === 'user' && (
                      <span className="message-status">
                        {message.status === 'sending' ? '🕒' : '✓✓'}
                      </span>
                    )}
                  </div>
                </div>
                {message.sender === 'user' && (
                  <div className="message-avatar user">
                    <Code size={16} />
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {isTyping && (
            <div className="message-container ai-message" style={{ animationDelay: '0.1s' }}>
              <div className="message-content">
                <div className="message-avatar">
                  <Terminal size={16} />
                </div>
                <div className="message-bubble">
                  <div className="typing-indicator">
                    {[0, 0.2, 0.4].map((delay, i) => (
                      <span 
                        key={i} 
                        className="typing-dot"
                        style={{ animationDelay: `${delay}s` }}
                      />
                    ))}
                    <span className="typing-text">
                      JARVIS is thinking...
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="input-form">
          <div 
            className={`input-container ${inputFocused ? 'focused' : ''}`}
            onFocus={() => setInputFocused(true)}
            onBlur={() => setInputFocused(false)}
          >
            <button 
              className="attachment-button"
              type="button"
              aria-label="Attach file"
              title="Attach file"
            >
              <Paperclip size={20} />
            </button>
            
            <input 
              type="text" 
              value={inputValue} 
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              onFocus={() => setInputFocused(true)}
              onBlur={() => setInputFocused(false)}
              placeholder="Message JARVIS..." 
              className="input"
              aria-label="Type your message"
            />
            
            <button 
              onClick={handleSend} 
              className={`send-button ${!inputValue.trim() ? 'disabled' : ''}`}
              disabled={!inputValue.trim()}
              aria-label="Send message"
            >
              <Send size={18} />
            </button>
          </div>
          
          <div className="input-hint">
            Press <kbd>Enter</kbd> to send, <kbd>Shift</kbd> + <kbd>Enter</kbd> for new line
          </div>
        </div>
      </main>
    </div>
  );
}

// Export voor CDN-gebruik
if (typeof module !== 'undefined' && module.exports) {
  module.exports = EnhancedJarvisAI;
} else {
  window.EnhancedJarvisAI = EnhancedJarvisAI;
}