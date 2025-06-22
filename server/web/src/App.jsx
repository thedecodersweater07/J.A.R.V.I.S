import React from 'react';
let uuidv4;
try {
  uuidv4 = require('uuid').v4;
} catch {
  uuidv4 = () => Math.random().toString(36).substring(2, 15);
}
import Chat from './components/Chat';
import StatusPanel from './components/StatusPanel';
import './App.css';

const App = () => {
  // Generate a unique client ID for this session
  const clientId = React.useMemo(() => uuidv4(), []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>J.A.R.V.I.S.</h1>
        <StatusPanel />
      </header>

      <main className="app-main">
        <Chat clientId={clientId} />
      </main>

      <footer className="app-footer">
        <p>© {new Date().getFullYear()} J.A.R.V.I.S. AI Assistant</p>
      </footer>
    </div>
  );
};

export default App;