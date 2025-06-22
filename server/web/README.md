# J.A.R.V.I.S. Web Interface

This is the web interface for the J.A.R.V.I.S. AI Assistant. It provides a modern, responsive UI for interacting with the J.A.R.V.I.S. system.

## Features

- Real-time chat interface with WebSocket communication
- System status monitoring
- Dark mode support
- Responsive design
- Modern component architecture

## Tech Stack

- React 18
- Vite
- WebSocket API
- CSS Modules
- Custom Hooks

## Development

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The development server will start at http://localhost:5173 and proxy API requests to the backend at http://localhost:8000.

## Building for Production

1. Build the project:
```bash
npm run build
```

This will create a `static` directory with the production build. The FastAPI backend will serve these files automatically.

## Project Structure

```
server/web/
├── src/
│   ├── components/      # React components
│   ├── hooks/          # Custom React hooks
│   ├── styles/         # Global styles
│   ├── App.jsx         # Main application component
│   └── main.jsx        # Application entry point
├── public/             # Static assets
├── index.html          # HTML template
├── vite.config.js      # Vite configuration
└── package.json        # Project dependencies
```

## Development Guidelines

1. Components should be small and focused (50-70 lines maximum)
2. Use custom hooks for complex logic and state management
3. Follow the CSS naming conventions:
   - BEM-like class names
   - CSS variables for theming
   - Responsive design with mobile-first approach
4. Implement proper error handling and loading states
5. Use TypeScript for better type safety (planned)

## API
- The frontend communicates with the FastAPI backend at `/api` and `/ws` (WebSocket).

## Customization
- Edit `src/components/Chat.jsx` for chat UI logic.
- Edit `src/App.jsx` for main app logic.
- Edit `src/index.css` for global styles.

---
MIT License
