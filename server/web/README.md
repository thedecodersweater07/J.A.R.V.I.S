# JARVIS Web Interface

This is the modern web frontend for the JARVIS AI Assistant.

## Features
- Modern React (Vite) chat UI
- Responsive design (desktop/mobile)
- Theme toggle (light/dark)
- API integration with FastAPI backend
- Production-ready static build

## Project Structure
```
server/web/
├── static/          # Static files for direct serving
├── dist/            # Built/compiled files (production)
├── src/             # Source files (React components, CSS)
├── index.html       # Main HTML file
├── package.json     # Dependencies and scripts
└── README.md        # This file
```

## Setup & Development

1. **Install dependencies:**
   ```sh
   npm install
   ```
2. **Start development server:**
   ```sh
   npm run dev
   ```
   The app will be available at http://localhost:5173

3. **Build for production:**
   ```sh
   npm run build
   ```
   The build output will be in `dist/`.

4. **Copy build to static (if needed):**
   ```sh
   cp -r dist/* static/
   ```

## API
- The frontend communicates with the FastAPI backend at `/api` and `/ws` (WebSocket).

## Customization
- Edit `src/components/Chat.jsx` for chat UI logic.
- Edit `src/App.jsx` for main app logic.
- Edit `src/index.css` for global styles.

---
MIT License
