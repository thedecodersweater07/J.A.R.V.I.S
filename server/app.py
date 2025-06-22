"""
J.A.R.V.I.S. FastAPI Server with Auto Frontend Build & Deploy
-------------------------------------------------------------
Main server application that handles HTTP and WebSocket endpoints.
Automatically builds and deploys frontend assets on startup.
"""

import logging
import asyncio
import subprocess
import shutil
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from server.config import Settings
from server.api.routes import api_router
from server.websocket.manager import websocket_router
from server.middleware.security import add_security_headers
from server.utils.logging import configure_logging


class FrontendBuilder:
    """Handles frontend build and deployment operations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.base_path = Path(__file__).parent.parent
        self.frontend_dir = self.base_path / "frontend"
        self.build_dir = self.frontend_dir / "build" 
        self.dist_dir = self.frontend_dir / "dist"
        self.deploy_dir = self.base_path / "web" / "static"
    
    async def auto_build_and_deploy(self) -> bool:
        """Automatically detect, build and deploy frontend."""
        try:
            # Check if frontend directory exists
            if not self.frontend_dir.exists():
                self.logger.info("No frontend directory found - skipping auto build")
                return False
            
            # Detect frontend type and build
            build_success = await self._detect_and_build()
            if not build_success:
                return False
            
            # Deploy built assets
            deploy_success = await self._deploy_assets()
            if deploy_success:
                self.logger.info("✅ Frontend auto-build and deploy completed successfully")
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Frontend auto-build failed: {e}")
            
        return False
    
    async def _detect_and_build(self) -> bool:
        """Detect frontend type and run appropriate build command."""
        package_json = self.frontend_dir / "package.json"
        
        if not package_json.exists():
            self.logger.warning("No package.json found - trying manual JSX/JS detection")
            return await self._build_manual()
        
        # Parse package.json to determine build type
        try:
            with open(package_json, 'r', encoding='utf-8') as f:
                pkg_data = json.load(f)
            
            scripts = pkg_data.get('scripts', {})
            dependencies = {**pkg_data.get('dependencies', {}), **pkg_data.get('devDependencies', {})}
            
            # Detect framework and build
            if 'react' in dependencies or 'react-scripts' in dependencies:
                return await self._build_react(scripts)
            elif 'vue' in dependencies or '@vue/cli' in dependencies:
                return await self._build_vue(scripts)
            elif 'vite' in dependencies:
                return await self._build_vite(scripts)
            elif 'webpack' in dependencies:
                return await self._build_webpack(scripts)
            else:
                return await self._build_generic(scripts)
                
        except Exception as e:
            self.logger.error(f"Error parsing package.json: {e}")
            return await self._build_manual()
    
    async def _build_react(self, scripts: dict) -> bool:
        """Build React application."""
        self.logger.info("🔨 Building React application...")
        
        build_cmd = scripts.get('build', 'npm run build')
        if await self._run_build_command(build_cmd):
            # React typically builds to 'build' directory
            self.build_output_dir = self.build_dir
            return True
        return False
    
    async def _build_vue(self, scripts: dict) -> bool:
        """Build Vue application."""
        self.logger.info("🔨 Building Vue application...")
        
        build_cmd = scripts.get('build', 'npm run build')
        if await self._run_build_command(build_cmd):
            # Vue typically builds to 'dist' directory
            self.build_output_dir = self.dist_dir
            return True
        return False
    
    async def _build_vite(self, scripts: dict) -> bool:
        """Build Vite application."""
        self.logger.info("🔨 Building Vite application...")
        
        build_cmd = scripts.get('build', 'npm run build')
        if await self._run_build_command(build_cmd):
            # Vite typically builds to 'dist' directory
            self.build_output_dir = self.dist_dir
            return True
        return False
    
    async def _build_webpack(self, scripts: dict) -> bool:
        """Build Webpack application."""
        self.logger.info("🔨 Building Webpack application...")
        
        build_cmd = scripts.get('build', 'npm run build')
        if await self._run_build_command(build_cmd):
            # Check both common output directories
            if self.build_dir.exists():
                self.build_output_dir = self.build_dir
            elif self.dist_dir.exists():
                self.build_output_dir = self.dist_dir
            else:
                self.logger.error("No build output found in 'build' or 'dist' directories")
                return False
            return True
        return False
    
    async def _build_generic(self, scripts: dict) -> bool:
        """Build generic Node.js application."""
        self.logger.info("🔨 Building generic frontend application...")
        
        # Try common build commands
        for cmd in ['npm run build', 'yarn build', 'pnpm build']:
            if await self._run_build_command(cmd, ignore_errors=True):
                # Find the build output
                for output_dir in [self.build_dir, self.dist_dir]:
                    if output_dir.exists():
                        self.build_output_dir = output_dir
                        return True
        
        return False
    
    async def _build_manual(self) -> bool:
        """Manual build for simple JSX/JS files using basic tools."""
        self.logger.info("🔨 Attempting manual JSX/JS build...")
        
        # Check for JSX files
        jsx_files = list(self.frontend_dir.rglob("*.jsx")) + list(self.frontend_dir.rglob("*.js"))
        if not jsx_files:
            self.logger.warning("No JSX/JS files found for manual build")
            return False
        
        # Create a simple build directory and copy files
        manual_build_dir = self.frontend_dir / "manual_build"
        manual_build_dir.mkdir(exist_ok=True)
        
        # Copy all frontend files to build directory
        for file_path in self.frontend_dir.iterdir():
            if file_path.is_file() and file_path.suffix in ['.html', '.js', '.jsx', '.css']:
                shutil.copy2(file_path, manual_build_dir)
        
        self.build_output_dir = manual_build_dir
        self.logger.info(f"Manual build completed: {len(jsx_files)} files processed")
        return True
    
    async def _run_build_command(self, command: str, ignore_errors: bool = False) -> bool:
        """Run a build command asynchronously."""
        try:
            self.logger.info(f"Running: {command}")
            
            # Split command for subprocess
            cmd_parts = command.split()
            
            # Run the command
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                cwd=self.frontend_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info("✅ Build command completed successfully")
                if stdout:
                    self.logger.debug(f"Build output: {stdout.decode()}")
                return True
            else:
                if not ignore_errors:
                    self.logger.error(f"❌ Build command failed with code {process.returncode}")
                    if stderr:
                        self.logger.error(f"Build error: {stderr.decode()}")
                return False
                
        except FileNotFoundError:
            if not ignore_errors:
                self.logger.error(f"❌ Command not found: {command}")
            return False
        except Exception as e:
            if not ignore_errors:
                self.logger.error(f"❌ Build command error: {e}")
            return False
    
    async def _deploy_assets(self) -> bool:
        """Deploy built assets to web/static directory."""
        try:
            if not hasattr(self, 'build_output_dir') or not self.build_output_dir.exists():
                self.logger.error("No build output directory found")
                return False
            
            # Ensure deploy directory exists
            self.deploy_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear existing deployment
            for item in self.deploy_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            
            # Copy built assets
            deployed_files = 0
            for item in self.build_output_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, self.deploy_dir)
                    deployed_files += 1
                elif item.is_dir():
                    shutil.copytree(item, self.deploy_dir / item.name)
                    deployed_files += len(list(item.rglob("*")))
            
            self.logger.info(f"✅ Deployed {deployed_files} files to {self.deploy_dir}")
            
            # Verify index.html exists
            index_file = self.deploy_dir / "index.html"
            if index_file.exists():
                self.logger.info("✅ index.html successfully deployed")
                return True
            else:
                self.logger.warning("⚠️ No index.html found in build output")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Asset deployment failed: {e}")
            return False


class JarvisServer:
    """Main server class for J.A.R.V.I.S. web interface."""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = configure_logging(self.settings)
        self.frontend_builder = FrontendBuilder(self.logger)
        self.app = self._create_app()
        self._setup_static_paths()
        self._setup_middleware()
        self._setup_routes()
        self._setup_static_files()
        self._setup_exception_handlers()
    
    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        return FastAPI(
            title="J.A.R.V.I.S. Web UI",
            description="A modern, integrated web interface for the J.A.R.V.I.S. system.",
            version=self.settings.version,
            docs_url="/api/docs" if self.settings.debug else None,
            redoc_url="/api/redoc" if self.settings.debug else None,
            lifespan=self._lifespan
        )
    
    def _setup_static_paths(self):
        """Initialize static file paths."""
        base_path = Path(__file__).parent.parent
        self._static_dir = (base_path / "web" / "static").resolve()
        self._web_dir = (base_path / "web").resolve()
        
        # Log the paths for debugging
        self.logger.info(f"Static directory: {self._static_dir}")
        self.logger.info(f"Web directory: {self._web_dir}")
    
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Handle application startup and shutdown."""
        self.logger.info("🚀 J.A.R.V.I.S. server startup initiated")
        
        # Auto-build and deploy frontend
        await self.frontend_builder.auto_build_and_deploy()
        
        # Initialize other services
        await self._startup_tasks()
        
        self.logger.info("✅ J.A.R.V.I.S. server startup complete")
        yield
        
        # Cleanup services, cache, etc.
        await self._shutdown_tasks()
        self.logger.info("🛑 J.A.R.V.I.S. server shutdown complete")
    
    async def _startup_tasks(self):
        """Execute startup tasks."""
        # Add your startup initialization here
        pass
    
    async def _shutdown_tasks(self):
        """Execute shutdown cleanup tasks."""
        # Add your cleanup tasks here
        pass
    
    def _setup_middleware(self):
        """Configure application middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Compression middleware
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Security headers middleware
        self.app.add_middleware(BaseHTTPMiddleware, dispatch=add_security_headers)
    
    def _setup_routes(self):
        """Configure application routes."""
        self.app.include_router(api_router, prefix="/api")
        self.app.include_router(websocket_router)

        # Root endpoint - serve index.html directly
        @self.app.get("/")
        async def root():
            return await self._serve_index()
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy", 
                "version": self.settings.version,
                "frontend_deployed": (self._static_dir / "index.html").exists()
            }
        
        # Manual rebuild endpoint (development only)
        @self.app.post("/api/rebuild-frontend")
        async def rebuild_frontend():
            if not self.settings.debug:
                raise HTTPException(status_code=403, detail="Rebuild only available in debug mode")
            
            success = await self.frontend_builder.auto_build_and_deploy()
            return {
                "success": success,
                "message": "Frontend rebuild completed" if success else "Frontend rebuild failed"
            }
    
    def _setup_static_files(self):
        """Configure static file serving."""
        # Only mount directories that exist and have files
        if self._directory_has_files(self._static_dir):
            self.app.mount("/static", StaticFiles(directory=str(self._static_dir)), name="static")
            self.logger.info(f"📁 Mounted static directory: {self._static_dir}")

        if self._directory_has_files(self._web_dir):
            # Mount web directory for other assets
            self.app.mount("/assets", StaticFiles(directory=str(self._web_dir)), name="web-assets")
            self.logger.info(f"📁 Mounted web assets directory: {self._web_dir}")
    
    def _setup_exception_handlers(self):
        """Configure global exception handlers."""
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            self.logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "path": str(request.url.path)}
            )
        
        @self.app.exception_handler(StarletteHTTPException)
        async def http_exception_handler(request: Request, exc: StarletteHTTPException):
            self.logger.warning(f"HTTP {exc.status_code} on {request.url}: {exc.detail}")
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail, "path": str(request.url.path)}
            )
        
        @self.app.exception_handler(404)
        async def not_found_handler(request: Request, exc: HTTPException):
            # Try to serve index.html for SPA routing
            if not request.url.path.startswith("/api/"):
                index_response = await self._serve_index()
                if isinstance(index_response, FileResponse):
                    return index_response
            
            return JSONResponse(
                status_code=404,
                content={"detail": f"Resource not found: {request.url.path}"}
            )
    
    async def _serve_index(self) -> Response:
        """Serve the main index.html file or fallback."""
        # Priority: static/index.html first, then web/index.html
        static_index = self._static_dir / "index.html"
        web_index = self._web_dir / "index.html"
        
        if static_index.exists():
            self.logger.debug(f"📄 Serving index.html from: {static_index}")
            return FileResponse(
                static_index,
                media_type="text/html",
                headers={"Cache-Control": "no-cache"}
            )
        
        if web_index.exists():
            self.logger.debug(f"📄 Serving index.html from: {web_index}")
            return FileResponse(
                web_index,
                media_type="text/html",
                headers={"Cache-Control": "no-cache"}
            )
        
        # Fallback: show status page when no index.html is found
        self.logger.warning("⚠️ No index.html found - serving status page")
        return await self._serve_status_page()
    
    async def _serve_status_page(self) -> HTMLResponse:
        """Serve a status page when index.html is missing."""
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>J.A.R.V.I.S. Server Status</title>
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #0f1419 0%, #1a1f29 100%);
                    color: #e8eaed;
                    margin: 0;
                    padding: 2rem;
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .container {
                    max-width: 700px;
                    text-align: center;
                    background: rgba(255, 255, 255, 0.05);
                    padding: 3rem;
                    border-radius: 20px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                h1 { 
                    font-size: 2.5rem;
                    margin-bottom: 1rem;
                    background: linear-gradient(45deg, #00d4ff, #0099cc);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
                .status { 
                    font-size: 1.5rem;
                    margin: 1rem 0;
                    padding: 1rem;
                    background: rgba(0, 255, 0, 0.1);
                    border-radius: 10px;
                    border: 1px solid rgba(0, 255, 0, 0.3);
                }
                .error {
                    background: rgba(255, 0, 0, 0.1);
                    border: 1px solid rgba(255, 0, 0, 0.3);
                    color: #ff6b6b;
                }
                .info-box {
                    text-align: left;
                    background: rgba(255, 255, 255, 0.03);
                    padding: 1.5rem;
                    border-radius: 10px;
                    margin: 2rem 0;
                }
                .info-box h3 {
                    color: #00d4ff;
                    margin-top: 0;
                }
                ul { 
                    list-style: none;
                    padding: 0;
                }
                li {
                    padding: 0.5rem 0;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }
                li:last-child {
                    border-bottom: none;
                }
                code {
                    background: rgba(0, 0, 0, 0.3);
                    padding: 0.2rem 0.5rem;
                    border-radius: 4px;
                    font-family: 'Courier New', monospace;
                }
                a {
                    color: #00d4ff;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
                .rebuild-btn {
                    background: linear-gradient(45deg, #00d4ff, #0099cc);
                    color: white;
                    border: none;
                    padding: 1rem 2rem;
                    border-radius: 10px;
                    cursor: pointer;
                    font-size: 1rem;
                    margin: 1rem;
                    transition: transform 0.2s;
                }
                .rebuild-btn:hover {
                    transform: scale(1.05);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 J.A.R.V.I.S.</h1>
                <div class="status">
                    ✅ Server Online & Auto-Build Ready
                </div>
                <div class="status error">
                    ❌ Frontend Build Missing
                </div>
                
                <div class="info-box">
                    <h3>🔨 Auto-Build Support:</h3>
                    <ul>
                        <li>React (create-react-app, Next.js)</li>
                        <li>Vue.js (Vue CLI, Nuxt.js)</li>
                        <li>Vite (React, Vue, Vanilla)</li>
                        <li>Webpack (custom configs)</li>
                        <li>Manual JSX/JS files</li>
                    </ul>
                </div>
                
                <div class="info-box">
                    <h3>📁 Expected Structure:</h3>
                    <ul>
                        <li><code>frontend/package.json</code> - For npm/yarn projects</li>
                        <li><code>frontend/src/</code> - Source files (JSX/JS/CSS)</li>
                        <li><code>frontend/*.jsx</code> - For manual builds</li>
                        <li>Output: <code>web/static/index.html</code></li>
                    </ul>
                </div>
                
                <div class="info-box">
                    <h3>🚀 Quick Start:</h3>
                    <ul>
                        <li>Create <code>frontend/</code> directory</li>
                        <li>Add your React/Vue/JS project files</li>
                        <li>Restart server for auto-build</li>
                        <li>Or use manual rebuild button below</li>
                    </ul>
                </div>
                
                <button class="rebuild-btn" onclick="rebuildFrontend()">
                    🔨 Manual Rebuild Frontend
                </button>
                
                <div style="margin-top: 2rem;">
                    <a href="/api/health">📊 Server Health</a> |
                    <a href="/api/docs">📚 API Docs</a>
                </div>
            </div>
            
            <script>
                async function rebuildFrontend() {
                    const btn = document.querySelector('.rebuild-btn');
                    btn.disabled = true;
                    btn.textContent = '🔨 Building...';
                    
                    try {
                        const response = await fetch('/api/rebuild-frontend', { method: 'POST' });
                        const result = await response.json();
                        
                        if (result.success) {
                            btn.textContent = '✅ Build Complete!';
                            setTimeout(() => location.reload(), 2000);
                        } else {
                            btn.textContent = '❌ Build Failed';
                            setTimeout(() => {
                                btn.textContent = '🔨 Manual Rebuild Frontend';
                                btn.disabled = false;
                            }, 3000);
                        }
                    } catch (error) {
                        btn.textContent = '❌ Error';
                        setTimeout(() => {
                            btn.textContent = '🔨 Manual Rebuild Frontend';
                            btn.disabled = false;
                        }, 3000);
                    }
                }
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)
    
    @staticmethod
    def _directory_has_files(directory: Path) -> bool:
        """Check if directory exists and contains files."""
        return directory.exists() and any(directory.iterdir())
    
    def get_app(self) -> FastAPI:
        """Get the configured FastAPI application."""
        return self.app


# Create server instance
server = JarvisServer()
app = server.get_app()


if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )