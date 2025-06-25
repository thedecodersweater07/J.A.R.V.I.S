"""
J.A.R.V.I.S. FastAPI Server with Auto Frontend Build & Deploy
-------------------------------------------------------------
Main server application that handles HTTP and WebSocket endpoints.
Automatically builds and deploys frontend assets on startup.

USAGE (Windows/PowerShell):
  Start de server altijd met:
      python -m server.app
  Gebruik géén Uvicorn CLI (uvicorn server.app:app), want dat werkt niet goed in PowerShell/Windows.
  De server is bereikbaar op http://127.0.0.1:8080
"""

import logging
import asyncio
# import subprocess  # Removed: unused import
import shutil
import json
import os
from pathlib import Path
from contextlib import asynccontextmanager
# from typing import Dict, Any, Optional  # Removed: unused imports
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response, HTMLResponse
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
from models.jarvis import JarvisModel


class FrontendBuilder:
    """Handles frontend build and deployment operations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.base_path = Path(__file__).parent.parent
        self.frontend_dir = self.base_path / "frontend"
        self.build_dir = self.frontend_dir / "build" 
        self.dist_dir = self.frontend_dir / "dist"
        # Gebruik nu server/web/static en server/web
        self.deploy_dir = self.base_path / "server" / "web" / "static"
        self.web_dir = self.base_path / "server" / "web"
        
        # Create necessary directories
        self.deploy_dir.mkdir(parents=True, exist_ok=True)
        self.web_dir.mkdir(parents=True, exist_ok=True)
    
    async def auto_build_and_deploy(self) -> bool:
        """Automatically detect, build and deploy frontend."""
        try:
            # Check if frontend directory exists
            if not self.frontend_dir.exists():
                self.logger.info("No frontend directory found - creating sample structure")
                await self._create_sample_frontend()
                return await self._build_sample_frontend()
            
            # Check if directory is empty
            if not any(self.frontend_dir.iterdir()):
                self.logger.info("Frontend directory is empty - creating sample structure")
                await self._create_sample_frontend()
                return await self._build_sample_frontend()
            
            # Detect frontend type and build
            build_success = await self._detect_and_build()
            if not build_success:
                self.logger.warning("Standard build failed - trying fallback methods")
                return await self._build_fallback()
            
            # Deploy built assets
            deploy_success = await self._deploy_assets()
            if deploy_success:
                self.logger.info("✅ Frontend auto-build and deploy completed successfully")
                return True
            
        except Exception as e:
            self.logger.error(f"❌ Frontend auto-build failed: {e}")
            # Try fallback build
            return await self._build_fallback()
            
        return False
    
    async def _create_sample_frontend(self):
        """Create a sample frontend structure for testing."""
        self.logger.info("🏗️ Creating sample frontend structure...")
        
        # Create frontend directory
        self.frontend_dir.mkdir(exist_ok=True)
        
        # Create a simple index.html
        sample_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>J.A.R.V.I.S. Web Interface</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f1419 0%, #1a1f29 100%);
            color: #e8eaed;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            max-width: 800px;
            text-align: center;
            background: rgba(255, 255, 255, 0.05);
            padding: 3rem;
            border-radius: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        h1 {
            font-size: 3rem;
            margin-bottom: 1rem;
            background: linear-gradient(45deg, #00d4ff, #0099cc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: glow 2s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 20px rgba(0, 212, 255, 0.5); }
            to { text-shadow: 0 0 30px rgba(0, 212, 255, 0.8); }
        }
        .status {
            font-size: 1.2rem;
            margin: 2rem 0;
            padding: 1rem;
            background: rgba(0, 255, 0, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(0, 255, 0, 0.3);
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        .feature {
            background: rgba(255, 255, 255, 0.03);
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .feature h3 {
            color: #00d4ff;
            margin-top: 0;
        }
        .api-button {
            background: linear-gradient(45deg, #00d4ff, #0099cc);
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1rem;
            margin: 0.5rem;
            text-decoration: none;
            display: inline-block;
            transition: transform 0.2s;
        }
        .api-button:hover {
            transform: scale(1.05);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 J.A.R.V.I.S.</h1>
        <div class="status">
            ✅ Frontend Successfully Built & Deployed
        </div>
        
        <div class="features">
            <div class="feature">
                <h3>🚀 Auto-Build System</h3>
                <p>Automatically detects and builds React, Vue, Vite, and other frontend frameworks.</p>
            </div>
            <div class="feature">
                <h3>🔧 API Integration</h3>
                <p>RESTful API endpoints with WebSocket support for real-time communication.</p>
            </div>
            <div class="feature">
                <h3>🛡️ Security</h3>
                <p>Built-in security headers, CORS configuration, and request validation.</p>
            </div>
            <div class="feature">
                <h3>📊 Monitoring</h3>
                <p>Health checks, logging, and performance monitoring out of the box.</p>
            </div>
        </div>
        
        <div style="margin-top: 2rem;">
            <a href="/api/health" class="api-button">📊 Health Check</a>
            <a href="/api/docs" class="api-button">📚 API Documentation</a>
        </div>
        
        <div style="margin-top: 2rem; font-size: 0.9rem; opacity: 0.7;">
            <p>Replace this with your custom frontend by adding files to the <code>frontend/</code> directory.</p>
            <p>Supported: React, Vue, Vite, Webpack, or plain HTML/JS/CSS files.</p>
        </div>
    </div>
    
    <script>
        // Simple interactive elements
        document.addEventListener('DOMContentLoaded', function() {
            const title = document.querySelector('h1');
            title.addEventListener('click', function() {
                this.style.transform = 'scale(1.1)';
                setTimeout(() => {
                    this.style.transform = 'scale(1)';
                }, 200);
            });
        });
    </script>
</body>
</html>"""
        
        # Write the sample HTML file
        sample_file = self.frontend_dir / "index.html"
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_html)
        
        self.logger.info(f"✅ Created sample frontend at {sample_file}")
    
    async def _build_sample_frontend(self) -> bool:
        """Build the sample frontend."""
        try:
            # Copy sample frontend to deploy directory (static), NEVER overwrite web/index.html
            sample_file = self.frontend_dir / "index.html"
            if sample_file.exists():
                shutil.copy2(sample_file, self.deploy_dir / "index.html")
                self.logger.info("✅ Sample frontend deployed successfully (to static only, web/index.html never overwritten)")
                return True
        except Exception as e:
            self.logger.error(f"❌ Failed to deploy sample frontend: {e}")
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
            
            # Install dependencies first if node_modules doesn't exist
            node_modules = self.frontend_dir / "node_modules"
            if not node_modules.exists():
                self.logger.info("📦 Installing npm dependencies...")
                if not await self._run_build_command("npm install"):
                    self.logger.warning("npm install failed, trying yarn...")
                    if not await self._run_build_command("yarn install"):
                        self.logger.error("Both npm and yarn install failed")
                        return False
            
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
        self.logger.info("🔨 Attempting manual JSX/JS/HTML build...")
        
        # Check for any web files
        web_files = []
        for pattern in ['*.html', '*.js', '*.jsx', '*.css', '*.ts', '*.tsx']:
            web_files.extend(list(self.frontend_dir.glob(pattern)))
        
        if not web_files:
            self.logger.warning("No web files found for manual build")
            return False
        
        # Create a simple build directory and copy files
        manual_build_dir = self.frontend_dir / "manual_build"
        manual_build_dir.mkdir(exist_ok=True)
        
        # Copy all frontend files to build directory
        copied_files = 0
        for file_path in self.frontend_dir.iterdir():
            if file_path.is_file() and file_path.suffix in ['.html', '.js', '.jsx', '.css', '.ts', '.tsx', '.json']:
                try:
                    shutil.copy2(file_path, manual_build_dir)
                    copied_files += 1
                except Exception as e:
                    self.logger.warning(f"Failed to copy {file_path}: {e}")
        
        # Also copy subdirectories
        for item in self.frontend_dir.iterdir():
            if item.is_dir() and item.name not in ['node_modules', 'build', 'dist', 'manual_build']:
                try:
                    shutil.copytree(item, manual_build_dir / item.name, dirs_exist_ok=True)
                    copied_files += len(list(item.rglob("*")))
                except Exception as e:
                    self.logger.warning(f"Failed to copy directory {item}: {e}")
        
        if copied_files > 0:
            self.build_output_dir = manual_build_dir
            self.logger.info(f"✅ Manual build completed: {copied_files} files processed")
            return True
        
        return False
    
    async def _build_fallback(self) -> bool:
        """Fallback build method when all else fails."""
        self.logger.info("🔄 Attempting fallback build methods...")
        
        # Try to find any HTML files in the frontend directory or subdirectories
        html_files = list(self.frontend_dir.rglob("*.html"))
        
        if html_files:
            self.logger.info(f"Found {len(html_files)} HTML files, using fallback build")
            
            # Create fallback build directory
            fallback_dir = self.frontend_dir / "fallback_build"
            fallback_dir.mkdir(exist_ok=True)
            
            # Copy all files
            for root, dirs, files in os.walk(self.frontend_dir):
                # Skip certain directories
                dirs[:] = [d for d in dirs if d not in ['node_modules', 'build', 'dist', 'fallback_build', '.git']]
                
                for file in files:
                    if file.endswith(('.html', '.js', '.jsx', '.css', '.ts', '.tsx', '.json', '.png', '.jpg', '.jpeg', '.gif', '.svg')):
                        src_path = Path(root) / file
                        rel_path = src_path.relative_to(self.frontend_dir)
                        dst_path = fallback_dir / rel_path
                        
                        # Create directory if it doesn't exist
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        try:
                            shutil.copy2(src_path, dst_path)
                        except Exception as e:
                            self.logger.warning(f"Failed to copy {src_path}: {e}")
            
            self.build_output_dir = fallback_dir
            return True
        
        # If no HTML files found, create a minimal one
        self.logger.info("No HTML files found, creating minimal fallback")
        await self._create_sample_frontend()
        return await self._build_sample_frontend()
    
    async def _run_build_command(self, command: str, ignore_errors: bool = False) -> bool:
        """Run a build command asynchronously."""
        try:
            self.logger.info(f"Running: {command}")
            
            # Split command for subprocess
            cmd_parts = command.split()
            
            # Check if the command exists
            if shutil.which(cmd_parts[0]) is None:
                if not ignore_errors:
                    self.logger.error(f"❌ Command not found: {cmd_parts[0]}")
                return False
            
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
        """Deploy built assets to web/static directory (never overwrite web/index.html)."""
        try:
            if not hasattr(self, 'build_output_dir') or not self.build_output_dir.exists():
                self.logger.error("No build output directory found")
                return False
            # Ensure deploy directory exists
            self.deploy_dir.mkdir(parents=True, exist_ok=True)
            # Clear existing deployment (static only)
            for item in self.deploy_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    self.logger.warning(f"Failed to remove {item}: {e}")
            # Copy built assets (never touch web/index.html)
            deployed_files = 0
            for item in self.build_output_dir.iterdir():
                try:
                    if item.is_file():
                        shutil.copy2(item, self.deploy_dir)
                        deployed_files += 1
                    elif item.is_dir():
                        shutil.copytree(item, self.deploy_dir / item.name)
                        deployed_files += len(list(item.rglob("*")))
                except Exception as e:
                    self.logger.warning(f"Failed to deploy {item}: {e}")
            self.logger.info(f"✅ Deployed {deployed_files} files to {self.deploy_dir} (web/index.html never overwritten)")
            # Verify index.html exists in static
            index_file = self.deploy_dir / "index.html"
            css_file = self.deploy_dir / "static" / "css" / "index.css"
            if not index_file.exists() and css_file.exists():
                shutil.copy2(css_file, index_file)
            if index_file.exists():
                self.logger.info("✅ index.html successfully deployed to static")
                return True
            else:
                self.logger.warning("⚠️ No index.html found in build output (static)")
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
        self.jarvis_model = JarvisModel()
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
        # Gebruik nu server/web/static en server/web
        self._static_dir = (base_path / "server" / "web" / "static").resolve()
        self._web_dir = (base_path / "server" / "web").resolve()
        
        # Create directories if they don't exist
        self._static_dir.mkdir(parents=True, exist_ok=True)
        self._web_dir.mkdir(parents=True, exist_ok=True)
        
        # Log the paths for debugging
        self.logger.info(f"Static directory: {self._static_dir}")
        self.logger.info(f"Web directory: {self._web_dir}")
    
    @asynccontextmanager
    async def _lifespan(self, _):
        """Handle application startup and shutdown."""
        self.logger.info("🚀 J.A.R.V.I.S. server startup initiated")
        
        # Auto-build and deploy frontend
        build_success = await self.frontend_builder.auto_build_and_deploy()
        if build_success:
            self.logger.info("✅ Frontend build and deployment successful")
        else:
            self.logger.warning("⚠️ Frontend build failed, maar de server gaat door")
        
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

        # Redirect /style.css to /web/style.css for compatibility
        @self.app.get("/style.css")
        async def style_css_redirect():
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/web/style.css")

        # Serve /script.js from /web/script.js for root-level requests
        @self.app.get("/script.js")
        async def script_js():
            script_path = self._web_dir / "script.js"
            if script_path.exists():
                return FileResponse(script_path, media_type="application/javascript")
            return JSONResponse({"detail": "script.js not found"}, status_code=404)

        # Serve /favicon.png from /web/favicon.png for root-level requests
        @self.app.get("/favicon.png")
        async def favicon_png():
            favicon_path = self._web_dir / "favicon.png"
            if favicon_path.exists():
                return FileResponse(favicon_path, media_type="image/png")
            # Return 204 No Content if favicon is missing
            return Response(status_code=204)

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
                "frontend_deployed": (self._static_dir / "index.html").exists(),
                "frontend_directory_exists": self.frontend_builder.frontend_dir.exists(),
                "build_capabilities": {
                    "npm": shutil.which("npm") is not None,
                    "yarn": shutil.which("yarn") is not None,
                    "node": shutil.which("node") is not None
                }
            }
        
        # Manual rebuild endpoint
        @self.app.post("/api/rebuild-frontend")
        async def rebuild_frontend():
            try:
                self.logger.info("🔄 Manual frontend rebuild requested")
                success = await self.frontend_builder.auto_build_and_deploy()
                return {
                    "success": success,
                    "message": "Frontend rebuild completed successfully" if success else "Frontend rebuild failed - check logs"
                }
            except Exception as e:
                self.logger.error(f"Manual rebuild failed: {e}")
                return {
                    "success": False,
                    "message": f"Frontend rebuild failed: {str(e)}"
                }
        
        # Chat endpoint voor frontend
        @self.app.post("/api/chat")
        async def chat_endpoint(request: Request):
            data = await request.json()
            message = data.get("message")
            if not message:
                return JSONResponse({"success": False, "error": "No message provided"}, status_code=400)
            # Verwerk via JarvisModel
            result = self.jarvis_model.process_input(message, user_id="frontend")
            return JSONResponse(result)
    
    def _setup_static_files(self):
        """Configure static file serving."""
        # Always mount static directory (it's created in _setup_static_paths)
        self.app.mount("/static", StaticFiles(directory=str(self._static_dir)), name="static")
        self.logger.info(f"📁 Mounted static directory: {self._static_dir}")

        # Mount web directory for other assets and direct CSS access
        if self._directory_has_files(self._web_dir):
            self.app.mount("/assets", StaticFiles(directory=str(self._web_dir)), name="web-assets")
            self.logger.info(f"📁 Mounted web assets directory: {self._web_dir}")
            # Also mount /web for direct access to style.css and others
            self.app.mount("/web", StaticFiles(directory=str(self._web_dir)), name="web")
            self.logger.info(f"📁 Mounted /web for direct access to style.css and other files")
    
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
        # Priority: server/web/index.html first, then server/web/static/index.html
        web_index = self._web_dir / "index.html"
        static_index = self._static_dir / "index.html"

        if web_index.exists():
            self.logger.info(f"*** DEBUG: Serving server/web/index.html: {web_index}")
            return FileResponse(
                web_index,
                media_type="text/html",
                headers={"Cache-Control": "no-cache"}
            )

        if static_index.exists():
            self.logger.info(f"*** DEBUG: Serving server/web/static/index.html: {static_index}")
            return FileResponse(
                static_index,
                media_type="text/html",
                headers={"Cache-Control": "no-cache"}
            )

        # Fallback: show status page when no index.html is found
        self.logger.warning("⚠️ No index.html found - serving status page")
        return await self._serve_status_page()
    
    async def _serve_status_page(self) -> HTMLResponse:
        """Serve a status page when index.html is missing."""
        npm_available = shutil.which("npm") is not None
        yarn_available = shutil.which("yarn") is not None
        node_available = shutil.which("node") is not None
        frontend_exists = self.frontend_builder.frontend_dir.exists()
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>J.A.R.V.I.S. Status</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #0f1419 0%, #1a1f29 100%);
                    color: #e8eaed;
                    margin: 0;
                    padding: 0;
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .container {{
                    max-width: 800px;
                    text-align: center;
                    background: rgba(255, 255, 255, 0.05);
                    padding: 3rem;
                    border-radius: 20px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}
                h1 {{
                    font-size: 3rem;
                    margin-bottom: 1rem;
                    background: linear-gradient(45deg, #00d4ff, #0099cc);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    animation: glow 2s ease-in-out infinite alternate;
                }}
                @keyframes glow {{
                    from {{ text-shadow: 0 0 20px rgba(0, 212, 255, 0.5); }}
                    to {{ text-shadow: 0 0 30px rgba(0, 212, 255, 0.8); }}
                }}
                .status {{
                    font-size: 1.2rem;
                    margin: 2rem 0;
                    padding: 1rem;
                    background: rgba(255, 0, 0, 0.1);
                    border-radius: 10px;
                    border: 1px solid rgba(255, 0, 0, 0.3);
                }}
                .info-box {{
                    background: rgba(255, 255, 255, 0.03);
                    padding: 1.5rem;
                    border-radius: 10px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    margin: 1rem 0;
                }}
                .info-box h3 {{
                    color: #00d4ff;
                    margin-top: 0;
                }}
                .rebuild-btn {{
                    background: linear-gradient(45deg, #ff4d4d, #ff0000);
                    color: white;
                    border: none;
                    padding: 1rem 2rem;
                    border-radius: 10px;
                    cursor: pointer;
                    font-size: 1rem;
                    margin: 1rem 0;
                    transition: transform 0.2s;
                }}
                .rebuild-btn:hover {{
                    transform: scale(1.05);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 J.A.R.V.I.S.</h1>
                <div class="status">
                    ❌ No frontend build found
                </div>
                <div class="info-box">
                    <h3>Build Capabilities</h3>
                    <ul>
                        <li>NPM available: {'✅' if npm_available else '❌'}</li>
                        <li>Yarn available: {'✅' if yarn_available else '❌'}</li>
                        <li>Node.js available: {'✅' if node_available else '❌'}</li>
                        <li>Frontend directory exists: {'✅' if frontend_exists else '❌'}</li>
                    </ul>
                </div>
                <div class="info-box">
                    <h3>Quick Start</h3>
                    <ul>
                        <li>Create a <code>frontend/</code> directory with a valid <code>package.json</code></li>
                        <li>Use the rebuild button below to trigger a build</li>
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
                async function rebuildFrontend() {{
                    var btn = document.querySelector('.rebuild-btn');
                    btn.disabled = true;
                    btn.textContent = '🔨 Building...';
                    try {{
                        const response = await fetch('/api/rebuild-frontend', {{ method: 'POST' }});
                        const result = await response.json();
                        if (result.success) {{
                            btn.textContent = '✅ Build Complete!';
                            setTimeout(() => location.reload(), 2000);
                        }} else {{
                            btn.textContent = '❌ Build Failed';
                            setTimeout(() => {{
                                btn.textContent = '🔨 Manual Rebuild Frontend';
                                btn.disabled = false;
                            }}, 3000);
                        }}
                    }} catch (error) {{
                        btn.textContent = '❌ Error';
                        setTimeout(() => {{
                            btn.textContent = '🔨 Manual Rebuild Frontend';
                            btn.disabled = false;
                        }}, 3000);
                    }}
                }}
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
    import os
    # Ensure working directory is project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("[JARVIS] Starting server with: python -m server.app (no reloader, safe for Windows/PowerShell)")
    print("[JARVIS] Server will be available at: http://127.0.0.1:8080\n")
    
    uvicorn.run(
        "server.app:app",  # Always use full module path
        host="127.0.0.1",
        port=8080,
        reload=False,  # Disable reloader for compatibility
        log_level="info"
    )