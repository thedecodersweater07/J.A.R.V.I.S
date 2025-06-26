"""
REST API for Bluetooth health monitoring.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, cast

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directory to path to allow package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import BluetoothHealthMonitor with proper error handling
try:
    from bluetooth_health.core import BluetoothHealthMonitor
except ImportError as e:
    # For direct script execution
    try:
        from core import BluetoothHealthMonitor
    except ImportError:
        raise ImportError(
            "Could not import BluetoothHealthMonitor. "
            "Make sure the package is properly installed."
        ) from e

app = FastAPI(title="Bluetooth Health API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active connections and monitor instances
active_connections: List[WebSocket] = []
monitor: Optional[BluetoothHealthMonitor] = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logging.error(f"Error broadcasting message: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    logging.basicConfig(level=logging.INFO)
    logging.info("Bluetooth Health API started")

@app.on_event("shutdown")
async def shutdown_event():
    global monitor
    if monitor:
        await monitor.stop_heart_rate_monitoring()
        await monitor.disconnect()
        monitor = None
    logging.info("Bluetooth Health API stopped")

@app.get("/api/devices")
async def list_devices():
    """List available Bluetooth devices."""
    global monitor
    if not monitor:
        monitor = BluetoothHealthMonitor()
    
    try:
        devices = await monitor.discover_devices()
        return {"devices": [{"name": name, "address": addr} for name, addr in devices.items()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time health data."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def on_heart_rate(rate: int):
    """Callback for heart rate updates."""
    message = json.dumps({"type": "heart_rate", "data": rate})
    await manager.broadcast(message)

@app.post("/api/connect")
async def connect_device(device: dict):
    """Connect to a Bluetooth device."""
    global monitor
    
    if not device.get("address"):
        raise HTTPException(status_code=400, detail="Device address is required")
    
    try:
        if not monitor:
            monitor = BluetoothHealthMonitor(device.get("name"), device["address"])
        
        connected = await monitor.connect()
        if not connected:
            raise HTTPException(status_code=500, detail="Failed to connect to device")
            
        return {"status": "connected", "device": device}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start-monitoring")
async def start_monitoring():
    """Start monitoring health data."""
    global monitor
    
    if not monitor or not monitor.connected:
        raise HTTPException(status_code=400, detail="Not connected to any device")
        
    try:
        await monitor.start_heart_rate_monitoring(on_heart_rate)
        return {"status": "monitoring_started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stop-monitoring")
async def stop_monitoring():
    """Stop monitoring health data."""
    global monitor
    
    if not monitor:
        return {"status": "not_monitoring"}
        
    try:
        await monitor.stop_heart_rate_monitoring()
        return {"status": "monitoring_stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/disconnect")
async def disconnect_device():
    """Disconnect from the current device."""
    global monitor
    
    if not monitor:
        return {"status": "not_connected"}
        
    try:
        await monitor.disconnect()
        monitor = None
        return {"status": "disconnected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)
