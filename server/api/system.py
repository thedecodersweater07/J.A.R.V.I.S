"""
System API Routes
Handles system status and management endpoints
"""
import os
import logging
import platform
import psutil
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status

# Import dependencies for auth
from .dependencies import get_current_active_user, get_admin_user

# Setup logging
logger = logging.getLogger("jarvis-server.system")

# Create router
router = APIRouter(
    prefix="/system",
    tags=["system"],
    responses={404: {"description": "Not found"}},
)

# Routes
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@router.get("/status")
async def system_status(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Get system status"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get AI components status
        from ..app import get_ai_components
        llm_core, model_manager, nlp_processor = get_ai_components()
        
        return {
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            },
            "components": {
                "llm_available": llm_core is not None,
                "ml_available": model_manager is not None,
                "nlp_available": nlp_processor is not None
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.post("/restart")
async def restart_system(current_user: Dict[str, Any] = Depends(get_admin_user)):
    """Restart system (admin only)"""
    logger.warning(f"System restart requested by {current_user['username']}")
    
    # In a real system, you would implement actual restart logic here
    return {
        "status": "restart_initiated",
        "message": "System restart has been initiated",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/logs")
async def get_logs(
    lines: int = 100,
    current_user: Dict[str, Any] = Depends(get_admin_user)
):
    """Get system logs (admin only)"""
    try:
        log_file = os.path.join('logs', 'server.log')
        
        if not os.path.exists(log_file):
            return {
                "status": "error",
                "error": "Log file not found",
                "timestamp": datetime.now().isoformat()
            }
        
        # Read the last N lines of the log file
        with open(log_file, 'r') as f:
            log_lines = f.readlines()
            log_lines = log_lines[-lines:] if len(log_lines) > lines else log_lines
        
        return {
            "status": "ok",
            "log_lines": log_lines,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reading logs: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
