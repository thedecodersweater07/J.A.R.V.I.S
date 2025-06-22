import requests
import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class APIClient:
    """
    Client for interacting with the JARVIS API Server
    Provides methods for authentication and AI requests
    """
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.token = None
        self.user_info = None
        self.is_authenticated = False
        self.callbacks = {
            "on_response": [],
            "on_error": [],
            "on_auth_change": []
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'JARVIS-UI/1.0',
            'Accept': 'application/json'
        })
        logger.debug(f"APIClient initialized with base URL: {base_url}")
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate with the API server
        Returns user info on success
        """
        try:
            response = self.session.post(
                f"{self.base_url}/token",
                data={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                self.user_info = {
                    "id": data["user_id"],
                    "username": data["username"],
                    "role": data["role"]
                }
                self.is_authenticated = True
                self._update_auth_headers()
                self._notify_callbacks("on_auth_change", True)
                return self.user_info
            else:
                error_msg = response.json().get("detail", "Authentication failed")
                logger.error(f"Login failed: {error_msg}")
                self._notify_callbacks("on_error", error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Login request error: {str(e)}"
            logger.error(error_msg)
            self._notify_callbacks("on_error", error_msg)
            return {"error": error_msg}
    
    def logout(self):
        """Clear authentication state"""
        self.token = None
        self.user_info = None
        self.is_authenticated = False
        self.session = requests.Session()  # Reset session
        self._notify_callbacks("on_auth_change", False)
    
    def _update_auth_headers(self):
        """Update session headers with auth token"""
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get current user info"""
        if not self.is_authenticated:
            return {"error": "Not authenticated"}
        
        try:
            response = self.session.get(f"{self.base_url}/users/me")
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = response.json().get("detail", "Failed to get user info")
                logger.error(f"Get user info failed: {error_msg}")
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"User info request error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def query_ai(self, query: str, request_type: str = "text", 
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send query to AI
        request_type: "text", "nlp", "ml", or "full"
        """
        if not self.is_authenticated:
            error_msg = "Not authenticated"
            self._notify_callbacks("on_error", error_msg)
            return {"error": error_msg}
        
        try:
            payload = {
                "query": query,
                "request_type": request_type,
                "context": context or {}
            }
            
            response = self.session.post(
                f"{self.base_url}/ai/query",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                self._notify_callbacks("on_response", result)
                return result
            else:
                error_msg = response.json().get("detail", "AI query failed")
                logger.error(f"AI query failed: {error_msg}")
                self._notify_callbacks("on_error", error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"AI query request error: {str(e)}"
            logger.error(error_msg)
            self._notify_callbacks("on_error", error_msg)
            return {"error": error_msg}
    
    def async_query_ai(self, query: str, request_type: str = "text",
                      context: Optional[Dict[str, Any]] = None,
                      callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Send query to AI asynchronously
        Calls the callback with the result when complete
        """
        def _async_query():
            result = self.query_ai(query, request_type, context)
            if callback:
                callback(result)
        
        thread = threading.Thread(target=_async_query)
        thread.daemon = True
        thread.start()
        return thread
    
    def check_health(self) -> Dict[str, Any]:
        """Check API server health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = "Health check failed"
                logger.error(error_msg)
                return {"status": "unhealthy", "error": error_msg}
                
        except Exception as e:
            error_msg = f"Health check error: {str(e)}"
            logger.error(error_msg)
            return {"status": "unhealthy", "error": error_msg}
    
    def add_callback(self, event_type: str, callback: Callable):
        """
        Add callback for events
        event_type: "on_response", "on_error", or "on_auth_change"
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: Callable):
        """Remove callback"""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    def _notify_callbacks(self, event_type: str, data: Any):
        """Notify all callbacks for an event"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in {event_type} callback: {e}")
