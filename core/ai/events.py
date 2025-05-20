"""
AI Event System Module
Provides a publish-subscribe pattern for AI component communication.
"""

import logging
import threading
from typing import Dict, Any, List, Optional, Callable, Set
import time
import queue
import uuid

# Import core components
from core.logging import get_logger

class EventBus:
    """
    Event bus for AI component communication using the publish-subscribe pattern.
    Allows components to communicate without direct dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Event Bus.
        
        Args:
            config: Configuration dictionary for the event bus
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.subscribers = {}
        self.event_history = {}
        self.max_history = self.config.get("max_history", 100)
        self.async_mode = self.config.get("async_mode", True)
        self.event_queue = queue.Queue()
        self.running = False
        self.event_thread = None
        
        # Start event processing thread if in async mode
        if self.async_mode:
            self._start_event_thread()
            
    def _start_event_thread(self):
        """Start the event processing thread."""
        if self.event_thread is not None and self.event_thread.is_alive():
            return
            
        self.running = True
        self.event_thread = threading.Thread(target=self._process_events, daemon=True)
        self.event_thread.start()
        self.logger.debug("Event processing thread started")
        
    def _process_events(self):
        """Process events from the queue (runs in a separate thread)."""
        while self.running:
            try:
                event = self.event_queue.get(timeout=0.1)
                self._dispatch_event(event)
                self.event_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error processing event: {e}", exc_info=True)
                
    def _dispatch_event(self, event: Dict[str, Any]):
        """Dispatch an event to subscribers with validation."""
        try:
            if not isinstance(event, dict) or 'type' not in event:
                self.logger.error("Invalid event format")
                return
                
            event_type = event["type"]
            if not event_type:
                self.logger.error("Empty event type")
                return
                
            # Store in history
            if event_type not in self.event_history:
                self.event_history[event_type] = []
                
            history = self.event_history[event_type]
            history.append(event)
            
            # Trim history if needed
            if len(history) > self.max_history:
                history.pop(0)
                
            # Notify subscribers
            if event_type in self.subscribers:
                for callback in self.subscribers[event_type]:
                    try:
                        callback(event)
                    except Exception as e:
                        self.logger.error(f"Error in event subscriber: {e}", exc_info=True)
                        
        except Exception as e:
            self.logger.error(f"Error dispatching event: {e}", exc_info=True)
            
    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs
            
        Returns:
            Subscription ID
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = set()
            
        self.subscribers[event_type].add(callback)
        subscription_id = str(uuid.uuid4())
        
        self.logger.debug(f"Added subscriber to event type '{event_type}'")
        return subscription_id
        
    def unsubscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to unsubscribe
        """
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            self.logger.debug(f"Removed subscriber from event type '{event_type}'")
            
            # Clean up empty subscriber sets
            if not self.subscribers[event_type]:
                del self.subscribers[event_type]
                
    def publish(self, event_type: str, data: Dict[str, Any] = None):
        """
        Publish an event.
        
        Args:
            event_type: Type of event to publish
            data: Data to include with the event
        """
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "id": str(uuid.uuid4()),
            "data": data or {}
        }
        
        if self.async_mode:
            # Add to queue for async processing
            self.event_queue.put(event)
        else:
            # Process synchronously
            self._dispatch_event(event)
            
        self.logger.debug(f"Published event of type '{event_type}'")
        
    def get_event_history(self, event_type: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get event history for a specific type or all types.
        
        Args:
            event_type: Optional type of events to retrieve
            limit: Optional limit on number of events to retrieve
            
        Returns:
            List of events
        """
        if event_type:
            history = self.event_history.get(event_type, [])
            if limit:
                return history[-limit:]
            return history.copy()
        else:
            # Combine all event types
            all_events = []
            for events in self.event_history.values():
                all_events.extend(events)
                
            # Sort by timestamp
            all_events.sort(key=lambda e: e["timestamp"])
            
            if limit:
                return all_events[-limit:]
            return all_events
            
    def clear_history(self, event_type: str = None):
        """
        Clear event history.
        
        Args:
            event_type: Optional type of events to clear
        """
        if event_type:
            if event_type in self.event_history:
                self.event_history[event_type] = []
                self.logger.debug(f"Cleared history for event type '{event_type}'")
        else:
            self.event_history = {}
            self.logger.debug("Cleared all event history")
            
    def shutdown(self):
        """Shut down the event bus."""
        self.running = False
        
        if self.event_thread and self.event_thread.is_alive():
            self.event_thread.join(timeout=1.0)
            
        self.logger.debug("Event bus shut down")
