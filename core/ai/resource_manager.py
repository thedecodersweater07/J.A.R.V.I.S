"""
AI Resource Management Module
Optimizes allocation of computing resources for AI components.
"""

import os
import sys
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Tuple
import psutil
import torch

# Import core components
from core.logging import get_logger

class ResourceManager:
    """
    Resource manager for AI components in JARVIS.
    Handles resource allocation, monitoring, and optimization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Resource Manager.
        
        Args:
            config: Configuration dictionary for the resource manager
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Resource limits
        self.memory_limit = self._parse_memory_limit(self.config.get("memory_limit", "80%"))
        self.cpu_limit = self.config.get("cpu_limit", 0.8)  # 80% by default
        
        # Resource allocation
        self.allocations = {}
        self.allocation_lock = threading.Lock()
        
        # Resource monitoring
        self.monitoring_interval = self.config.get("monitoring_interval", 5.0)  # seconds
        self.monitoring_thread = None
        self.running = False
        
        # GPU support
        self.gpu_enabled = self.config.get("gpu_enabled", torch.cuda.is_available())
        self.gpu_memory_limit = self.config.get("gpu_memory_limit", 0.8)  # 80% by default
        
        # Initialize monitoring
        self._init_monitoring()
        
    def _parse_memory_limit(self, limit: str) -> int:
        """
        Parse memory limit string to bytes.
        
        Args:
            limit: Memory limit string (e.g., "80%", "8G", "8000M")
            
        Returns:
            Memory limit in bytes
        """
        if isinstance(limit, int):
            return limit
            
        if isinstance(limit, str):
            # Percentage of total memory
            if limit.endswith("%"):
                percentage = float(limit.rstrip("%")) / 100.0
                return int(psutil.virtual_memory().total * percentage)
                
            # Absolute value with unit
            unit_multipliers = {
                "K": 1024,
                "M": 1024 * 1024,
                "G": 1024 * 1024 * 1024,
                "T": 1024 * 1024 * 1024 * 1024
            }
            
            for unit, multiplier in unit_multipliers.items():
                if limit.upper().endswith(unit):
                    return int(float(limit[:-1]) * multiplier)
                    
            # No unit, assume bytes
            return int(limit)
            
        # Default: 80% of total memory
        return int(psutil.virtual_memory().total * 0.8)
        
    def _init_monitoring(self):
        """Initialize resource monitoring."""
        if self.config.get("enable_monitoring", True):
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self.monitoring_thread.start()
            self.logger.debug("Resource monitoring started")
            
    def _monitor_resources(self):
        """Monitor system resources (runs in a separate thread)."""
        while self.running:
            try:
                # Get current resource usage
                cpu_percent = psutil.cpu_percent(interval=None) / 100.0
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent / 100.0
                
                # Check if we're over limits
                if cpu_percent > self.cpu_limit:
                    self.logger.warning(f"CPU usage ({cpu_percent:.2%}) exceeds limit ({self.cpu_limit:.2%})")
                    self._handle_resource_pressure("cpu", cpu_percent)
                    
                if memory_info.used > self.memory_limit:
                    self.logger.warning(f"Memory usage ({memory_info.used / 1024**3:.2f} GB) exceeds limit ({self.memory_limit / 1024**3:.2f} GB)")
                    self._handle_resource_pressure("memory", memory_info.used)
                    
                # Check GPU if enabled
                if self.gpu_enabled and torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        gpu_memory = torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory
                        if gpu_memory > self.gpu_memory_limit:
                            self.logger.warning(f"GPU {i} memory usage ({gpu_memory:.2%}) exceeds limit ({self.gpu_memory_limit:.2%})")
                            self._handle_resource_pressure("gpu", gpu_memory, device=i)
                            
            except Exception as e:
                self.logger.error(f"Error monitoring resources: {e}", exc_info=True)
                
            # Sleep for monitoring interval
            time.sleep(self.monitoring_interval)
            
    def _handle_resource_pressure(self, resource_type: str, current_usage: float, device: int = None):
        """
        Handle resource pressure by reducing allocations.
        
        Args:
            resource_type: Type of resource under pressure
            current_usage: Current usage of the resource
            device: Device ID for GPU resources
        """
        with self.allocation_lock:
            # Sort allocations by priority (lower is higher priority)
            sorted_allocations = sorted(
                self.allocations.items(),
                key=lambda x: x[1].get("priority", 5)
            )
            
            # Reduce allocations starting with lowest priority
            for component_id, allocation in reversed(sorted_allocations):
                if resource_type == "cpu" and allocation.get("cpu_intensive", False):
                    self.logger.info(f"Reducing CPU allocation for {component_id}")
                    self._notify_component(component_id, "reduce_cpu")
                    
                elif resource_type == "memory" and allocation.get("memory_intensive", False):
                    self.logger.info(f"Reducing memory allocation for {component_id}")
                    self._notify_component(component_id, "reduce_memory")
                    
                elif resource_type == "gpu" and allocation.get("gpu_intensive", False):
                    if device is None or allocation.get("gpu_device") == device:
                        self.logger.info(f"Reducing GPU allocation for {component_id}")
                        self._notify_component(component_id, "reduce_gpu", device=device)
                        
    def _notify_component(self, component_id: str, action: str, **kwargs):
        """
        Notify a component of a resource action.
        
        Args:
            component_id: ID of the component to notify
            action: Action to take
            **kwargs: Additional action parameters
        """
        allocation = self.allocations.get(component_id)
        if not allocation:
            return
            
        callback = allocation.get("callback")
        if callback:
            try:
                callback(action, **kwargs)
            except Exception as e:
                self.logger.error(f"Error notifying component {component_id}: {e}", exc_info=True)
                
    def allocate_resources(self, component_id: str, allocation: Dict[str, Any]) -> bool:
        """
        Allocate resources for a component.
        
        Args:
            component_id: ID of the component requesting resources
            allocation: Resource allocation request
            
        Returns:
            True if allocation successful, False otherwise
        """
        with self.allocation_lock:
            # Check if we have enough resources
            cpu_request = allocation.get("cpu", 0.0)
            memory_request = self._parse_memory_limit(allocation.get("memory", "0"))
            gpu_request = allocation.get("gpu", 0.0)
            gpu_device = allocation.get("gpu_device", 0)
            
            # Calculate current allocations
            current_cpu = sum(a.get("cpu", 0.0) for a in self.allocations.values())
            current_memory = sum(self._parse_memory_limit(a.get("memory", "0")) for a in self.allocations.values())
            
            # Check CPU availability
            if current_cpu + cpu_request > self.cpu_limit:
                self.logger.warning(f"CPU allocation request from {component_id} exceeds available resources")
                return False
                
            # Check memory availability
            if current_memory + memory_request > self.memory_limit:
                self.logger.warning(f"Memory allocation request from {component_id} exceeds available resources")
                return False
                
            # Check GPU availability if requested
            if gpu_request > 0 and self.gpu_enabled and torch.cuda.is_available():
                if gpu_device >= torch.cuda.device_count():
                    self.logger.warning(f"GPU device {gpu_device} requested by {component_id} does not exist")
                    return False
                    
                # Calculate current GPU allocations for this device
                current_gpu = sum(
                    a.get("gpu", 0.0) 
                    for a in self.allocations.values() 
                    if a.get("gpu_device", 0) == gpu_device
                )
                
                if current_gpu + gpu_request > self.gpu_memory_limit:
                    self.logger.warning(f"GPU allocation request from {component_id} exceeds available resources")
                    return False
                    
            # Store allocation
            self.allocations[component_id] = allocation
            self.logger.info(f"Resources allocated for {component_id}")
            return True
            
    def release_resources(self, component_id: str):
        """
        Release resources allocated to a component.
        
        Args:
            component_id: ID of the component releasing resources
        """
        with self.allocation_lock:
            if component_id in self.allocations:
                del self.allocations[component_id]
                self.logger.info(f"Resources released for {component_id}")
                
    def get_available_resources(self) -> Dict[str, Any]:
        """
        Get available system resources.
        
        Returns:
            Dictionary of available resources
        """
        # Get current system resources
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=None) / 100.0
        memory_info = psutil.virtual_memory()
        
        # Calculate allocated resources
        allocated_cpu = sum(a.get("cpu", 0.0) for a in self.allocations.values())
        allocated_memory = sum(self._parse_memory_limit(a.get("memory", "0")) for a in self.allocations.values())
        
        # Calculate available resources
        available_cpu = max(0.0, self.cpu_limit - cpu_percent)
        available_memory = max(0, self.memory_limit - memory_info.used)
        
        # GPU resources
        gpu_resources = []
        if self.gpu_enabled and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = props.total_memory
                
                gpu_resources.append({
                    "device": i,
                    "name": props.name,
                    "total_memory": total,
                    "allocated_memory": allocated,
                    "reserved_memory": reserved,
                    "available_memory": total - allocated,
                    "utilization": allocated / total
                })
                
        return {
            "cpu": {
                "total": cpu_count,
                "used_percent": cpu_percent,
                "allocated": allocated_cpu,
                "available": available_cpu
            },
            "memory": {
                "total": memory_info.total,
                "used": memory_info.used,
                "available": available_memory,
                "allocated": allocated_memory
            },
            "gpu": gpu_resources
        }
        
    def shutdown(self):
        """Shut down the resource manager."""
        self.running = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
            
        self.logger.debug("Resource manager shut down")
