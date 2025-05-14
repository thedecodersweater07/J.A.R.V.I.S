"""
AI Pipeline Orchestration Module
Standardizes the flow of data between AI components in JARVIS.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union
import time
import uuid

# Import core components
from core.logging import get_logger

class Pipeline:
    """
    Pipeline for processing data through a series of AI components.
    Manages the flow of data and handles errors and fallbacks.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize a new pipeline.
        
        Args:
            name: Name of the pipeline
            config: Configuration dictionary for the pipeline
        """
        self.logger = get_logger(__name__)
        self.name = name
        self.config = config or {}
        self.steps = []
        self.error_handlers = {}
        self.fallbacks = {}
        self.metrics = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "average_runtime": 0
        }
        
    def add_step(self, name: str, processor: Callable, config: Dict[str, Any] = None):
        """
        Add a processing step to the pipeline.
        
        Args:
            name: Name of the step
            processor: Function or callable to process data
            config: Configuration for the step
        """
        step = {
            "name": name,
            "processor": processor,
            "config": config or {},
            "metrics": {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "average_runtime": 0
            }
        }
        
        self.steps.append(step)
        self.logger.debug(f"Added step '{name}' to pipeline '{self.name}'")
        
    def add_error_handler(self, step_name: str, handler: Callable):
        """
        Add an error handler for a specific step.
        
        Args:
            step_name: Name of the step to handle errors for
            handler: Function to handle errors
        """
        self.error_handlers[step_name] = handler
        self.logger.debug(f"Added error handler for step '{step_name}' in pipeline '{self.name}'")
        
    def add_fallback(self, step_name: str, fallback: Callable):
        """
        Add a fallback processor for a specific step.
        
        Args:
            step_name: Name of the step to add fallback for
            fallback: Function to use as fallback
        """
        self.fallbacks[step_name] = fallback
        self.logger.debug(f"Added fallback for step '{step_name}' in pipeline '{self.name}'")
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data through the pipeline.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        start_time = time.time()
        run_id = str(uuid.uuid4())
        self.metrics["total_runs"] += 1
        
        # Initialize pipeline context
        context = {
            "pipeline_name": self.name,
            "run_id": run_id,
            "start_time": start_time,
            "steps_completed": [],
            "steps_failed": [],
            "current_step": None
        }
        
        # Add context to data
        data["_context"] = context
        
        self.logger.debug(f"Starting pipeline '{self.name}' run {run_id}")
        
        # Process each step
        for step in self.steps:
            step_name = step["name"]
            processor = step["processor"]
            step_config = step["config"]
            
            # Update context
            context["current_step"] = step_name
            
            try:
                # Measure step execution time
                step_start = time.time()
                
                # Process data
                self.logger.debug(f"Executing step '{step_name}' in pipeline '{self.name}'")
                data = processor(data, step_config)
                
                # Update step metrics
                step_runtime = time.time() - step_start
                step["metrics"]["total_calls"] += 1
                step["metrics"]["successful_calls"] += 1
                step["metrics"]["average_runtime"] = (
                    (step["metrics"]["average_runtime"] * (step["metrics"]["successful_calls"] - 1) + step_runtime) / 
                    step["metrics"]["successful_calls"]
                )
                
                # Update context
                context["steps_completed"].append(step_name)
                
            except Exception as e:
                # Log error
                self.logger.error(f"Error in pipeline '{self.name}' step '{step_name}': {e}", exc_info=True)
                
                # Update step metrics
                step["metrics"]["total_calls"] += 1
                step["metrics"]["failed_calls"] += 1
                
                # Update context
                context["steps_failed"].append(step_name)
                
                # Handle error
                handled = False
                
                # Try step-specific error handler
                if step_name in self.error_handlers:
                    try:
                        handler = self.error_handlers[step_name]
                        data = handler(data, e, step_config)
                        handled = True
                        self.logger.debug(f"Error in step '{step_name}' handled successfully")
                    except Exception as handler_error:
                        self.logger.error(f"Error handler for step '{step_name}' failed: {handler_error}", exc_info=True)
                
                # Try step-specific fallback
                if not handled and step_name in self.fallbacks:
                    try:
                        fallback = self.fallbacks[step_name]
                        data = fallback(data, step_config)
                        handled = True
                        self.logger.debug(f"Fallback for step '{step_name}' executed successfully")
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback for step '{step_name}' failed: {fallback_error}", exc_info=True)
                
                # If error not handled, raise it
                if not handled:
                    self.metrics["failed_runs"] += 1
                    raise
        
        # Calculate total runtime
        total_runtime = time.time() - start_time
        
        # Update pipeline metrics
        self.metrics["successful_runs"] += 1
        self.metrics["average_runtime"] = (
            (self.metrics["average_runtime"] * (self.metrics["successful_runs"] - 1) + total_runtime) / 
            self.metrics["successful_runs"]
        )
        
        # Update context
        context["end_time"] = time.time()
        context["total_runtime"] = total_runtime
        context["status"] = "completed"
        
        self.logger.debug(f"Pipeline '{self.name}' completed in {total_runtime:.2f}s")
        
        return data
        
class PipelineManager:
    """
    Manager for AI pipelines in JARVIS.
    Handles pipeline creation, registration, and execution.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Pipeline Manager.
        
        Args:
            config: Configuration dictionary for the manager
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.pipelines = {}
        
    def create_pipeline(self, name: str, config: Dict[str, Any] = None) -> Pipeline:
        """
        Create a new pipeline.
        
        Args:
            name: Name of the pipeline
            config: Configuration for the pipeline
            
        Returns:
            The created pipeline
        """
        if name in self.pipelines:
            self.logger.warning(f"Pipeline '{name}' already exists, returning existing pipeline")
            return self.pipelines[name]
            
        pipeline = Pipeline(name, config)
        self.pipelines[name] = pipeline
        
        self.logger.info(f"Created pipeline '{name}'")
        return pipeline
        
    def get_pipeline(self, name: str) -> Optional[Pipeline]:
        """
        Get a pipeline by name.
        
        Args:
            name: Name of the pipeline to retrieve
            
        Returns:
            The pipeline or None if not found
        """
        pipeline = self.pipelines.get(name)
        
        if pipeline is None:
            self.logger.warning(f"Pipeline '{name}' not found")
            
        return pipeline
        
    def execute_pipeline(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a pipeline by name.
        
        Args:
            name: Name of the pipeline to execute
            data: Input data for the pipeline
            
        Returns:
            Processed data
        """
        pipeline = self.get_pipeline(name)
        
        if pipeline is None:
            raise ValueError(f"Pipeline '{name}' not found")
            
        return pipeline.process(data)
        
    def get_pipeline_metrics(self, name: str = None) -> Dict[str, Any]:
        """
        Get metrics for a pipeline or all pipelines.
        
        Args:
            name: Optional name of the pipeline to get metrics for
            
        Returns:
            Dictionary of pipeline metrics
        """
        if name:
            pipeline = self.get_pipeline(name)
            
            if pipeline is None:
                return {}
                
            return {
                "pipeline": pipeline.name,
                "metrics": pipeline.metrics,
                "steps": [
                    {
                        "name": step["name"],
                        "metrics": step["metrics"]
                    }
                    for step in pipeline.steps
                ]
            }
        else:
            return {
                name: {
                    "metrics": pipeline.metrics,
                    "steps_count": len(pipeline.steps)
                }
                for name, pipeline in self.pipelines.items()
            }
