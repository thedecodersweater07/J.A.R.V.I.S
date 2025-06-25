from typing import Dict, Any, List, Optional
import logging
import asyncio
import numpy as np
import torch
import json
import gzip
import os
import time
import random
from datetime import datetime
from pathlib import Path
from core.logging.advanced_logger import AdvancedLogger, CustomJSONEncoder
from data.processing.virtual_env_summarizer import VirtualEnvSummarizer

logger = AdvancedLogger("virtual_environment").get_logger()

class VirtualEnvironment:
    """Enhanced virtual environment with quantum processing"""
    def __init__(self, env_type: str, params: Dict[str, Any]):
        self.type = env_type
        self.params = params
        self.state = {}
        self.history = []
        self.data_path = params.get('data_path')
        self.env_path = params.get('env_path')
        self.quantum_enabled = params.get('quantum_enabled', False)
        
        # Create necessary directories
        for path in [self.data_path, self.env_path]:
            if path is not None:
                os.makedirs(path, exist_ok=True)

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute step with quantum processing"""
        # Add artificial delay for quantum simulation
        if self.quantum_enabled:
            delay = random.uniform(0.5, 2.0)
            time.sleep(delay)
            
        result = self._process_action(action)
        
        if self.quantum_enabled:
            result['quantum_delay'] = delay
            result['quantum_state'] = self._simulate_quantum_state()
        
        # Call progress callback if provided
        if "parameters" in action and "progress_callback" in action["parameters"]:
            action["parameters"]["progress_callback"](len(self.history) + 1)
            
        self.history.append({"action": action, "result": result})
        return result

    async def _store_step_data(self, action: Dict[str, Any], result: Dict[str, Any]):
        """Store step data with proper serialization"""
        if not self.data_path:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filepath = os.path.join(
            self.data_path,
            f"step_data_{timestamp}.jsonl.gz"
        )

        # Convert numpy arrays and tensors to lists before serializing
        sanitized_data = {
            "timestamp": timestamp,
            "action": self._sanitize_for_json(action),
            "result": self._sanitize_for_json(result),
            "state": self._sanitize_for_json(self.state)
        }

        with gzip.open(filepath, 'at') as f:
            f.write(json.dumps(sanitized_data, cls=CustomJSONEncoder) + '\n')

    def _sanitize_for_json(self, obj: Any) -> Any:
        """Convert numpy arrays and tensors to JSON-serializable format"""
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if isinstance(obj, np.ndarray) else obj.cpu().detach().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        return obj

    def _process_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process an action and return results"""
        if self.type == "knowledge_validation":
            return self._validate_knowledge(action)
        return {"status": "unknown_environment_type"}
        
    def _validate_knowledge(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Validate knowledge through simulation"""
        confidence = np.random.random()  # Simplified validation
        return {
            "confidence_score": confidence,
            "validation_success": confidence > 0.7,
            "timestamp": datetime.utcnow()
        }
        
    def _simulate_quantum_state(self) -> Dict[str, float]:
        """Simulate quantum state evolution"""
        return {
            'coherence': random.random(),
            'entanglement': random.random(),
            'superposition': random.random()
        }

    def _store_results(self, topic: str, results: Dict[str, Any]):
        """Store validation results with proper JSON serialization"""
        if not self.data_path:
            return
        filepath = os.path.join(
            self.data_path, 
            topic, 
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json.gz"
        )
        
        # Sanitize results before storing
        sanitized_results = self._sanitize_for_json(results)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with gzip.open(filepath, 'wt') as f:
            json.dump(sanitized_results, f, cls=CustomJSONEncoder, indent=2)

    async def shutdown(self):
        """Cleanup and generate summary before shutting down"""
        if self.data_path:
            summarizer = VirtualEnvSummarizer(self.data_path)
            csv_path = summarizer.generate_summary()
            if csv_path:
                logger.info(f"Generated summary report: {csv_path}")
        
        # Cleanup code
        self.history.clear()
        self.state.clear()

class VirtualEnvironmentManager:
    """Enhanced manager with data handling"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_simulations: Dict[str, VirtualEnvironment] = {}
        self.learning_data: List[Dict[str, Any]] = []
        
        # Setup data directory
        self.data_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "ai_training_data"
        )
        os.makedirs(self.data_root, exist_ok=True)

    async def create_simulation(self, simulation_type: str, params: Dict[str, Any]) -> str:
        """Create a new simulation environment"""
        sim_id = f"sim_{len(self.active_simulations)}"
        self.active_simulations[sim_id] = VirtualEnvironment(simulation_type, params)
        return sim_id
        
    async def run_experiment(self, sim_id: str, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run an experiment in a simulation"""
        if sim_id not in self.active_simulations:
            raise ValueError(f"Simulation {sim_id} not found")
            
        env = self.active_simulations[sim_id]
        results = []
        
        for step in range(experiment_config.get("validation_steps", 1)):
            action = self._create_experiment_action(experiment_config, step)
            result = await env.step(action)
            results.append(result)
            
        final_result = self._aggregate_results(results)
        self.learning_data.append({
            "sim_id": sim_id,
            "config": experiment_config,
            "results": final_result
        })
        
        return final_result
    
    def _create_experiment_action(self, config: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Create an action for the experiment step"""
        return {
            "type": "validate",
            "topic": config.get("topic", "unknown"),
            "step": step,
            "parameters": config.get("parameters", {})
        }
        
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced result aggregation with metadata"""
        confidence_scores = [r.get("confidence_score", 0) for r in results]
        base_results = {
            "confidence_score": np.mean(confidence_scores),
            "validation_successes": sum(r.get("validation_success", False) for r in results),
            "total_steps": len(results),
            "timestamp": datetime.utcnow(),
            "learned_data": {
                "confidence_distribution": confidence_scores,
                "validation_rate": sum(r.get("validation_success", False) for r in results) / len(results)
            }
        }
        
        # Add additional metadata
        base_results.update({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "environment_version": "2.0",
                "data_path": self.data_root
            }
        })
        
        return base_results
        
    async def cleanup(self) -> None:
        """Cleanup all active simulations"""
        self.active_simulations.clear()
        logger.info("Cleaned up all virtual environments")
