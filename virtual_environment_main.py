import asyncio
import argparse
import json
import logging
import os
from typing import Dict, Any
from datetime import datetime
from llm.virtual_environment import VirtualEnvironmentManager
from quantum_computing.quantum_simulator import QuantumSimulator  # Import QuantumSimulator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.layout import Layout
from rich import print
from rich.text import Text
from rich.table import Table
from core.logging.advanced_logger import AdvancedLogger, CustomJSONEncoder

# Setup data directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "ai_training_data")
DATA_DIR = os.path.join(BASE_DIR, "data", "ai_training_data")
VIRTUAL_ENV_DIR = os.path.join(BASE_DIR, "virtual_environments")

# Create directories if they don't exist
for directory in [DATA_DIR, VIRTUAL_ENV_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize advanced logger
logger = AdvancedLogger("virtual_environment").get_logger()

console = Console()

class VirtualEnvironmentSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_manager = VirtualEnvironmentManager(config)
        self.data_path = DATA_DIR
        self.env_path = VIRTUAL_ENV_DIR
        self.progress_data = {}
        self.quantum_sim = QuantumSimulator(n_qubits=config.get('n_qubits', 8))
        
    async def run_knowledge_validation(self, topic: str, steps: int = 10):
        """Run a knowledge validation experiment with quantum processing"""
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[green]Validating {topic}", total=steps)
            
            sim_id = await self.env_manager.create_simulation(
                "knowledge_validation",
                {
                    "topic": topic,
                    "data_path": os.path.join(self.data_path, topic),
                    "env_path": os.path.join(self.env_path, f"sim_{topic}")
                }
            )
            
            self.progress_data[sim_id] = {
                "topic": topic,
                "progress": 0.0,
                "current_step": 0,
                "total_steps": steps,
                "status": "Running"
            }
            
            # Add quantum processing tasks
            quantum_task = progress.add_task(
                "[cyan]Quantum Processing", 
                total=steps * 3  # Triple steps for quantum processing
            )
            
            results = await self.env_manager.run_experiment(
                sim_id,
                {
                    "topic": topic,
                    "validation_steps": steps,
                    "parameters": {
                        "depth": 3,
                        "breadth": 5,
                        "learning_rate": 0.001,
                        "data_collection": True,
                        "store_results": True,
                        "quantum_enabled": True,
                        "quantum_complexity": 5,
                        "progress_callback": lambda step: self._update_progress(sim_id, step, steps, progress, task),
                        "quantum_progress_callback": lambda step: progress.update(quantum_task, completed=step)
                    },
                    "storage": {
                        "format": "jsonl",
                        "compress": True,
                        "path": os.path.join(self.data_path, topic, 
                                           f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl.gz")
                    }
                }
            )
            
            # Add quantum processing results
            quantum_results = await self.quantum_sim.run_quantum_process(
                complexity=self.config.get('quantum_complexity', 5)
            )
            results.update({"quantum_metrics": quantum_results})
            
            progress.update(task, completed=steps)
            self._update_progress(sim_id, steps, steps, progress, task)
            
            # Store results and log
            self._store_results(topic, results)
            self._log_results(topic, results)
            return results
            
    def _update_progress(self, sim_id: str, current: int, total: int, 
                        progress: Progress, task_id: Any) -> None:
        """Update progress for both terminal and GUI"""
        progress_value = current / total
        self.progress_data[sim_id].update({
            "progress": progress_value,
            "current_step": current,
            "status": "Running" if current < total else "Completed"
        })
        
        progress.update(task_id, completed=current)
        
        # Display rich terminal summary
        self._display_summary()
        
    def _display_summary(self) -> None:
        """Display a rich summary in the terminal"""
        layout = Layout()
        layout.split_column(
            Layout(Panel("Virtual Environment Status", title="JARVIS")),
            Layout(name="simulations")
        )
        
        sims_text = ""
        for sim_id, data in self.progress_data.items():
            sims_text += f"\n[bold blue]{sim_id}[/bold blue]\n"
            sims_text += f"Topic: {data['topic']}\n"
            sims_text += f"Progress: {data['progress']*100:.1f}%\n"
            sims_text += f"Status: {data['status']}\n"
            sims_text += "─" * 40 + "\n"
            
        layout["simulations"].update(Panel(sims_text, title="Active Simulations"))
        
        console.clear()
        console.print(layout)

    def _store_results(self, topic: str, results: Dict[str, Any]):
        """Store validation results with proper JSON serialization"""
        import json
        import gzip
        
        filepath = os.path.join(
            self.data_path, 
            topic, 
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json.gz"
        )
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with gzip.open(filepath, 'wt') as f:
            json.dump(results, f, cls=CustomJSONEncoder, indent=2)

    def _log_results(self, topic: str, results: Dict[str, Any]):
        """Enhanced logging with quantum metrics and structured output"""
        log_data = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "confidence_score": float(results['confidence_score']),
            "validation_rate": f"{results['validation_successes']}/{results['total_steps']}",
            "storage_path": f"{self.data_path}/{topic}"
        }
        
        if "quantum_metrics" in results:
            qm = results["quantum_metrics"]
            log_data.update({
                "quantum_process_time": float(qm['process_time']),
                "quantum_coherence": float(qm['coherence'])
            })
            
        logger.info(f"Validation Results:\n{json.dumps(log_data, indent=2, cls=CustomJSONEncoder)}")

    async def run_simulation_batch(self, topics: list, steps_per_topic: int = 5):
        """Run multiple simulations in parallel"""
        tasks = []
        for topic in topics:
            task = asyncio.create_task(self.run_knowledge_validation(topic, steps_per_topic))
            tasks.append(task)
        return await asyncio.gather(*tasks)

async def main():
    parser = argparse.ArgumentParser(description='Virtual Environment Testing System')
    parser.add_argument('--topics', nargs='+', default=['physics', 'math', 'ai'],
                      help='Topics to validate')
    parser.add_argument('--steps', type=int, default=5,
                      help='Number of validation steps per topic')
    parser.add_argument('--parallel', action='store_true',
                      help='Run simulations in parallel')
    
    args = parser.parse_args()
    
    config = {
        "simulation_types": ["knowledge_validation", "scenario_testing"],
        "max_parallel_sims": 10,
        "default_steps": 5,
        "logging_level": "INFO",
        "n_qubits": 8,
        "quantum_complexity": 5
    }
    
    system = VirtualEnvironmentSystem(config)
    
    try:
        if args.parallel:
            results = await system.run_simulation_batch(args.topics, args.steps)
            logger.info(f"Completed {len(results)} parallel simulations")
        else:
            for topic in args.topics:
                await system.run_knowledge_validation(topic, args.steps)
                
    except Exception as e:
        logger.error(f"Simulation error: {e}")
    finally:
        await system.env_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
