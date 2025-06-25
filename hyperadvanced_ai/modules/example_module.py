import logging
from hyperadvanced_ai.core.logging import get_logger
from hyperadvanced_ai.utils.dynamic_loader import dynamic_import

logger = get_logger("hyperadvanced_ai.modules.example_module")

def run_example_task(data: dict) -> dict:
    """Run an example AI task."""
    logger.info(f"Running example task with data: {data}")
    # Simulate processing
    result = {"result": "success", "input": data}
    return result
