"""
Model Runner for executing multiple AI models simultaneously.
"""
import asyncio
import logging
import time
import traceback
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic, Protocol

# Configure logging
logger = logging.getLogger(__name__)


class ModelProtocol(Protocol):
    """Protocol defining the interface for models."""
    async def __call__(self, input_data: Any, **kwargs) -> Any:
        ...


T = TypeVar('T')


class ModelRunner:
    """
    Handles the execution of multiple AI models in parallel.
    """
    
    def __init__(self):
        self.models = {}
        
    def register_model(self, model_id: str, model_fn: Callable, **kwargs):
        """
        Register a model with the runner.
        
        Args:
            model_id: Unique identifier for the model
            model_fn: Callable function that executes the model
            **kwargs: Additional configuration for the model
        """
        self.models[model_id] = {
            "function": model_fn,
            "config": kwargs
        }
        
    async def _execute_model(self, model_id: str, input_data: Any, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Execute a single model asynchronously with timeout support.
        
        Args:
            model_id: ID of the model to execute
            input_data: Input data for the model
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary containing model outputs and metadata
        """
        """
        Execute a single model asynchronously.
        
        Args:
            model_id: ID of the model to execute
            input_data: Input data for the model
            
        Returns:
            Dictionary containing model outputs and metadata
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
            
        logger.info(f"Executing model {model_id}")
            
        model_info = self.models[model_id]
        model_fn = model_info["function"]
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    model_fn(input_data, **model_info["config"]),
                    timeout=timeout
                )
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Collect telemetry
                logger.info(f"Model {model_id} executed successfully in {execution_time:.2f}s")
                
                return {
                    "model_id": model_id,
                    "output": result,
                    "execution_time": execution_time,
                    "status": "success"
                }
            except asyncio.TimeoutError:
                logger.warning(f"Model {model_id} execution timed out after {timeout}s")
                return {
                    "model_id": model_id,
                    "output": None,
                    "status": "timeout",
                    "error": f"Execution timed out after {timeout}s"
                }
            
        except Exception as e:
            logger.error(f"Error executing model {model_id}: {str(e)}")
            return {
                "model_id": model_id,
                "output": None,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    async def run_models(self, 
                    input_data: Any, 
                    model_ids: Optional[List[str]] = None,
                    timeout: float = 30.0) -> Dict[str, Any]:
        """
        Run multiple models in parallel.
        
        Args:
            input_data: Input data to be processed by the models
            model_ids: List of model IDs to run (runs all registered models if None)
            
        Returns:
            Dictionary mapping model IDs to their outputs
        """
        if model_ids is None:
            model_ids = list(self.models.keys())
            
        tasks = [self._execute_model(model_id, input_data, timeout=timeout) 
                for model_id in model_ids if model_id in self.models]
        
        results = await asyncio.gather(*tasks)
        return {result["model_id"]: result for result in results}

async def run_models_batch(self, 
                        batch_inputs: List[Any], 
                        model_ids: Optional[List[str]] = None,
                        timeout: float = 30.0) -> List[Dict[str, Any]]:
    """
    Run models on a batch of inputs in parallel.
    
    Args:
        batch_inputs: List of inputs to be processed
        model_ids: List of model IDs to run (runs all registered models if None)
        timeout: Maximum execution time per model in seconds
        
    Returns:
        List of dictionaries mapping model IDs to their outputs, one per input
    """
    if model_ids is None:
        model_ids = list(self.models.keys())
        
    # Create tasks for all inputs and all models
    batch_tasks = []
    for input_data in batch_inputs:
        batch_tasks.append(self.run_models(input_data, model_ids, timeout))
        
    # Run all tasks concurrently
    return await asyncio.gather(*batch_tasks)

def get_available_models(self) -> List[str]:
    """
    Get a list of all available model IDs.
    
    Returns:
        List of registered model IDs
    """
    return list(self.models.keys())

def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific model.
    
    Args:
        model_id: ID of the model
        
    Returns:
        Dictionary with model information or None if model not found
    """
    if model_id not in self.models:
        return None
        
    return {
        "model_id": model_id,
        "config": self.models[model_id]["config"]
    }
