#!/usr/bin/env python3
"""
ev0x: Evolutionary Model Consensus Mechanism
Main entry point for the application.
"""

import os
import logging
import asyncio
import json
from typing import Dict, List, Any
from pathlib import Path

# Import ev0x components
from src.models.model_runner import ModelRunner
from src.consensus.synthesizer import ConsensusSynthesizer
from src.rewards.shapley import ShapleyCalculator
from src.router.openrouter import OpenRouterClient
from src.evaluation.metrics import PerformanceTracker
from src.evolution.selection import ModelSelector
from src.evolution.meta_intelligence import MetaIntelligence
from src.bias.detector import BiasDetector
from src.bias.neutralizer import BiasNeutralizer
from src.config.models import AVAILABLE_MODELS
from src.data.datasets import DatasetLoader
from src.api.server import APIServer
from src.tee.attestation import TEEAttestationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvolutionaryConsensusSystem:
    """Main class for the ev0x system that coordinates all components."""
    
    def __init__(self):
        """Initialize the evolutionary consensus system components."""
        # Initialize TEE attestation
        self.tee_manager = TEEAttestationManager()
        self.is_tee_verified = self.tee_manager.verify_environment()
        if self.is_tee_verified:
            logger.info(f"Running in verified TEE environment: {self.tee_manager.get_tee_type()}")
            # Export attestation information to a file for verification
            attestation_path = Path("/tmp/tee_attestation.json")
            if self.tee_manager.export_attestation(str(attestation_path)):
                logger.info(f"TEE attestation exported to {attestation_path}")
        else:
            logger.warning("Not running in a verified TEE environment")
            
        # Initialize core components
        self.router_client = OpenRouterClient()
        self.model_runner = ModelRunner(router_client=self.router_client)
        
        # Register each available model
        for model_config in AVAILABLE_MODELS:
            model_id = model_config.name  # Use name as the model ID
            model_function = self._create_model_function(model_config)
            
            # Extract parameters from ModelConfig
            params = model_config.parameters if model_config.parameters else {}
            
            # Register the model with the runner
            self.model_runner.register_model(
                model_id,
                model_function,
                **params
            )
        # Create a basic configuration for the consensus synthesizer
        # Create a basic configuration for the consensus synthesizer
        consensus_config = {
            "iterations": {
                "max_iterations": 3,
                "improvement_threshold": 0.05,
                "feedback_prompt": "Please improve the answer based on the following feedback: {feedback_points}"
            },
            "models": [
                {"id": model_config.name, "params": model_config.parameters if model_config.parameters else {}} 
                for model_config in AVAILABLE_MODELS
            ],
            "aggregator": {
                "model_id": AVAILABLE_MODELS[0].name if AVAILABLE_MODELS else ""
            }
        }
        self.consensus = ConsensusSynthesizer(consensus_config)
        self.rewards = ShapleyCalculator()
        self.metrics = PerformanceTracker()
        
        # Initialize evolutionary components
        self.model_selector = ModelSelector()
        self.meta_intelligence = MetaIntelligence()
        
        # Initialize bias components
        self.bias_detector = BiasDetector()
        self.bias_neutralizer = BiasNeutralizer()
        
        # Initialize data and API components
        self.data_loader = DatasetLoader()
        self.api_server = APIServer(None)  # Pass None or a default config path instead of self
        
        # System state
        self.current_generation = 0
        self.performance_history = {}
        self.tee_status = {
            "verified": self.is_tee_verified,
            "type": self.tee_manager.get_tee_type() if self.is_tee_verified else "NONE",
            "attestation_available": self.tee_manager.get_attestation() is not None
        }
        
        logger.info("Evolutionary Consensus System initialized")
    
    def _create_model_function(self, model_config):
        """
        Create a callable model function from a ModelConfig object.
        
        Args:
            model_config: A ModelConfig object containing model information
            
        Returns:
            An async callable function that processes input data
        """
        import os
        import httpx
        import json
        
        async def model_function(input_data, **kwargs):
            """
            Process input data using the configured model.
            
            Args:
                input_data: Input text or data to process
                **kwargs: Additional parameters to override defaults
                
            Returns:
                Processed output from the model
            """
            # Get API key from environment variable
            api_key = os.environ.get(model_config.api_key_env_var)
            if not api_key:
                raise ValueError(f"API key not found in environment variable {model_config.api_key_env_var}")
            
            # Merge default parameters with any overrides
            params = {**(model_config.parameters or {}), **kwargs}
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Prepare request data based on provider and model type
            if model_config.provider == "gemini" and model_config.model_type == "llm":
                # Handle Gemini LLM models (text generation)
                request_data = {
                    "contents": [{"role": "user", "parts": [{"text": input_data}]}],
                    **params
                }
            elif model_config.provider == "openrouter" and model_config.model_type == "llm":
                # Handle OpenRouter LLM models (using OpenAI-compatible format)
                request_data = {
                    "messages": [{"role": "user", "content": input_data}],
                    **params
                }
            elif model_config.model_type == "embedding":
                # Handle embedding models
                request_data = {
                    "text": input_data,
                    **params
                }
            else:
                # Generic fallback
                request_data = {
                    "input": input_data,
                    **params
                }
            
            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    model_config.endpoint,
                    headers=headers,
                    json=request_data,
                    timeout=30.0
                )
                
                # Process the response
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Format the response based on the provider
                    if model_config.provider == "openrouter":
                        # Extract content from OpenRouter response format
                        try:
                            if "choices" in response_data and len(response_data["choices"]) > 0:
                                if "message" in response_data["choices"][0]:
                                    content = response_data["choices"][0]["message"]["content"]
                                    # Restructure to a standardized format for ev0x
                                    return {
                                        "candidates": [{"content": {"parts": [{"text": content}]}}],
                                        "original_response": response_data
                                    }
                            # If extraction fails, return the full response
                            return response_data
                        except Exception as e:
                            logger.warning(f"Error extracting content from OpenRouter response: {e}")
                            return response_data
                    else:
                        # Return the original response for other providers
                        return response_data
                else:
                    raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        
        return model_function
        
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the evolutionary consensus system.
        
        Args:
            query: The user's input query
            
        Returns:
            A dictionary containing the consensus result and metadata
        """
        logger.info(f"Processing query: {query}")
        
        # 1. Run all models to get predictions
        model_outputs = await self.model_runner.run_models(query)
        
        # 2. Detect and neutralize bias in model outputs
        bias_scores = self.bias_detector.detect(model_outputs)
        neutralized_outputs = self.bias_neutralizer.neutralize(model_outputs, bias_scores)
        
        # 3. Generate consensus from neutralized outputs
        consensus_result = self.consensus.synthesize(neutralized_outputs)
        
        # 4. Calculate rewards using Shapley values
        rewards = self.rewards.calculate(neutralized_outputs, consensus_result)
        
        # 5. Update performance metrics
        self.metrics.update(model_outputs, consensus_result, rewards)
        
        # 6. Meta-intelligence learning from this interaction
        self.meta_intelligence.learn(query, model_outputs, consensus_result)
        
        # 7. Prepare response
        response = {
            "result": consensus_result,
            "meta": {
                "bias_scores": bias_scores,
                "model_contributions": rewards,
                "generation": self.current_generation,
                "tee_status": self.tee_status
            }
        }
        
        return response
        
    async def evolve_system(self):
        """Evolve the system by updating model selection and weights."""
        logger.info(f"Evolving system to generation {self.current_generation + 1}")
        
        # Get performance metrics
        performance_data = self.metrics.get_metrics()
        
        # Update model selection based on performance
        model_updates = self.model_selector.select(performance_data)
        await self.model_runner.update_models(model_updates)
        
        # Update consensus weights
        new_weights = self.meta_intelligence.optimize_weights(performance_data)
        self.consensus.update_weights(new_weights)
        
        # Generate training data for improvement
        training_data = self.meta_intelligence.generate_training_data()
        if training_data:
            await self.model_runner.fine_tune(training_data)
        
        # Increment generation counter
        self.current_generation += 1
        
        # Store performance history
        self.performance_history[self.current_generation] = performance_data
        
        logger.info(f"System evolved to generation {self.current_generation}")
    
    def start(self):
        """Start the evolutionary consensus system."""
        logger.info("Starting ev0x system")
        
        # Start the API server
        self.api_server.start_server()
        
        # Schedule regular system evolution
        asyncio.create_task(self._schedule_evolution())
    
    async def _schedule_evolution(self, evolution_interval=3600):
        """Schedule regular system evolution at specified intervals."""
        while True:
            await asyncio.sleep(evolution_interval)  # Default: evolve once per hour
            await self.evolve_system()

async def main():
    """Main entry point for the application."""
    # Initialize the system
    system = EvolutionaryConsensusSystem()
    
    # Start the system
    system.start()
    
    # Keep the application running
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

