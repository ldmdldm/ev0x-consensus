#!/usr/bin/env python3
"""
ev0x: Evolutionary Model Consensus Mechanism
Main entry point for the application.
"""

import os
import logging
import asyncio
from typing import Dict, List, Any

# Import ev0x components
from src.models.model_runner import ModelRunner
from src.consensus.synthesizer import ConsensusSynthesizer
from src.rewards.shapley import ShapleyCalculator
from src.evaluation.metrics import PerformanceTracker
from src.evolution.selection import ModelSelector
from src.evolution.meta_intelligence import MetaIntelligence
from src.bias.detector import BiasDetector
from src.bias.neutralizer import BiasNeutralizer
from src.config.models import AVAILABLE_MODELS
from src.data.datasets import DatasetLoader
from src.api.server import APIServer

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
        # Initialize core components
        self.model_runner = ModelRunner(models=AVAILABLE_MODELS)
        self.consensus = ConsensusSynthesizer()
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
        self.api_server = APIServer(self)
        
        # System state
        self.current_generation = 0
        self.performance_history = {}
        
        logger.info("Evolutionary Consensus System initialized")
        
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
                "generation": self.current_generation
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
        self.api_server.start()
        
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

