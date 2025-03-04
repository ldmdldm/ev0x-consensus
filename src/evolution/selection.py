"""Adaptive model selection implementation for the ev0x project."""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Data class for storing model performance metrics."""
    model_id: str
    accuracy: float
    confidence: float
    novelty: float
    bias_score: float
    latency: float
    timestamp: float
    
    def get_composite_score(self, weights: Dict[str, float]) -> float:
        """
        Calculate a composite score based on weighted metrics.
        
        Args:
            weights: Dictionary of weights for each metric
            
        Returns:
            Composite performance score
        """
        return (
            weights.get("accuracy", 0.4) * self.accuracy +
            weights.get("confidence", 0.2) * self.confidence +
            weights.get("novelty", 0.2) * self.novelty +
            weights.get("bias", 0.1) * (1 - self.bias_score) +
            weights.get("latency", 0.1) * (1 / (1 + self.latency))
        )


class AdaptiveModelSelector:
    """
    System for adaptive selection of AI models based on performance metrics.
    
    This class implements algorithms for ranking, selecting, tracking, and 
    automatically retiring/introducing AI models based on continuous performance
    evaluation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the adaptive model selector.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Default configuration
        self.config = {
            "performance_window": 100,  # Number of inferences to track
            "retirement_threshold": 0.3,  # Models below this percentile are retired
            "selection_weights": {
                "accuracy": 0.4,
                "confidence": 0.2,
                "novelty": 0.2,
                "bias": 0.1,
                "latency": 0.1
            },
            "active_models_count": 5,  # Number of active models to maintain
            "model_cooldown": 50,  # Inferences before re-evaluating retired models
        }
        
        # Load custom configuration if provided
        if config_path:
            try:
                import json
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    self.config.update(custom_config)
            except Exception as e:
                logger.error(f"Failed to load model selector config: {e}")
        
        # Performance history for each model
        self.performance_history: Dict[str, List[ModelPerformance]] = {}
        
        # Set of currently active models
        self.active_models: List[str] = []
        
        # Tracking cooldown periods for retired models
        self.retired_models: Dict[str, int] = {}
        
        logger.info("Initialized AdaptiveModelSelector")
    
    def update_performance(self, model_id: str, 
                        metrics: Dict[str, float]) -> None:
        """
        Update the performance metrics for a model.
        
        Args:
            model_id: Identifier for the model
            metrics: Dictionary of performance metrics
        """
        performance = ModelPerformance(
            model_id=model_id,
            accuracy=metrics.get("accuracy", 0.0),
            confidence=metrics.get("confidence", 0.0),
            novelty=metrics.get("novelty", 0.0),
            bias_score=metrics.get("bias_score", 0.0),
            latency=metrics.get("latency", 0.0),
            timestamp=time.time()
        )
        
        # Initialize history for new models
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        # Add new performance record
        self.performance_history[model_id].append(performance)
        
        # Trim history to window size
        window_size = self.config["performance_window"]
        if len(self.performance_history[model_id]) > window_size:
            self.performance_history[model_id] = self.performance_history[model_id][-window_size:]
        
        logger.debug(f"Updated performance metrics for model {model_id}")
    
    def select_models(self) -> List[str]:
        """
        Select the best performing models based on current metrics.
        
        Returns:
            List of selected model IDs
        """
        # Calculate composite scores for each model
        model_scores: List[Tuple[str, float]] = []
        weights = self.config["selection_weights"]
        
        for model_id, history in self.performance_history.items():
            if not history:
                continue
            
            # Skip models in cooldown
            if model_id in self.retired_models:
                self.retired_models[model_id] -= 1
                if self.retired_models[model_id] <= 0:
                    del self.retired_models[model_id]
                else:
                    continue
            
            # Calculate average performance over recent history
            recent_history = history[-min(10, len(history)):]
            avg_score = np.mean([p.get_composite_score(weights) for p in recent_history])
            model_scores.append((model_id, avg_score))
        
        # Sort models by score (descending)
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Determine active models count
        active_count = min(self.config["active_models_count"], len(model_scores))
        
        # Determine retirement threshold
        if len(model_scores) > active_count:
            scores = [score for _, score in model_scores]
            retirement_percentile = np.percentile(
                scores, self.config["retirement_threshold"] * 100
            )
            
            # Retire low-performing models
            for model_id, score in model_scores:
                if score <= retirement_percentile and model_id not in self.retired_models:
                    self.retired_models[model_id] = self.config["model_cooldown"]
                    logger.info(f"Model {model_id} retired due to low performance (score: {score:.4f})")
        
        # Select top performing models
        selected_models = [model_id for model_id, _ in model_scores[:active_count]]
        
        # Update active models list
        self.active_models = selected_models
        
        logger.info(f"Selected {len(selected_models)} active models")
        return selected_models
    
    def get_model_rankings(self) -> List[Dict[str, Any]]:
        """
        Get the current rankings of all models.
        
        Returns:
            List of model ranking information
        """
        rankings = []
        weights = self.config["selection_weights"]
        
        for model_id, history in self.performance_history.items():
            if not history:
                continue
            
            # Calculate average metrics
            recent_history = history[-min(10, len(history)):]
            avg_metrics = {
                "accuracy": np.mean([p.accuracy for p in recent_history]),
                "confidence": np.mean([p.confidence for p in recent_history]),
                "novelty": np.mean([p.novelty for p in recent_history]),
                "bias_score": np.mean([p.bias_score for p in recent_history]),
                "latency": np.mean([p.latency for p in recent_history])
            }
            
            composite_score = np.mean([p.get_composite_score(weights) for p in recent_history])
            
            rankings.append({
                "model_id": model_id,
                "composite_score": composite_score,
                "metrics": avg_metrics,
                "status": "active" if model_id in self.active_models else "retired",
                "cooldown": self.retired_models.get(model_id, 0)
            })
        
        # Sort by composite score
        rankings.sort(key=lambda x: x["composite_score"], reverse=True)
        
        return rankings

