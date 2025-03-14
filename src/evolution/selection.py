"""Adaptive model selection implementation for the ev0x project."""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, TypedDict, Union
from enum import Enum, auto
from dataclasses import dataclass
import time

# Initialize logger
logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Enumeration of supported verification source types."""
    WEB = auto()
    ACADEMIC = auto()
    NEWS = auto()
    KNOWLEDGE_BASE = auto()
    GENERIC = auto()

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


class PerformanceMetrics(TypedDict, total=False):
    """Type definition for performance metrics."""
    accuracy: float
    confidence: float
    novelty: float
    bias_score: float
    latency: float


class ModelRanking(TypedDict):
    """Type definition for model ranking information."""
    model_id: str
    composite_score: float
    metrics: Dict[str, float]
    status: str
    cooldown: int


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
        self.config: Dict[str, Union[int, float, Dict[str, float]]] = {
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
                           metrics: PerformanceMetrics) -> None:
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
        window_size = int(self.config["performance_window"]) if isinstance(self.config["performance_window"], (int, float, str)) else 100
        if len(self.performance_history[model_id]) > window_size:
            self.performance_history[model_id] = self.performance_history[model_id][-window_size:]

        logger.debug(f"Updated performance metrics for model {model_id}")

    def select_models(self, input_data: Optional[Any] = None) -> List[str]:
        """
        Select the best performing models based on current metrics.

        Args:
            input_data: Optional input data that might be used for model selection (not used currently)

        Returns:
            List of selected model IDs
        """
        # Calculate composite scores for each model
        model_scores: List[Tuple[str, float]] = []
        weights = self.config["selection_weights"] if isinstance(self.config["selection_weights"], dict) else {}

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
            w_dict: Dict[str, float] = {k: float(v) for k, v in weights.items() if isinstance(v, (int, float, str))}
            avg_score = np.mean([p.get_composite_score(w_dict) for p in recent_history])
            model_scores.append((model_id, float(avg_score)))

        # Sort models by score (descending)
        model_scores.sort(key=lambda x: x[1], reverse=True)

        # Determine active models count
        active_count = min(int(self.config["active_models_count"]) if isinstance(self.config["active_models_count"], (int, float, str)) else 5, len(model_scores))

        # Determine retirement threshold
        if len(model_scores) > active_count:
            scores = [score for _, score in model_scores]
            retirement_percentile = np.percentile(
                scores, float(self.config["retirement_threshold"]) * 100 if isinstance(self.config["retirement_threshold"], (int, float, str)) else 30
            )

            # Retire low-performing models
            for model_id, score in model_scores:
                if score <= retirement_percentile and model_id not in self.retired_models:
                    self.retired_models[model_id] = int(self.config["model_cooldown"]) if isinstance(self.config["model_cooldown"], (int, float, str)) else 50
                    logger.info(f"Model {model_id} retired due to low performance (score: {score:.4f})")

        # Select top performing models
        selected_models = [model_id for model_id, _ in model_scores[:active_count]]

        # Update active models list
        self.active_models = selected_models

        logger.info(f"Selected {len(selected_models)} active models")
        return selected_models

    def get_model_rankings(self) -> List[ModelRanking]:
        """
        Get the current rankings of all models.

        Returns:
            List of model ranking information
        """
        rankings = []
        weights_dict = self.config["selection_weights"] if isinstance(self.config["selection_weights"], dict) else {}
        weights: Dict[str, float] = {k: float(v) for k, v in weights_dict.items() if isinstance(v, (int, float, str))}

        for model_id, history in self.performance_history.items():
            if not history:
                continue

            # Calculate average metrics
            recent_history = history[-min(10, len(history)):]
            avg_metrics = {
                "accuracy": float(np.mean([p.accuracy for p in recent_history])),
                "confidence": float(np.mean([p.confidence for p in recent_history])),
                "novelty": float(np.mean([p.novelty for p in recent_history])),
                "bias_score": float(np.mean([p.bias_score for p in recent_history])),
                "latency": float(np.mean([p.latency for p in recent_history]))
            }

            composite_score = float(np.mean([p.get_composite_score(weights) for p in recent_history]))

            ranking: ModelRanking = {
                "model_id": model_id,
                "composite_score": composite_score,
                "metrics": avg_metrics,
                "status": "active" if model_id in self.active_models else "retired",
                "cooldown": self.retired_models.get(model_id, 0)
            }
            rankings.append(ranking)

        # Sort by composite score
        rankings.sort(key=lambda x: x["composite_score"], reverse=True)

        return rankings


class ModelSelectorResult(TypedDict):
    """Type definition for model selector result."""
    selected_models: List[str]
    rankings: List[ModelRanking]
    active_count: int
    total_models: int
    timestamp: float


class ModelSelector:
    """
    Wrapper around AdaptiveModelSelector to provide a standardized interface for model selection.

    This class serves as a facade for the AdaptiveModelSelector, providing a simpler interface
    that abstracts the underlying complexity while maintaining the adaptive selection capabilities.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ModelSelector with an AdaptiveModelSelector.

        Args:
            config_path: Optional path to configuration file
        """
        self.adaptive_selector = AdaptiveModelSelector(config_path)
        logger.info("Initialized ModelSelector wrapping AdaptiveModelSelector")

    def select(self, performance_data: Dict[str, PerformanceMetrics]) -> ModelSelectorResult:
        """
        Select models based on performance data.

        Args:
            performance_data: Dictionary mapping model IDs to their performance metrics
                             Example format:
                             {
                                "model1": {"accuracy": 0.85, "confidence": 0.9, ...},
                                "model2": {"accuracy": 0.78, "confidence": 0.85, ...},
                             }

        Returns:
            Dictionary containing model updates, rankings, and other selection information
        """
        # Update performance data for all models
        for model_id, metrics in performance_data.items():
            self.adaptive_selector.update_performance(model_id, metrics)

        # Select the best models
        selected_model_ids = self.adaptive_selector.select_models()

        # Get detailed rankings for all models
        rankings = self.adaptive_selector.get_model_rankings()

        # Prepare the response with model updates
        model_updates: ModelSelectorResult = {
            "selected_models": selected_model_ids,
            "rankings": rankings,
            "active_count": len(selected_model_ids),
            "total_models": len(performance_data),
            "timestamp": time.time()
        }

        logger.info(f"Selected {len(selected_model_ids)} models out of {len(performance_data)} total models")
        return model_updates

    def get_active_models(self) -> List[str]:
        """
        Get the currently active models.

        Returns:
            List of active model IDs
        """
        return self.adaptive_selector.active_models

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.

        Returns:
            Current configuration dictionary
        """
        return self.adaptive_selector.config

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the configuration.

        Args:
            new_config: New configuration values to update
        """
        self.adaptive_selector.config.update(new_config)
        logger.info("Updated ModelSelector configuration")
    def verify_text(self, text: str, source_types: Optional[List[SourceType]] = None) -> ModelSelectorResult:
        """
        Verify the given text using currently active models.

        Args:
            text: The text content to verify
            source_types: Optional list of source types to use for verification

        Returns:
            ModelSelectorResult containing verification results
        """
        # If no source types specified, use all types
        if source_types is None:
            source_types = list(SourceType)

        # Get currently active models
        active_model_ids = self.get_active_models()
        
        # Mock performance metrics since we don't have actual verification logic here
        # In a real implementation, you would process the text with each model and gather metrics
        mock_performance_data: Dict[str, PerformanceMetrics] = {}
        
        for model_id in active_model_ids:
            # Simulate verification process
            # This would be replaced with actual text verification using the model
            mock_performance_data[model_id] = {
                "accuracy": 0.8 + 0.1 * (hash(model_id + text) % 10) / 10,
                "confidence": 0.7 + 0.2 * (hash(text + model_id) % 10) / 10,
                "novelty": 0.5 + 0.3 * (hash(model_id) % 10) / 10,
                "bias_score": 0.2 + 0.1 * (hash(text) % 10) / 10,
                "latency": 0.5 + 0.5 * (hash(model_id + str(source_types)) % 10) / 10
            }
        
        # Use the select method to process the verification results
        result = self.select(mock_performance_data)
        
        logger.info(f"Verified text using {len(active_model_ids)} models with {len(source_types)} source types")
        return result

