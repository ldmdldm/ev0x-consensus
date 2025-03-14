"""Meta-intelligence implementation for the ev0x project."""

import logging
import random
from typing import Dict, List, Any, Optional, Tuple, DefaultDict, Union
from collections import defaultdict
from dataclasses import dataclass
import json
import os
import time

# Import our new citation module
from src.factual.citation import CitationVerifier, VerifiedOutput

# Initialize logger
logger = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """Data class for storing model behavioral profiles."""
    model_id: str
    strengths: Dict[str, float]  # Task types and performance scores
    weaknesses: Dict[str, float]  # Task types and performance scores
    bias_tendencies: Dict[str, float]  # Bias categories and scores
    confidence_profile: Dict[str, float]  # Task types and confidence levels
    latency: float  # Response time in seconds


class MetaIntelligence:
    """
    Meta-Intelligence system that learns how different AI models behave and can predict
    their responses, strengths, and weaknesses.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the Meta-Intelligence system.

        Args:
            storage_path: Path to store/load model profiles
        """
        self.model_profiles: Dict[str, ModelProfile] = {}
        self.meta_models: Dict[str, Union[Dict[str, Any], Any]] = {}  # Models that predict other models' behavior
        self.source_type_counts: DefaultDict[str, int] = defaultdict(int)  # Tracking citation source types
        self.storage_path = storage_path or os.path.join(os.getcwd(), "data", "model_profiles")
        os.makedirs(self.storage_path, exist_ok=True)

        # Load existing profiles if available
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load existing model profiles from storage."""
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith(".json"):
                    with open(os.path.join(self.storage_path, filename), "r") as f:
                        data = json.load(f)
                        model_id = data.pop("model_id")
                        self.model_profiles[model_id] = ModelProfile(model_id=model_id, **data)

            logger.info(f"Loaded {len(self.model_profiles)} model profiles from storage")
        except Exception as e:
            logger.warning(f"Failed to load model profiles: {e}")

    def update_model_profile(self,
                             model_id: str,
                             task_type: str,
                             performance: float,
                             confidence: float,
                             latency: float,
                             biases: Optional[Dict[str, float]] = None) -> None:
        """
        Update a model's profile with new performance data.

        Args:
            model_id: Identifier for the model
            task_type: Category of task performed
            performance: Score between 0-1 indicating performance
            confidence: Model's reported confidence (0-1)
            latency: Response time in seconds
            biases: Optional dictionary of bias types and measured values
        """
        # Get or create model profile
        if model_id not in self.model_profiles:
            self.model_profiles[model_id] = ModelProfile(
                model_id=model_id,
                strengths={},
                weaknesses={},
                bias_tendencies={},
                confidence_profile={},
                latency=0.0
            )

        profile = self.model_profiles[model_id]

        # Update strengths and weaknesses
        if performance >= 0.7:
            profile.strengths[task_type] = profile.strengths.get(task_type, 0) * 0.9 + performance * 0.1
        elif performance <= 0.4:
            profile.weaknesses[task_type] = profile.weaknesses.get(task_type, 0) * 0.9 + (1 - performance) * 0.1

        # Update confidence profile
        profile.confidence_profile[task_type] = profile.confidence_profile.get(task_type, 0) * 0.9 + confidence * 0.1

        # Update latency with exponential moving average
        profile.latency = profile.latency * 0.9 + latency * 0.1

        # Update bias tendencies if provided
        if biases:
            for bias_type, bias_value in biases.items():
                profile.bias_tendencies[bias_type] = profile.bias_tendencies.get(bias_type, 0) * 0.9 + bias_value * 0.1

        # Persist updated profile
        self._save_profile(profile)

    def _save_profile(self, profile: ModelProfile) -> None:
        """Save a model profile to storage."""
        try:
            filepath = os.path.join(self.storage_path, f"{profile.model_id}.json")
            with open(filepath, "w") as f:
                json.dump(profile.__dict__, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save profile for {profile.model_id}: {e}")

    def get_model_recommendation(self, task_type: str, bias_sensitivity: float = 0.5) -> List[Tuple[str, float]]:
        """
        Get recommended models for a specific task type.

        Args:
            task_type: Type of task to be performed
            bias_sensitivity: How important bias avoidance is (0-1)

        Returns:
            List of (model_id, score) tuples sorted by recommendation score
        """
        recommendations = []

        for model_id, profile in self.model_profiles.items():
            # Base score on strengths for this task
            base_score = profile.strengths.get(task_type, 0.5)

            # Penalize for known weaknesses
            weakness_penalty = profile.weaknesses.get(task_type, 0) * 0.5

            # Penalize for known biases
            bias_penalty = sum(profile.bias_tendencies.values()) / max(len(profile.bias_tendencies), 1) * bias_sensitivity

            # Calculate final score
            score = base_score - weakness_penalty - bias_penalty

            recommendations.append((model_id, score))

        # Sort by score descending
        return sorted(recommendations, key=lambda x: x[1], reverse=True)

    def generate_synthetic_data(self, task_type: str, quantity: int = 10) -> List[Dict[str, Any]]:
        """
        Generate synthetic training data to improve model performance on specific tasks.

        Args:
            task_type: Type of task to generate data for
            quantity: Number of examples to generate

        Returns:
            List of synthetic data points
        """
        # This is a simplified implementation - in a real system, this would use
        # meta-learning to generate targeted training examples
        synthetic_data = []

        # Find models that are weak at this task type
        weak_models = []
        for model_id, profile in self.model_profiles.items():
            if task_type in profile.weaknesses and profile.weaknesses[task_type] > 0.6:
                weak_models.append(model_id)

        # Generate examples that would challenge these models
        for _ in range(quantity):
            # Simple random data generation for demonstration
            # In a real implementation, this would use more sophisticated techniques
            synthetic_example = {
                "input": f"Synthetic example for {task_type} #{_}",
                "expected_output": f"Expected response for {task_type} example #{_}",
                "target_models": weak_models,
                "difficulty": random.uniform(0.5, 0.9),
                "timestamp": time.time()
            }
            synthetic_data.append(synthetic_example)

        return synthetic_data

    def predict_model_behavior(self, model_id: str, input_data: Any) -> Dict[str, Any]:
        """
        Predict how a specific model would respond to an input without actually running it.

        Args:
            model_id: The model to simulate
            input_data: The input that would be given to the model

        Returns:
            Prediction of the model's response
        """
        # This would use the meta-models in a real implementation
        if model_id not in self.model_profiles:
            return {"error": "Model profile not found"}

        profile = self.model_profiles[model_id]

        # Determine likely task type from input
        task_type = self._infer_task_type(input_data)

        # Estimate confidence based on historical data
        confidence = profile.confidence_profile.get(task_type, 0.5)

        # Estimate performance based on strengths/weaknesses
        estimated_performance = profile.strengths.get(task_type, 0.5)

        # In a real system, we would have a trained meta-model that can actually
        # predict the content of the response, not just these metrics
        return {
            "estimated_confidence": confidence,
            "estimated_performance": estimated_performance,
            "estimated_latency": profile.latency,
            "likely_biases": [k for k, v in profile.bias_tendencies.items() if v > 0.3]
        }

    def _infer_task_type(self, input_data: Any) -> str:
        """Infer the type of task represented by the input data."""
        # This is a placeholder implementation
        # In a real system, this would use NLP to categorize the task
        if isinstance(input_data, str):
            if "?" in input_data:
                return "question_answering"
            elif len(input_data.split()) < 10:
                return "classification"
            else:
                return "generation"
        else:
            return "unknown"

    async def enhance_with_citations(self, model_output: str, domain: Optional[str] = None) -> VerifiedOutput:
        """
        Enhance a model's output with factual verification and citations.

        Args:
            model_output: The raw text output from an AI model
            domain: Optional domain context for specialized verification (e.g., "scientific", "medical")

        Returns:
            VerifiedOutput containing the original output, verified output with citations,
            list of citations, and overall confidence score
        """
        logger.info(f"Enhancing output with citations, domain: {domain or 'general'}")

        try:
            # Create a CitationVerifier instance
            verifier = CitationVerifier()

            # Use the verifier to verify the output
            # Convert domain to List[SourceType] if specified
            source_types = None
            if domain is not None:
                from src.factual.citation import SourceType
                if domain.lower() == "scientific":
                    source_types = [SourceType.ARXIV]
                elif domain.lower() == "medical":
                    source_types = [SourceType.PUBMED]
                
            # Call verify_text with proper parameters
            verified_output = verifier.verify_text(model_output, source_types)

            # Log verification statistics
            verified_count = sum(1 for citation in verified_output.citations 
                                if citation.verification_status.value == "verified" 
                                or citation.verification_status.value == "partially_verified")
            logger.info(
                f"Citation enhancement complete. Found {len(verified_output.citations)} factual claims, "
                f"{verified_count} verified")

            self.source_type_counts = defaultdict(int)
            for citation in verified_output.citations:
                source_type = citation.source_type.value if hasattr(citation.source_type, 'value') else 'unknown'
                self.source_type_counts[source_type] += 1
            logger.info(f"Citation sources: {self.source_type_counts}")

            # Update model profiles if we can determine which model generated this
            # In a real implementation, we would track the source model ID

            return verified_output

        except Exception as e:
            logger.error(f"Citation enhancement failed: {e}", exc_info=True)
            # Return unmodified output if verification fails
            return VerifiedOutput(
                original_output=model_output,
                verified_output=model_output,
                citations=[],
                overall_confidence=0.0
            )

# Standalone function for easier use


async def add_citations_to_output(output: str, domain: Optional[str] = None) -> VerifiedOutput:
    """
    Convenience function to add citations to model output without creating a MetaIntelligence instance.

    Args:
        output: Text to verify and enhance with citations
        domain: Optional domain context (e.g., "scientific", "medical")

    Returns:
        VerifiedOutput with the original text, verified text with citations, and confidence score
    """
    try:
        # Create a CitationVerifier instance directly for better performance
        verifier = CitationVerifier()

        # Use the verifier to verify the output
        # Convert domain to List[SourceType] if specified
        source_types = None
        if domain is not None:
            from src.factual.citation import SourceType
            if domain.lower() == "scientific":
                source_types = [SourceType.ARXIV]
            elif domain.lower() == "medical":
                source_types = [SourceType.PUBMED]

        # Call verify_text with proper parameters
        verified_output = verifier.verify_text(output, source_types)

        # Log verification statistics
        verified_count = sum(1 for citation in verified_output.citations 
                             if citation.verification_status.value == "verified" 
                             or citation.verification_status.value == "partially_verified")
        logger.info(
            f"Citation verification complete. Found {len(verified_output.citations)} factual claims, "
            f"{verified_count} verified")

        return verified_output
    except Exception as e:
        logger.error(f"Citation verification failed: {e}", exc_info=True)
        # Return unmodified output with error information if verification fails
        return VerifiedOutput(
            original_output=output,
            verified_output=f"{output}\n\n[Error: Citation verification failed: {str(e)}]",
            citations=[],
            overall_confidence=0.0
        )
