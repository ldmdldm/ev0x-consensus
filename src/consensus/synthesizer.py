"""
Synthesizer for comparing and combining outputs from multiple AI models.
Enhanced with multi-step validation pipeline, factual consistency checking,
confidence scoring, hallucination detection, and historical performance-based weighting.
"""
import json
import logging
import re
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple, Set


class Synthesizer:
    """
    Creates consensus-based decisions from multiple model outputs.
    """

    def __init__(self):
        self.strategies = {
            "majority_vote": self._majority_vote,
            "weighted_average": self._weighted_average,
            "confidence_weighted": self._confidence_weighted,
            "meta_model": self._meta_model
        }
        self.meta_model = None

    def _majority_vote(self, outputs: List[Any], **kwargs) -> Any:
        """
        Simple majority voting mechanism for classification tasks.

        Args:
            outputs: List of outputs from different models

        Returns:
            Most common output
        """
        from collections import Counter

        # Count occurrences of each output
        counter = Counter(outputs)

        # Return the most common output
        return counter.most_common(1)[0][0]

    def _weighted_average(self, outputs: List[Any], weights: List[float] = None, **kwargs) -> Any:
        """
        Weighted average of numerical outputs.

        Args:
            outputs: List of numerical outputs
            weights: List of weights for each output

        Returns:
            Weighted average of outputs
        """
        if weights is None:
            weights = [1.0] * len(outputs)

        if len(weights) != len(outputs):
            raise ValueError("Number of weights must match number of outputs")

        # Normalize weights
        weights_sum = sum(weights)
        normalized_weights = [w / weights_sum for w in weights]

        # Calculate weighted average
        return sum(o * w for o, w in zip(outputs, normalized_weights))

    def _confidence_weighted(self, outputs: List[Any], confidences: List[float], **kwargs) -> Any:
        """
        Weight outputs by their confidence scores.

        Args:
            outputs: List of model outputs
            confidences: List of confidence scores for each output

        Returns:
            Confidence-weighted result
        """
        return self._weighted_average(outputs, confidences)

    def _meta_model(self, outputs: List[Any], **kwargs) -> Any:
        """
        Use a meta-model to combine outputs.

        Args:
            outputs: List of model outputs

        Returns:
            Meta-model prediction
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not set")

        return self.meta_model(outputs)

    def set_meta_model(self, model: Callable):
        """
        Set the meta-model for synthesis.

        Args:
            model: Callable that takes model outputs and returns synthesized result
        """
        self.meta_model = model

    def synthesize(self, model_outputs: Dict[str, Any],
                   strategy: str = "majority_vote", **kwargs) -> Any:
        """
        Synthesize outputs from multiple models into a single result.

        Args:
            model_outputs: Dictionary mapping model IDs to their outputs
            strategy: Strategy to use for synthesis
            **kwargs: Additional arguments for the synthesis strategy

        Returns:
            Synthesized output
        """
        if strategy not in self.strategies:
            raise ValueError(f"Strategy {strategy} not supported")

        # Extract the actual outputs from model results
        outputs = []
        for model_id, output_data in model_outputs.items():
            if output_data["status"] == "success":
                outputs.append(output_data["output"])

        if not outputs:
            raise ValueError("No valid outputs to synthesize")

        return self.strategies[strategy](outputs, **kwargs)


class ModelOutputType:
    """Enum-like class defining possible model output types."""
    TEXT = "text"
    CLASSIFICATION = "classification"
    STRUCTURED = "structured"
    NUMERIC = "numeric"

    @staticmethod
    def detect_type(output: Any) -> str:
        """Automatically detect the type of model output."""
        if isinstance(output, (int, float)):
            return ModelOutputType.NUMERIC
        elif isinstance(output, str):
            return ModelOutputType.TEXT
        elif isinstance(output, (dict, list)):
            return ModelOutputType.STRUCTURED
        else:
            return ModelOutputType.TEXT  # Default to text


class ConsensusSynthesizer:
    """
    Takes multiple model outputs and generates a consensus result with confidence scores.
    Capable of handling different types of outputs and analyzing disagreements.
    Supports iterative feedback loop for consensus refinement and factual verification.
    """

    def __init__(self, config):
        """
        Initialize the synthesizer with configuration.

        Args:
            config: Either a path to a JSON configuration file or a dictionary with configuration
        """
        self.router_client = None  # Will be set externally

        self.strategies = {
            "text_consensus": self._text_consensus,
            "structured_consensus": self._structured_consensus,
            "numeric_consensus": self._numeric_consensus,
            "classification_consensus": self._classification_consensus
        }
        self.history = []
        self.confidence_threshold = 0.7
        self.iteration_history = []
        
        # For hallucination detection
        self.hallucination_patterns = [
            r"I don't know|I'm not sure|I cannot|I do not have|no information|no knowledge",
            r"As of my (?:last update|knowledge cutoff|training|knowledge)",
            r"would need to|I'd need to|I would have to|cannot access|cannot search",
            r"my training data does not|my training does not include|I don't have access to"
        ]
        
        # For factual consistency tracking
        self.fact_check_results = {}
        
        # For model performance tracking
        self.model_performance_history = {}

        # Load configuration
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = json.load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError("Config must be either a path string or a dictionary")

        # Set up configuration parameters
        # Set up configuration parameters
        if self.config:
            self.max_iterations = self.config.get("iterations", {}).get("max_iterations", 3)
            self.improvement_threshold = self.config.get("iterations", {}).get("improvement_threshold", 0.05)
            self.feedback_prompt = self.config.get("iterations", {}).get("feedback_prompt", "")
            self.models = self.config.get("models", [])
            self.aggregator_config = self.config.get("aggregator", {})
            self.citation_verification = self.config.get("citation_verification", {"enabled": False})
            self.validation_pipeline = self.config.get("validation_pipeline", {"enabled": True})
            self.hallucination_detection = self.config.get("hallucination_detection", {"enabled": True})
            self.factual_consistency = self.config.get("factual_consistency", {"enabled": True})
            self.model_weighting = self.config.get("model_weighting", {"use_historical_performance": True})
        else:
            # Default values if no configuration is found
            self.max_iterations = 3
            self.improvement_threshold = 0.05
            self.feedback_prompt = ""
            self.models = []
            self.aggregator_config = {}
            self.citation_verification = {"enabled": False}
            self.validation_pipeline = {"enabled": True}
            self.hallucination_detection = {"enabled": True}
            self.factual_consistency = {"enabled": True}
            self.model_weighting = {"use_historical_performance": True}
        # Initialize factual verification if enabled
        if self.citation_verification.get("enabled", False):
            try:
                from src.factual.citation import CitationVerifier
                self.citation_verifier = CitationVerifier()
            except ImportError:
                import logging
                logging.warning("Citation verification enabled but CitationVerifier could not be imported")
                self.citation_verification["enabled"] = False

    def synthesize(self, model_outputs: Dict[str, Dict[str, Any]],
                   output_type: str = None) -> Dict[str, Any]:
        """
        Generate a consensus result from multiple model outputs.
        Enhanced with multi-step validation pipeline, hallucination detection,
        and improved confidence scoring.

        Args:
            model_outputs: Dictionary mapping model IDs to their outputs
            output_type: Type of output (text, structured, numeric, classification)

        Returns:
            Dictionary containing consensus result and confidence score
        """
        # Extract valid outputs
        valid_outputs = {}
        model_confidences = {}
        
        # First validation step: Check for basic validity and extract outputs
        for model_id, output_data in model_outputs.items():
            if output_data.get("status") == "success":
                # Extract the output
                model_output = output_data["output"]
                valid_outputs[model_id] = model_output
                
                # Calculate initial confidence based on historical performance if available
                if self.model_weighting.get("use_historical_performance", True) and model_id in self.model_performance_history:
                    model_confidences[model_id] = self.model_performance_history[model_id].get("average_score", 0.5)
                else:
                    model_confidences[model_id] = 0.5  # Default confidence

        if not valid_outputs:
            raise ValueError("No valid outputs to synthesize")
            
        # Second validation step: Detect hallucinations if enabled
        if self.hallucination_detection.get("enabled", True):
            for model_id, output in valid_outputs.items():
                if isinstance(output, str):
                    hallucination_score = self._detect_hallucinations(output)
                    # Adjust confidence based on hallucination detection
                    if model_id in model_confidences:
                        # Reduce confidence if hallucinations detected
                        model_confidences[model_id] *= (1 - hallucination_score)

        # Auto-detect output type if not specified
        if output_type is None:
            first_output = next(iter(valid_outputs.values()))
            output_type = ModelOutputType.detect_type(first_output)

        # Select appropriate strategy
        if output_type == ModelOutputType.TEXT:
            strategy = "text_consensus"
        elif output_type == ModelOutputType.STRUCTURED:
            strategy = "structured_consensus"
        elif output_type == ModelOutputType.NUMERIC:
            strategy = "numeric_consensus"
        elif output_type == ModelOutputType.CLASSIFICATION:
            strategy = "classification_consensus"
        else:
            strategy = "text_consensus"  # Default to text

        if strategy not in self.strategies:
            raise ValueError(f"Strategy {strategy} not supported")

        # Third validation step: Check factual consistency if enabled
        # This helps determine model weights
        if self.factual_consistency.get("enabled", True) and output_type == ModelOutputType.TEXT:
            factual_scores = self._check_factual_consistency(valid_outputs)
            for model_id, score in factual_scores.items():
                if model_id in model_confidences:
                    # Blend in factual consistency score
                    model_confidences[model_id] = 0.7 * model_confidences[model_id] + 0.3 * score

        # Final validation step: Generate weighted consensus based on all validations
        result = self.strategies[strategy](valid_outputs, weights=model_confidences)
        
        # Update the model performance history based on this consensus generation
        self._update_model_performance(valid_outputs, result, model_confidences)

        # Store in history
        self.history.append({
            "inputs": model_outputs,
            "output_type": output_type,
            "result": result,
            "model_confidences": model_confidences
        })

        return result
    def _text_consensus(self, outputs: Dict[str, Any], weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Generate consensus for text outputs using similarity measures.
        Enhanced with weighted similarity calculations.

        Args:
            outputs: Dictionary mapping model IDs to text outputs
            weights: Dictionary mapping model IDs to their confidence weights

        Returns:
            Dictionary with consensus text and confidence score
        """
        from collections import Counter
        import difflib
        
        # Initialize weights if not provided
        if not weights:
            weights = {model_id: 1.0 for model_id in outputs}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            norm_weights = {k: v/total_weight for k, v in weights.items()}
        else:
            norm_weights = {k: 1.0/len(outputs) for k in outputs}

        texts = list(outputs.values())
        model_ids = list(outputs.keys())

        # For exact matches, use weighted majority voting
        counter = Counter(texts)
        most_common = counter.most_common()

        if most_common[0][1] > 1:  # If there are exact matches
            # Find which models contributed to the most common text
            consensus_text = most_common[0][0]
            contributing_models = [model_id for model_id, text in outputs.items() if text == consensus_text]
            
            # Calculate weighted confidence
            weighted_count = sum(norm_weights.get(model_id, 0) for model_id in contributing_models)
            confidence = weighted_count
        else:
            # No exact matches, find text with highest weighted similarity
            similarity_scores = {}
            
            # Calculate similarity matrix for all text pairs
            similarity_matrix = {}
            for model_i, text_i in outputs.items():
                for model_j, text_j in outputs.items():
                    if model_i != model_j:
                        if (model_i, model_j) not in similarity_matrix and (model_j, model_i) not in similarity_matrix:
                            # Calculate similarity score
                            similarity = difflib.SequenceMatcher(None, text_i, text_j).ratio()
                            similarity_matrix[(model_i, model_j)] = similarity
            
            # Calculate weighted similarity score for each text
            for model_id, text in outputs.items():
                weighted_sim_score = 0
                total_weight = 0
                
                for other_model, other_text in outputs.items():
                    if model_id != other_model:
                        if (model_id, other_model) in similarity_matrix:
                            sim = similarity_matrix[(model_id, other_model)]
                        else:
                            sim = similarity_matrix[(other_model, model_id)]
                        
                        # Weight by the other model's confidence
                        weight = norm_weights[other_model]
                        weighted_sim_score += sim * weight
                        total_weight += weight
                
                if total_weight > 0:
                    # Normalize the similarity score
                    similarity_scores[model_id] = weighted_sim_score / total_weight
                else:
                    similarity_scores[model_id] = 0
            
            # Adjust similarity score by model's own weight
            weighted_similarity = {
                model_id: score * norm_weights[model_id]
                for model_id, score in similarity_scores.items()
            }
            
            # Find the text with highest weighted similarity
            if weighted_similarity:
                best_model = max(weighted_similarity, key=weighted_similarity.get)
                consensus_text = outputs[best_model]
                
                # Calculate confidence based on average similarity to this text
                avg_similarity = similarity_scores[best_model]
                confidence = avg_similarity
            else:
                # Fallback if no similarities could be calculated
                consensus_text = texts[0]
                confidence = 0.3
                
        return {
            "consensus": consensus_text,
            "confidence": confidence,
            "output_type": ModelOutputType.TEXT,
            "contributing_models": [model_id for model_id, text in outputs.items() if text == consensus_text]
        }


    def _structured_consensus(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate consensus for structured data (dictionaries or lists).

        Args:
            outputs: Dictionary mapping model IDs to structured outputs

        Returns:
            Dictionary with consensus structure and confidence score
        """
        import json

        # Convert all outputs to string representation for comparison
        str_outputs = {model: json.dumps(output, sort_keys=True)
                       for model, output in outputs.items()}

        # Get text consensus using the string representation
        text_result = self._text_consensus(str_outputs)

        # Convert consensus back to structured data
        try:
            consensus_struct = json.loads(text_result["consensus"])
            return {
                "consensus": consensus_struct,
                "confidence": text_result["confidence"],
                "output_type": ModelOutputType.STRUCTURED
            }
        except json.JSONDecodeError:
            # Fallback if conversion fails
            return text_result

    def _numeric_consensus(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate consensus for numeric outputs.

        Args:
            outputs: Dictionary mapping model IDs to numeric outputs

        Returns:
            Dictionary with consensus value and confidence score
        """
        import numpy as np
        import math

        values = list(outputs.values())

        if not all(isinstance(v, (int, float)) for v in values):
            # Convert all to float if possible
            try:
                values = [float(v) for v in values]
            except (ValueError, TypeError):
                # If conversion fails, use text consensus
                return self._text_consensus(outputs)

        # Calculate mean and standard deviation
        mean_val = np.mean(values)
        std_val = np.std(values) if len(values) > 1 else 0

        # Calculate confidence based on coefficient of variation
        if mean_val != 0:
            cv = std_val / abs(mean_val)  # Coefficient of variation
            confidence = math.exp(-cv) if cv > 0 else 1.0  # Transform to 0-1 range
        else:
            confidence = 1.0 if std_val == 0 else 0.5

        return {
            "consensus": mean_val,
            "confidence": confidence,
            "output_type": ModelOutputType.NUMERIC,
            "std_deviation": std_val
        }

    def _classification_consensus(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate consensus for classification outputs.

        Args:
            outputs: Dictionary mapping model IDs to classification labels

        Returns:
            Dictionary with consensus class and confidence score
        """
        from collections import Counter

        labels = list(outputs.values())
        counter = Counter(labels)
        most_common = counter.most_common()

        consensus_label = most_common[0][0]
        confidence = most_common[0][1] / len(labels)

        return {
            "consensus": consensus_label,
            "confidence": confidence,
            "output_type": ModelOutputType.CLASSIFICATION,
            "vote_distribution": {label: count / len(labels) for label, count in counter.items()}
        }

    def analyze_disagreements(self, model_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze disagreements between model outputs.

        Args:
            model_outputs: Dictionary mapping model IDs to their outputs

        Returns:
            Dictionary with disagreement analysis
        """
        # Extract valid outputs
        valid_outputs = {}
        for model_id, output_data in model_outputs.items():
            if output_data.get("status") == "success":
                valid_outputs[model_id] = output_data["output"]

        if not valid_outputs:
            raise ValueError("No valid outputs to analyze")

        # Detect output type
        first_output = next(iter(valid_outputs.values()))
        output_type = ModelOutputType.detect_type(first_output)

        # Analyze based on output type
        if output_type == ModelOutputType.TEXT:
            return self._analyze_text_disagreements(valid_outputs)
        elif output_type == ModelOutputType.STRUCTURED:
            return self._analyze_structured_disagreements(valid_outputs)
        elif output_type == ModelOutputType.NUMERIC:
            return self._analyze_numeric_disagreements(valid_outputs)
        elif output_type == ModelOutputType.CLASSIFICATION:
            return self._analyze_classification_disagreements(valid_outputs)
        else:
            return self._analyze_text_disagreements(valid_outputs)

    def _analyze_text_disagreements(self, outputs: Dict[str, str]) -> Dict[str, Any]:
        """Analyze disagreements between text outputs."""
        import difflib
        import numpy as np

        models = list(outputs.keys())

        # Calculate pairwise similarities
        similarity_matrix = np.zeros((len(models), len(models)))

        for i, (model1, text1) in enumerate(outputs.items()):
            for j, (model2, text2) in enumerate(outputs.items()):
                if i != j:
                    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
                    similarity_matrix[i, j] = similarity

        # Calculate average similarity for each model
        avg_similarities = np.mean(similarity_matrix, axis=1)

        # Identify outliers (models with low avg similarity)
        threshold = np.mean(avg_similarities) - np.std(avg_similarities)
        outliers = [models[i] for i, sim in enumerate(avg_similarities) if sim < threshold]

        return {
            "agreement_score": np.mean(similarity_matrix),
            "outlier_models": outliers,
            "pairwise_similarities": {
                f"{models[i]}-{models[j]}": similarity_matrix[i, j]
                for i in range(len(models)) for j in range(i + 1, len(models))
            }
        }

    def _analyze_structured_disagreements(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze disagreements between structured outputs."""
        import json

        # Convert to string for comparison
        str_outputs = {model: json.dumps(output, sort_keys=True)
                       for model, output in outputs.items()}

        # Use text disagreement analysis
        return self._analyze_text_disagreements(str_outputs)

    def _analyze_numeric_disagreements(self, outputs: Dict[str, float]) -> Dict[str, Any]:
        """Analyze disagreements between numeric outputs."""
        import numpy as np

        values = list(outputs.values())
        models = list(outputs.keys())

        mean_val = np.mean(values)
        std_val = np.std(values)

        # Identify outliers (values more than 2 std dev from mean)
        z_scores = np.abs((values - mean_val) / std_val) if std_val > 0 else np.zeros(len(values))
        outliers = [models[i] for i, z in enumerate(z_scores) if z > 2]

        return {
            "mean": mean_val,
            "std_deviation": std_val,
            "coefficient_of_variation": std_val / abs(mean_val) if mean_val != 0 else 0,
            "outlier_models": outliers,
            "z_scores": {models[i]: z_scores[i] for i in range(len(models))}
        }

    def _analyze_classification_disagreements(self, outputs: Dict[str, str]) -> Dict[str, Any]:
        """Analyze disagreements between classification outputs."""
        from collections import Counter

        labels = list(outputs.values())
        counter = Counter(labels)

        # Calculate entropy as a measure of disagreement
        from math import log2
        total = len(labels)
        probabilities = [count / total for count in counter.values()]
        entropy = -sum(p * log2(p) for p in probabilities)
        max_entropy = log2(len(counter))  # Maximum possible entropy

        # Normalize entropy to 0-1 range
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "class_distribution": {label: count / total for label, count in counter.items()},
            "agreement_score": 1 - normalized_entropy
        }

    def get_confidence_threshold(self) -> float:
        """Get the current confidence threshold."""
        return self.confidence_threshold

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold."""
        if 0 <= threshold <= 1:
            self.confidence_threshold = threshold
        else:
            raise ValueError("Confidence threshold must be between 0 and 1")

    def get_history(self) -> List[Dict[str, Any]]:
        """Retrieve the history of consensus operations."""
        return self.history

    def update_weights(self, weights):
        """
        Update the weights used for consensus generation.
        
        Args:
            weights: Dictionary mapping model IDs to their weights
        """
        # Store the weights for future use
        self.model_weights = weights

    def _load_configuration(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from the specified JSON file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary containing configuration parameters
        """
        import os
        import json
        import logging

        logger = logging.getLogger(__name__)

        try:
            if not os.path.exists(config_path):
                # Try relative to project root
                alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), config_path)
                if os.path.exists(alt_path):
                    config_path = alt_path
                else:
                    logger.warning(f"Configuration file not found at {config_path}")
                    return {}

            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            return {}

    async def synthesize_with_iterations(self, query: str, model_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a consensus result from multiple model outputs with iterative refinement.

        Args:
            query: The original query or prompt
            model_outputs: Dictionary mapping model IDs to their outputs

        Returns:
            Dictionary containing the final consensus result and metadata
        """
        import logging
        import time
        from copy import deepcopy

        logger = logging.getLogger(__name__)

        # Initialize iteration tracking
        self.iteration_history = []
        iterations_performed = 0
        previous_score = 0
        best_result = None

        # Get initial consensus
        start_time = time.time()
        initial_consensus = self.synthesize(model_outputs)

        # Store initial result
        self.iteration_history.append({
            "iteration": 0,
            "consensus": deepcopy(initial_consensus),
            "model_outputs": deepcopy(model_outputs),
            "improvement": 0,
            "feedback": None
        })

        best_result = initial_consensus
        previous_score = self._evaluate_consensus_quality(initial_consensus)

        logger.info(f"Initial consensus quality score: {previous_score:.4f}")

        # Perform iterative improvement if configured
        if self.max_iterations > 1:
            aggregator_model_id = self.aggregator_config.get("model_id", "")

            # Get the aggregator model configuration
            aggregator_params = self.aggregator_config.get("parameters", {})

            for iteration in range(1, self.max_iterations):
                try:
                    # Generate feedback on the current consensus
                    feedback = await self._generate_feedback(query, best_result, model_outputs)

                    # Skip if no meaningful feedback was generated
                    if not feedback or feedback.strip() == "":
                        logger.info(f"No meaningful feedback generated for iteration {iteration}, stopping")
                        break

                    # Create a new prompt that includes feedback
                    feedback_points = feedback.strip()
                    improvement_prompt = self.feedback_prompt.format(feedback_points=feedback_points)

                    # Build a new set of model outputs that includes the refinement prompt
                    refined_outputs = await self._get_refinement_outputs(
                        query,
                        best_result,
                        improvement_prompt,
                        aggregator_model_id,
                        aggregator_params
                    )

                    # Synthesize a new consensus
                    new_consensus = self.synthesize(refined_outputs)

                    # Evaluate the new consensus
                    new_score = self._evaluate_consensus_quality(new_consensus)
                    improvement = new_score - previous_score

                    logger.info(f"Iteration {iteration} - New score: {new_score:.4f}, Improvement: {improvement:.4f}")

                    # Store this iteration
                    self.iteration_history.append({
                        "iteration": iteration,
                        "consensus": deepcopy(new_consensus),
                        "model_outputs": deepcopy(refined_outputs),
                        "improvement": improvement,
                        "feedback": feedback
                    })

                    # Update tracking variables
                    iterations_performed += 1

                    # Check if we've made enough improvement to continue
                    if improvement > self.improvement_threshold:
                        best_result = new_consensus
                        previous_score = new_score
                    else:
                        logger.info(
                            f"Improvement below threshold ({
                                improvement:.4f} < {
                                self.improvement_threshold}), stopping")
                        break

                except Exception as e:
                    logger.error(f"Error in iteration {iteration}: {str(e)}")
                    break

        # Verify factual claims if enabled
        if self.citation_verification.get("enabled", False) and hasattr(self, "citation_verifier"):
            try:
                consensus_text = best_result.get("consensus", "")
                if isinstance(consensus_text, str) and consensus_text.strip():
                    verification_results = self.citation_verifier.verify_text(consensus_text)

                    # Add factual verification results
                    best_result["factual_verification"] = {
                        "verified_claims": verification_results.get("verified_claims", 0),
                        "total_claims": verification_results.get("total_claims", 0),
                        "verification_rate": verification_results.get("verification_rate", 0),
                        "claim_results": verification_results.get("claim_results", [])
                    }

                    # Add citations to the result
                    if self.citation_verification.get("extract_quotes", False):
                        best_result["citations"] = self._extract_citations_from_results(verification_results)
            except Exception as e:
                logger.error(f"Error during factual verification: {str(e)}")

        # Add metadata to the result
        elapsed_time = time.time() - start_time
        best_result["metadata"] = {
            "iterations_performed": iterations_performed,
            "final_quality_score": previous_score,
            "processing_time_seconds": elapsed_time,
            "improvement_from_initial": previous_score -
            self._evaluate_consensus_quality(initial_consensus) if iterations_performed > 0 else 0}

        return best_result

    async def generate_consensus_completion(self, prompt: str) -> Dict[str, Any]:
        """Generate a completion with consensus from multiple models."""
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Verify router client is initialized
            if not self.router_client:
                raise ValueError("Router client not initialized")

            # Get model configurations
            model_outputs = {}
            for model_config in self.config.get("models", []):
                model_id = model_config.get("id")
                params = model_config.get("params", {})

                if not model_id:
                    continue

                # Generate completion with this model
                response = await self.router_client.generate_async(
                    model=model_id,
                    prompt=prompt,
                    **params
                )

                model_outputs[model_id] = {
                    "status": "success" if response else "error",
                    "output": response.get("text", "") if response else ""
                }

            # Generate consensus with iterations
            consensus_result = await self.synthesize_with_iterations(prompt, model_outputs)

            # Format the result for the tests
            result = consensus_result.copy()
            # Ensure consensus_output key is present as expected by tests
            result["consensus_output"] = consensus_result.get("consensus", "")

            # Return the full consensus result
            return result
        except Exception as e:
            logger.error(f"Error generating consensus completion: {e}")
            return {"consensus_output": "", "error": str(e)}

    async def generate_consensus_chat_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate a chat completion with consensus from multiple models."""
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Verify router client is initialized
            if not self.router_client:
                raise ValueError("Router client not initialized")

            # Get model configurations
            model_outputs = {}
            for model_config in self.config.get("models", []):
                model_id = model_config.get("id")
                params = model_config.get("params", {})

                if not model_id:
                    continue

                # Generate chat completion with this model
                response = await self.router_client.chat_async(
                    model=model_id,
                    messages=messages,
                    **params
                )

                model_outputs[model_id] = {
                    "status": "success" if response else "error",
                    "output": response.get("text", "") if response else ""
                }

            # Extract the prompt from the last message
            last_message = messages[-1] if messages else {"content": ""}
            prompt = last_message.get("content", "")

            # Generate consensus with iterations
            consensus_result = await self.synthesize_with_iterations(prompt, model_outputs)

            # Format the result for the tests
            result = consensus_result.copy()
            # Ensure consensus_output key is present as expected by tests
            result["consensus_output"] = consensus_result.get("consensus", "")

            # Return the full consensus result
            return result
        except Exception as e:
            logger.error(f"Error generating consensus chat completion: {e}")
            return {"consensus_output": "", "error": str(e)}

    async def _generate_feedback(
            self, query: str, consensus_result: Dict[str, Any], model_outputs: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate feedback for improving the current consensus.

        Args:
            query: The original query
            consensus_result: The current consensus result
            model_outputs: The original model outputs

        Returns:
            String containing feedback points for improvement
        """
        import logging
        import re

        logger = logging.getLogger(__name__)

        try:
            # Use the aggregator model to generate feedback
            aggregator_model_id = self.aggregator_config.get("model_id", "")
            if not aggregator_model_id:
                logger.warning("No aggregator model configured for feedback generation")
                return ""

            # Extract consensus text
            consensus_text = consensus_result.get("consensus", "")
            if not isinstance(consensus_text, str) or not consensus_text.strip():
                logger.warning("Empty consensus text, cannot generate feedback")
                return ""

            # Construct the feedback prompt
            feedback_prompt = """
            You are tasked with analyzing a consensus answer derived from multiple AI model responses.
            Identify 3-5 specific ways this consensus answer could be improved:

            Original Question:
            {query}

            Current Consensus Answer:
            {consensus}

            Analyze this consensus answer and identify areas for improvement such as:
            1. Missing important information that was present in some model responses
            2. Logical inconsistencies or contradictions
            3. Factual errors or misleading statements
            4. Unclear explanations that need more detail
            5. Important nuances or caveats that were omitted

            Return ONLY a numbered list of specific improvement points, without any introduction or conclusion.
            Be concise and direct in your feedback.
            """.strip().format(query=query, consensus=consensus_text)

            # Generate feedback using the async method
            if not self.router_client:
                logger.warning("Router client not initialized")
                return ""

            # Generate feedback
            response = await self.router_client.generate_async(
                model=aggregator_model_id,
                prompt=feedback_prompt,
                max_tokens=1024,
                temperature=0.7
            )

            if not response:
                logger.warning("Failed to generate feedback")
                return ""

            # Extract the feedback text
            feedback_text = response.get("text", "").strip()

            # Clean up the feedback to ensure it's in the right format
            # Remove any introduction or conclusion paragraphs
            numbered_points = re.findall(r'\d+\.\s+[^\n]+', feedback_text)
            if numbered_points:
                cleaned_feedback = "\n".join(numbered_points)
                return cleaned_feedback
            else:
                return feedback_text

        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}")
            return ""

    def _evaluate_consensus_quality(self, consensus_result: Dict[str, Any]) -> float:
        """
        Evaluate the quality of a consensus result.

        Args:
            consensus_result: Dictionary containing consensus output and metadata

        Returns:
            Quality score between 0 and 1
        """
        try:
            # Initialize quality metrics
            metrics = []

            # 1. Basic validity check
            if not consensus_result or "consensus" not in consensus_result:
                return 0.0

            # 2. Check confidence score if available
            if "confidence" in consensus_result:
                metrics.append(consensus_result["confidence"])

            # 3. Check content quality
            consensus_text = str(consensus_result["consensus"])
            if consensus_text:
                # Length metric (penalize extremely short or long outputs)
                optimal_length = 500  # characters
                actual_length = len(consensus_text)
                length_score = min(
                    1.0, actual_length / optimal_length) if actual_length < optimal_length else optimal_length / actual_length
                metrics.append(length_score)

                # Structure metric (check for basic formatting)
                has_structure = bool(
                    consensus_text.strip() and
                    any(char in consensus_text for char in ".!?") and  # Has sentence endings
                    consensus_text[0].isupper()  # Starts with capital letter
                )
                metrics.append(1.0 if has_structure else 0.5)

            # 4. Check verification results if available
            if "factual_verification" in consensus_result:
                verification = consensus_result["factual_verification"]
                if verification.get("total_claims", 0) > 0:
                    verification_score = verification.get("verification_rate", 0)
                    metrics.append(verification_score)

            # Calculate final score
            if metrics:
                # Weight recent verification more heavily
                weights = [1.0] * len(metrics)
                if "factual_verification" in consensus_result:
                    weights[-1] = 2.0  # Double weight for verification

                # Normalize weights
                weights = [w / sum(weights) for w in weights]

                # Calculate weighted average
                final_score = sum(m * w for m, w in zip(metrics, weights))
                return max(0.0, min(1.0, final_score))  # Ensure score is between 0 and 1

            return 0.5  # Default score if no metrics available

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error evaluating consensus quality: {e}")
            return 0.0

    async def _get_refinement_outputs(
        self,
        query: str,
        previous_result: Dict[str, Any],
        improvement_prompt: str,
        aggregator_model_id: str,
        aggregator_params: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get refined outputs from models based on feedback.
        """
        try:
            if not self.router_client:
                raise ValueError("Router client not initialized")

            refined_outputs = {}

            # Format the refinement prompt
            refinement_context = f"""
            Original query: {query}
            Previous answer: {previous_result.get('consensus', '')}

            {improvement_prompt}

            Provide an improved answer that addresses these points:
            """

            # Get refined outputs from each model
            for model_config in self.models:
                model_id = model_config.get("id")
                params = model_config.get("params", {})

                if not model_id:
                    continue

                # Get refined output from this model
                response = await self.router_client.generate_async(
                    model=model_id,
                    prompt=refinement_context,
                    **{**params, **aggregator_params}
                )

                refined_outputs[model_id] = {
                    "status": "success" if response else "error",
                    "output": response.get("text", "") if response else ""
                }

            return refined_outputs

        except Exception as e:
            logging.error(f"Error getting refinement outputs: {e}")
            return {}

    def _detect_hallucinations(self, text: str) -> float:
        """
        Detect potential hallucinations in model output.
        
        Args:
            text: The text to analyze for hallucinations
            
        Returns:
            A score between 0.0 and 1.0 representing the likelihood of hallucinations
            (0.0 = no hallucinations detected, 1.0 = high likelihood of hallucinations)
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0
            
        # Check if text contains hedging or uncertainty phrases
        hallucination_score = 0.0
        total_patterns = len(self.hallucination_patterns)
        
        for pattern in self.hallucination_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                hallucination_score += 1.0 / total_patterns
        
        # Check for unsupported factual claims indicators
        factual_claim_indicators = [
            r"according to (?:recent|the latest|new)",
            r"studies show|research indicates|experts say",
            r"in \d{4},|as of \d{4}",
            r"statistics indicate|polls suggest"
        ]
        
        factual_citations = [
            r"according to \[.*?\]|cited in \[.*?\]",
            r"\(.*?\d{4}.*?\)",
            r"et al\.",
            r"reference \d+",
            r"citation \d+"
        ]
        
        # Count unsupported factual claims
        unsupported_claims = 0
        supported_claims = 0
        
        for indicator in factual_claim_indicators:
            if re.search(indicator, text, re.IGNORECASE):
                unsupported_claims += 1
                # Check if claim is supported by a citation
                for citation_pattern in factual_citations:
                    if re.search(citation_pattern, text, re.IGNORECASE):
                        supported_claims += 1
                        unsupported_claims -= 1
                        break
                        
        # Factor in unsupported claims to the hallucination score
        if unsupported_claims > 0:
            claim_ratio = max(0, unsupported_claims - supported_claims) / max(1, unsupported_claims + supported_claims)
            hallucination_score += claim_ratio * 0.3  # 30% weight for unsupported claims
            
        # Cap the score at 1.0
        return min(1.0, hallucination_score)
        
    def _check_factual_consistency(self, model_outputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the factual consistency of model outputs.
        
        Args:
            model_outputs: Dictionary mapping model IDs to their text outputs
            
        Returns:
            Dictionary mapping model IDs to factual consistency scores (0.0-1.0)
        """
        factual_scores = {}
        
        # Extract only text outputs for processing
        text_outputs = {}
        for model_id, output in model_outputs.items():
            if isinstance(output, str):
                text_outputs[model_id] = output
            elif isinstance(output, dict) and "text" in output:
                text_outputs[model_id] = output["text"]
            else:
                continue
                
        if not text_outputs:
            return {}
            
        # Find consistent facts across outputs
        fact_consistency = {}
        
        # Extract factual statements
        # This is a simplified approach - in production, use NLP to extract subject-predicate-object triples
        for model_id, text in text_outputs.items():
            # Simple sentence extraction as potential factual statements
            sentences = re.split(r'[.!?]', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter short sentences
            
            # Skip if no valid sentences found
            if not sentences:
                factual_scores[model_id] = 0.5  # Neutral score
                continue
                
            # Compare with other model outputs for consistency
            consistent_facts = 0
            total_facts = len(sentences)
            
            for sentence in sentences:
                # Check if this fact appears in other outputs
                sentence_consistency = []
                for other_id, other_text in text_outputs.items():
                    if other_id == model_id:
                        continue
                        
                    # Check for similar statements in other model outputs
                    # Use fuzzy matching for more robust comparison
                    max_similarity = 0
                    for other_sentence in re.split(r'[.!?]', other_text):
                        other_sentence = other_sentence.strip()
                        if len(other_sentence) > 20:
                            similarity = difflib.SequenceMatcher(None, sentence.lower(), other_sentence.lower()).ratio()
                            max_similarity = max(max_similarity, similarity)
                            
                    # Consider facts similar if similarity exceeds threshold
                    if max_similarity > 0.6:  # Threshold for similarity
                        sentence_consistency.append(max_similarity)
                        
                # Calculate average consistency for this statement
                if sentence_consistency:
                    avg_consistency = sum(sentence_consistency) / len(sentence_consistency)
                    consistent_facts += avg_consistency
            
            # Calculate overall factual consistency score
            if total_facts > 0:
                factual_scores[model_id] = consistent_facts / total_facts
            else:
                factual_scores[model_id] = 0.5
                
        # Store results for future reference
        self.fact_check_results = factual_scores
        return factual_scores
        
    def _update_model_performance(self, model_outputs: Dict[str, Any], 
                                 consensus_result: Dict[str, Any],
                                 model_confidences: Dict[str, float]) -> None:
        """
        Update historical performance tracking for models.
        
        Args:
            model_outputs: Dictionary mapping model IDs to their outputs
            consensus_result: The consensus result generated
            model_confidences: Dictionary mapping model IDs to their confidence scores
        """
        if not consensus_result or "consensus" not in consensus_result:
            return
            
        consensus_output = consensus_result["consensus"]
        output_type = consensus_result.get("output_type", ModelOutputType.TEXT)
        
        # Calculate similarity between each model's output and the consensus
        for model_id, model_output in model_outputs.items():
            # Skip models without valid outputs
            if not model_output:
                continue
                
            # Initialize model history if not present
            if model_id not in self.model_performance_history:
                self.model_performance_history[model_id] = {
                    "total_score": 0,
                    "count": 0,
                    "average_score": 0,
                    "recent_scores": [],
                    "by_output_type": {}
                }
                
            # Calculate similarity score based on output type
            similarity_score = 0
            
            if output_type == ModelOutputType.TEXT and isinstance(model_output, str) and isinstance(consensus_output, str):
                # Text similarity
                similarity_score = difflib.SequenceMatcher(None, model_output, consensus_output).ratio()
            elif output_type == ModelOutputType.STRUCTURED:
                # Structured data similarity
                try:
                    import json
                    model_json = json.dumps(model_output, sort_keys=True)
                    consensus_json = json.dumps(consensus_output, sort_keys=True)
                    similarity_score = difflib.SequenceMatcher(None, model_json, consensus_json).ratio()
                except:
                    similarity_score = 0
            elif output_type == ModelOutputType.NUMERIC and isinstance(model_output, (int, float)) and isinstance(consensus_output, (int, float)):
                # Numeric similarity based on relative difference
                max_val = max(abs(model_output), abs(consensus_output))
                if max_val > 0:
                    diff = abs(model_output - consensus_output) / max_val
                    similarity_score = max(0, 1 - diff)
                else:
                    similarity_score = 1 if model_output == consensus_output else 0
            elif output_type == ModelOutputType.CLASSIFICATION:
                # Classification similarity - exact match only
                similarity_score = 1.0 if model_output == consensus_output else 0.0
                
            # Apply confidence weighting if available
            confidence_weight = model_confidences.get(model_id, 0.5) if model_confidences else 0.5
            
            # Weighted performance score
            performance_score = similarity_score * (0.7 + 0.3 * confidence_weight)
            
            # Update history
            model_history = self.model_performance_history[model_id]
            model_history["total_score"] += performance_score
            model_history["count"] += 1
            model_history["average_score"] = model_history["total_score"] / model_history["count"]
            
            # Track recent scores (keep last 10)
            model_history["recent_scores"].append(performance_score)
            if len(model_history["recent_scores"]) > 10:
                model_history["recent_scores"] = model_history["recent_scores"][-10:]
                
            # Track by output type
            if output_type not in model_history["by_output_type"]:
                model_history["by_output_type"][output_type] = {
                    "total_score": 0,
                    "count": 0,
                    "average_score": 0
                }
                
            type_history = model_history["by_output_type"][output_type]
            type_history["total_score"] += performance_score
            type_history["count"] += 1
            type_history["average_score"] = type_history["total_score"] / type_history["count"]
            
    def get_model_performance(self, model_id: str = None) -> Dict[str, Any]:
        """
        Get historical performance metrics for models.
        
        Args:
            model_id: Optional specific model ID to retrieve metrics for
            
        Returns:
            Dictionary containing performance metrics
        """
        if model_id:
            return self.model_performance_history.get(model_id, {})
        else:
            return self.model_performance_history
            
    def get_best_models(self, output_type: str = None, top_n: int = 3) -> List[str]:
        """
        Get the best performing models overall or for a specific output type.
        
        Args:
            output_type: Optional specific output type to filter by
            top_n: Number of top models to return
            
        Returns:
            List of model IDs sorted by performance
        """
        if not self.model_performance_history:
            return []
            
        if output_type:
            # Get models with scores for this output type
            models_with_type = {}
            for model_id, history in self.model_performance_history.items():
                if output_type in history.get("by_output_type", {}):
                    type_score = history["by_output_type"][output_type]["average_score"]
                    models_with_type[model_id] = type_score
                    
            # Sort by score and return top N
            return sorted(models_with_type.keys(), 
