"""
Synthesizer for comparing and combining outputs from multiple AI models.
"""
from typing import Dict, List, Any, Callable, Optional, Union
import numpy as np


class ConsensusSynthesizer:
    """
    Takes multiple model outputs and generates a consensus result.
    Calculates confidence scores based on agreement between models.
    Handles different types of model outputs (text, structured data).
    Provides methods for analyzing disagreements between models.
    """
    
    def __init__(self):
        self.strategies = {
            "majority_vote": self._majority_vote,
            "weighted_average": self._weighted_average,
            "confidence_weighted": self._confidence_weighted,
            "meta_model": self._meta_model
        }
        self.meta_model = None
        self.output_history = []
        
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
    """
    
    def __init__(self):
        """Initialize the ConsensusSynthesizer with default parameters."""
        self.strategies = {
            "text_consensus": self._text_consensus,
            "structured_consensus": self._structured_consensus,
            "numeric_consensus": self._numeric_consensus,
            "classification_consensus": self._classification_consensus
        }
        self.history = []
        self.confidence_threshold = 0.7
        
    def synthesize(self, model_outputs: Dict[str, Dict[str, Any]], 
                output_type: str = None) -> Dict[str, Any]:
        """
        Generate a consensus result from multiple model outputs.
        
        Args:
            model_outputs: Dictionary mapping model IDs to their outputs
            output_type: Type of output (text, structured, numeric, classification)
            
        Returns:
            Dictionary containing consensus result and confidence score
        """
        # Extract valid outputs
        valid_outputs = {}
        for model_id, output_data in model_outputs.items():
            if output_data.get("status") == "success":
                valid_outputs[model_id] = output_data["output"]
        
        if not valid_outputs:
            raise ValueError("No valid outputs to synthesize")
        
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
        
        # Generate consensus
        result = self.strategies[strategy](valid_outputs)
        
        # Store in history
        self.history.append({
            "inputs": model_outputs,
            "output_type": output_type,
            "result": result
        })
        
        return result
    
    def _text_consensus(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate consensus for text outputs using similarity measures.
        
        Args:
            outputs: Dictionary mapping model IDs to text outputs
            
        Returns:
            Dictionary with consensus text and confidence score
        """
        from collections import Counter
        import difflib
        
        texts = list(outputs.values())
        
        # For exact matches, use majority voting
        counter = Counter(texts)
        most_common = counter.most_common()
        
        if most_common[0][1] > 1:  # If there are exact matches
            consensus_text = most_common[0][0]
            confidence = most_common[0][1] / len(texts)
        else:
            # No exact matches, find most similar text
            similarity_scores = {}
            
            for i, text1 in enumerate(texts):
                similarity_scores[i] = 0
                for j, text2 in enumerate(texts):
                    if i != j:
                        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
                        similarity_scores[i] += similarity
            
            # Select text with highest similarity to others
            best_idx = max(similarity_scores.items(), key=lambda x: x[1])[0]
            consensus_text = texts[best_idx]
            
            # Calculate confidence based on average similarity
            confidence = similarity_scores[best_idx] / (len(texts) - 1) if len(texts) > 1 else 0.5
        
        return {
            "consensus": consensus_text,
            "confidence": confidence,
            "output_type": ModelOutputType.TEXT
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
            "vote_distribution": {label: count/len(labels) for label, count in counter.items()}
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
        texts = list(outputs.values())
        
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
                for i in range(len(models)) for j in range(i+1, len(models))
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
        probabilities = [count/total for count in counter.values()]
        entropy = -sum(p * log2(p) for p in probabilities)
        max_entropy = log2(len(counter))  # Maximum possible entropy
        
        # Normalize entropy to 0-1 range
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "class_distribution": {label: count/total for label, count in counter.items()},
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
