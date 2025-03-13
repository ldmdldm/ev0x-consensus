"""
Implementation of Shapley value calculations for model reward allocation.
"""
from typing import Dict, List, Any, Callable, Set, Tuple
import itertools
import math


class ShapleyValueCalculator:
    """
    Calculates Shapley values to determine the contribution of each model
    to the final ensemble performance.
    """

    def __init__(self, evaluation_function: Callable[[Set[str]], float]):
        """
        Initialize the Shapley value calculator.

        Args:
            evaluation_function: Function that takes a set of model IDs and returns
                                a performance score
        """
        self.evaluation_function = evaluation_function

    def _get_all_subsets(self, model_ids: List[str]) -> List[Set[str]]:
        """
        Generate all possible subsets of models.

        Args:
            model_ids: List of model IDs

        Returns:
            List of all possible subsets (as sets)
        """
        all_subsets = []
        for i in range(len(model_ids) + 1):
            for subset in itertools.combinations(model_ids, i):
                all_subsets.append(set(subset))
        return all_subsets

    def calculate_shapley_values(self, model_ids: List[str]) -> Dict[str, float]:
        """
        Calculate Shapley values for each model.

        Args:
            model_ids: List of model IDs to calculate Shapley values for

        Returns:
            Dictionary mapping model IDs to their Shapley values
        """
        n = len(model_ids)
        shapley_values = {model_id: 0.0 for model_id in model_ids}

        for model_id in model_ids:
            # For each possible subset that doesn't include the current model
            other_models = [m for m in model_ids if m != model_id]

            for subset_size in range(len(other_models) + 1):
                for subset in itertools.combinations(other_models, subset_size):
                    subset_set = set(subset)

                    # Calculate marginal contribution
                    with_model = subset_set.union({model_id})
                    without_model = subset_set

                    marginal_contribution = (
                        self.evaluation_function(with_model) -
                        self.evaluation_function(without_model)
                    )

                    # Calculate the weight for this subset size
                    weight = (math.factorial(subset_size) *
                              math.factorial(n - subset_size - 1) /
                              math.factorial(n))

                    shapley_values[model_id] += weight * marginal_contribution

        return shapley_values

    def normalize_rewards(self, shapley_values: Dict[str, float],
                          total_reward: float = 1.0) -> Dict[str, float]:
        """
        Normalize Shapley values to distribute a fixed reward.

        Args:
            shapley_values: Dictionary of Shapley values
            total_reward: Total reward to distribute

        Returns:
            Dictionary mapping model IDs to their rewards
        """
        total_shapley = sum(shapley_values.values())

        if total_shapley == 0:
            # Equal distribution if all Shapley values are 0
            n = len(shapley_values)
            return {model_id: total_reward / n for model_id in shapley_values}

        return {model_id: (value / total_shapley) * total_reward
                for model_id, value in shapley_values.items()}


class ShapleyCalculator:
    """
    Advanced Shapley value calculator with support for different types of model outputs
    and additional analysis functionality.
    """

    def __init__(self, evaluation_function: Callable[[Set[str]], float] = None):
        """
        Initialize the Shapley calculator.

        Args:
            evaluation_function: Optional function that takes a set of model IDs and returns
                            a performance score. If None, must be provided when calling
                            calculation methods.
        """
        self.evaluation_function = evaluation_function
        self.base_calculator = ShapleyValueCalculator(evaluation_function) if evaluation_function else None
        self.cached_values = {}

    def calculate_shapley_values(self, model_ids: List[str],
                                 eval_function: Callable[[Set[str]], float] = None) -> Dict[str, float]:
        """
        Calculate Shapley values using the base calculator.

        Args:
            model_ids: List of model IDs to calculate Shapley values for
            eval_function: Optional override for the evaluation function

        Returns:
            Dictionary mapping model IDs to their Shapley values
        """
        eval_func = eval_function or self.evaluation_function
        if not eval_func:
            raise ValueError("No evaluation function provided")

        calculator = self.base_calculator if self.base_calculator else ShapleyValueCalculator(eval_func)
        shapley_values = calculator.calculate_shapley_values(model_ids)

        # Cache the calculated values for later analysis
        cache_key = tuple(sorted(model_ids))
        self.cached_values[cache_key] = shapley_values

        return shapley_values

    def calculate_for_text_outputs(self, model_outputs: Dict[str, dict],
                                   reference_output: str = None) -> Dict[str, float]:
        """
        Calculate Shapley values for text outputs by comparing them to a reference
        or to each other based on similarity metrics.

        Args:
            model_outputs: Dictionary mapping model IDs to their outputs, where each output
                          is a dict with 'status' and 'output' fields
            reference_output: Optional reference output to compare against

        Returns:
            Dictionary mapping model IDs to their Shapley values
        """
        model_ids = list(model_outputs.keys())

        def text_evaluation(model_subset: Set[str]) -> float:
            if not model_subset:
                return 0.0

            # If we have a reference, compare to it
            if reference_output:
                # Extract the actual text output from the nested structure
                subset_outputs = [model_outputs[model_id].get("output", "") if isinstance(model_outputs[model_id], dict)
                                  else model_outputs[model_id] for model_id in model_subset]
                # Use simple average similarity for this implementation
                # In practice, more sophisticated NLP metrics would be used
                similarities = [self._text_similarity(output, reference_output)
                                for output in subset_outputs]
                return sum(similarities) / len(similarities)

            # Without reference, measure internal agreement
            # Extract the actual text output from the nested structure
            subset_outputs = [model_outputs[model_id].get("output", "") if isinstance(model_outputs[model_id], dict)
                              else model_outputs[model_id] for model_id in model_subset]
            if len(subset_outputs) == 1:
                return 0.5  # Neutral score for single model

            # Calculate agreement between all pairs
            agreement_scores = []
            for i in range(len(subset_outputs)):
                for j in range(i + 1, len(subset_outputs)):
                    agreement_scores.append(
                        self._text_similarity(subset_outputs[i], subset_outputs[j])
                    )

            return sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0

        return self.calculate_shapley_values(model_ids, text_evaluation)

    def calculate_for_numerical_outputs(self, model_outputs: Dict[str, float],
                                        reference_value: float = None) -> Dict[str, float]:
        """
        Calculate Shapley values for numerical outputs by comparing them to a reference
        or determining their contribution to consensus.

        Args:
            model_outputs: Dictionary mapping model IDs to their numerical outputs
            reference_value: Optional reference value to compare against

        Returns:
            Dictionary mapping model IDs to their Shapley values
        """
        model_ids = list(model_outputs.keys())

        def numerical_evaluation(model_subset: Set[str]) -> float:
            if not model_subset:
                return 0.0

            subset_values = [model_outputs[model_id] for model_id in model_subset]

            # If we have a reference value, measure accuracy
            if reference_value is not None:
                errors = [abs(val - reference_value) for val in subset_values]
                # Convert errors to a score (smaller error = higher score)
                max_error = max(max(errors), 0.001)  # Avoid division by zero
                return 1.0 - (sum(errors) / len(errors)) / max_error

            # Without reference, measure consensus/agreement
            if len(subset_values) == 1:
                return 0.5  # Neutral score for single model

            # Calculate standard deviation as a measure of agreement
            mean_value = sum(subset_values) / len(subset_values)
            variance = sum((x - mean_value) ** 2 for x in subset_values) / len(subset_values)
            std_dev = variance ** 0.5

            # Convert to a score (lower std_dev = higher agreement = higher score)
            if std_dev == 0:
                return 1.0  # Perfect agreement
            return 1.0 / (1.0 + std_dev)  # Bounded between 0 and 1

        return self.calculate_shapley_values(model_ids, numerical_evaluation)

    def calculate_for_classification_outputs(self, model_outputs: Dict[str, Any],
                                             reference_class: Any = None) -> Dict[str, float]:
        """
        Calculate Shapley values for classification outputs by measuring agreement
        or accuracy compared to a reference class.

        Args:
            model_outputs: Dictionary mapping model IDs to their classification outputs
            reference_class: Optional reference class to compare against

        Returns:
            Dictionary mapping model IDs to their Shapley values
        """
        model_ids = list(model_outputs.keys())

        def classification_evaluation(model_subset: Set[str]) -> float:
            if not model_subset:
                return 0.0

            subset_classes = [model_outputs[model_id] for model_id in model_subset]

            if reference_class is not None:
                # Measure accuracy against reference
                correct_predictions = sum(1 for cls in subset_classes if cls == reference_class)
                return correct_predictions / len(subset_classes)

            # Without reference, measure agreement (majority voting)
            if len(subset_classes) == 1:
                return 0.5  # Neutral score for single model

            # Count occurrences of each class
            class_counts = {}
            for cls in subset_classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1

            # Find the most common class and its count
            most_common_class = max(class_counts.items(), key=lambda x: x[1])
            most_common_count = most_common_class[1]

            # Agreement score is the proportion of models that agree with the majority
            return most_common_count / len(subset_classes)

        return self.calculate_shapley_values(model_ids, classification_evaluation)

    def distribute_rewards(self, shapley_values: Dict[str, float],
                           total_reward: float = 1.0) -> Dict[str, float]:
        """
        Distribute a total reward amount based on Shapley values.

        Args:
            shapley_values: Dictionary of Shapley values
            total_reward: Total reward to distribute

        Returns:
            Dictionary mapping model IDs to their rewards
        """
        # Reuse the normalize_rewards method from the base calculator
        calculator = self.base_calculator or ShapleyValueCalculator(lambda x: 0)
        return calculator.normalize_rewards(shapley_values, total_reward)

    def get_relative_contributions(self, shapley_values: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate relative percentage contributions from Shapley values.

        Args:
            shapley_values: Dictionary of Shapley values

        Returns:
            Dictionary mapping model IDs to their relative contribution percentages
        """
        total = sum(max(0, value) for value in shapley_values.values())

        if total == 0:
            # Equal distribution if all values are 0 or negative
            return {model_id: 1.0 / len(shapley_values) * 100 for model_id in shapley_values}

        return {model_id: max(0, value) / total * 100
                for model_id, value in shapley_values.items()}

    def get_performance_ranking(self, shapley_values: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Create a ranked list of models based on their Shapley values.

        Args:
            shapley_values: Dictionary of Shapley values

        Returns:
            List of (model_id, value) tuples sorted by descending value
        """
        return sorted(shapley_values.items(), key=lambda x: x[1], reverse=True)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two text strings.

        This is a basic implementation using character-level Jaccard similarity.
        In practice, you would use more sophisticated NLP methods.

        Args:
            text1, text2: Text strings to compare

        Returns:
            Similarity score between 0 and 1
        """
        # For a real implementation, consider using:
        # - BLEU, ROUGE, or BERTScore for NLG evaluation
        # - Embedding similarity with sentence transformers
        # - Semantic similarity using LLMs

        # Simple character-level Jaccard similarity for demonstration
        set1, set2 = set(text1), set(text2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0
