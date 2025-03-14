"""
Metrics for evaluating model performance.
"""
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_consensus_quality(consensus_results: List[Dict[str, Any]],
                               model_responses: Optional[List[Dict[str, Any]]] = None,
                               previous_iterations: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
    """
    Evaluate the quality of consensus results based on multiple metrics.

    Args:
        consensus_results: List of final consensus results from multiple models
        model_responses: Optional list of individual model responses before consensus
        previous_iterations: Optional list of results from previous consensus iterations

    Returns:
        Dictionary of quality metrics including agreement rates, diversity, and consistency
    """
    metrics = {}

    # Calculate agreement rate if model_responses are provided
    if model_responses and len(model_responses) > 1:
        # Count how many models agree with the final consensus for each point
        agreements = []
        for i, consensus in enumerate(consensus_results):
            consensus_text = consensus.get('text', '')

            # Skip empty consensus results
            if not consensus_text:
                continue

            # Count agreements for this consensus item
            model_agreements = 0
            model_count = 0

            for response in model_responses:
                if i < len(response.get('responses', [])):
                    model_text = response.get('responses', [])[i].get('text', '')
                    model_count += 1

                    # Consider "agreement" if the texts have significant overlap
                    # This is a simple approach; more sophisticated text similarity could be used
                    if consensus_text and model_text:
                        # Calculate Jaccard similarity of word sets
                        consensus_words = set(consensus_text.lower().split())
                        model_words = set(model_text.lower().split())

                        if consensus_words and model_words:
                            overlap = len(consensus_words.intersection(model_words))
                            union = len(consensus_words.union(model_words))
                            similarity = overlap / union if union > 0 else 0

                            # Consider it an agreement if similarity is above threshold
                            if similarity > 0.5:
                                model_agreements += 1

            if model_count > 0:
                agreements.append(model_agreements / model_count)

        if agreements:
            metrics['agreement_rate'] = float(np.mean(agreements))
            metrics['agreement_std'] = float(np.std(agreements))

    # Calculate response diversity
    if model_responses and len(model_responses) > 1:
        diversity_scores = []

        for i in range(len(consensus_results)):
            # Collect all model responses for this item
            responses = []
            for response in model_responses:
                if i < len(response.get('responses', [])):
                    text = response.get('responses', [])[i].get('text', '')
                    if text:
                        responses.append(text)

            if len(responses) > 1:
                # Calculate pairwise dissimilarity between all responses
                pairwise_diversities = []
                for j in range(len(responses)):
                    for k in range(j + 1, len(responses)):
                        words_j = set(responses[j].lower().split())
                        words_k = set(responses[k].lower().split())

                        if words_j and words_k:
                            overlap = len(words_j.intersection(words_k))
                            union = len(words_j.union(words_k))
                            similarity = overlap / union if union > 0 else 0
                            diversity = 1.0 - similarity
                            pairwise_diversities.append(diversity)

                if pairwise_diversities:
                    diversity_scores.append(np.mean(pairwise_diversities))

        if diversity_scores:
            metrics['response_diversity'] = float(np.mean(diversity_scores))
            metrics['diversity_std'] = float(np.std(diversity_scores))

    # Calculate consistency across iterations if previous iterations are provided
    if previous_iterations and len(previous_iterations) > 0:
        consistency_scores = []

        for i, final_result in enumerate(consensus_results):
            final_text = final_result.get('text', '')
            if not final_text:
                continue

            # Track consistency with the last iteration result
            iteration_similarities = []

            for iteration in previous_iterations:
                if i < len(iteration.get('results', [])):
                    iter_text = iteration.get('results', [])[i].get('text', '')

                    if iter_text:
                        # Calculate word-based similarity
                        final_words = set(final_text.lower().split())
                        iter_words = set(iter_text.lower().split())

                        if final_words and iter_words:
                            overlap = len(final_words.intersection(iter_words))
                            union = len(final_words.union(iter_words))
                            similarity = overlap / union if union > 0 else 0
                            iteration_similarities.append(similarity)

            if iteration_similarities:
                # Calculate the trend of similarity (increasing similarity = more consistent)
                # This checks if later iterations are more similar to the final result
                consistency_trend = 0
                if len(iteration_similarities) > 1:
                    # Simple linear regression slope of similarities over iterations
                    x = np.arange(len(iteration_similarities))
                    slope = np.polyfit(x, iteration_similarities, 1)[0]
                    consistency_trend = slope

                # Also measure the final consistency with the last iteration
                final_consistency = iteration_similarities[-1] if iteration_similarities else 0
                consistency_scores.append(final_consistency)

                # Add the trend to metrics separately
                if i == 0:  # Only add iteration trends once
                    metrics['consistency_trend'] = float(consistency_trend)

        if consistency_scores:
            metrics['process_consistency'] = float(np.mean(consistency_scores))

    # Additional summary metrics
    # Comprehensiveness: were all inputs addressed in the consensus?
    if consensus_results and model_responses:
        completeness_scores = []

        for i, consensus in enumerate(consensus_results):
            consensus_text = consensus.get('text', '')
            if not consensus_text:
                continue

            # Count unique concepts in all model responses
            all_model_words = set()
            for response in model_responses:
                if i < len(response.get('responses', [])):
                    model_text = response.get('responses', [])[i].get('text', '')
                    if model_text:
                        all_model_words.update(model_text.lower().split())

            # Count concepts in consensus
            consensus_words = set(consensus_text.lower().split())

            # Calculate what portion of important concepts were included
            if all_model_words:
                completeness = len(consensus_words.intersection(all_model_words)) / len(all_model_words)
                completeness_scores.append(completeness)

        if completeness_scores:
            metrics['consensus_completeness'] = float(np.mean(completeness_scores))

    return metrics


class Metrics:
    """
    Collection of metrics for evaluating model performance.
    """

    @staticmethod
    def classification_metrics(y_true: List, y_pred: List,
                               average: str = 'weighted') -> Dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            y_true: List of ground truth labels
            y_pred: List of predicted labels
            average: Method for averaging metrics in multiclass settings

        Returns:
            Dictionary of classification metrics
        """
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0))
        }

    @staticmethod
    def regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Args:
            y_true: List of ground truth values
            y_pred: List of predicted values

        Returns:
            Dictionary of regression metrics
        """
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))
        }

    @staticmethod
    def ranking_metrics(rankings_true: List[List], rankings_pred: List[List],
                        top_k: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate ranking-based metrics.

        Args:
            rankings_true: List of ground truth rankings
            rankings_pred: List of predicted rankings
            top_k: Consider only top k elements

        Returns:
            Dictionary of ranking metrics
        """
        def precision_at_k(y_true, y_pred, k):
            """Precision at k."""
            if len(y_pred) > k:
                y_pred = y_pred[:k]
            common = set(y_pred).intersection(set(y_true))
            return len(common) / min(k, len(y_pred))

        def recall_at_k(y_true, y_pred, k):
            """Recall at k."""
            if len(y_pred) > k:
                y_pred = y_pred[:k]
            common = set(y_pred).intersection(set(y_true))
            return len(common) / len(y_true) if len(y_true) > 0 else 0

        metrics = {}

        # If top_k is not specified, use the length of the longest true ranking
        if top_k is None:
            # Find the maximum length of all true rankings
            max_len = max([len(y_true) for y_true in rankings_true], default=0)
            # Set a reasonable minimum value if all rankings are empty
            top_k = max(max_len, 10)
        # Calculate metrics for each pair of rankings
        precisions = []
        recalls = []

        for y_true, y_pred in zip(rankings_true, rankings_pred):
            precisions.append(precision_at_k(y_true, y_pred, top_k))
            recalls.append(recall_at_k(y_true, y_pred, top_k))

        metrics[f"precision@{top_k}"] = float(np.mean(precisions))
        metrics[f"recall@{top_k}"] = float(np.mean(recalls))

        return metrics


class PerformanceTracker:
    """
    Tracks and manages performance metrics over time.

    This class allows for recording performance metrics from different model runs,
    computing statistics on those metrics, and retrieving historical performance data.
    """

    def __init__(self, metrics_store_path: Optional[str] = None):
        """
        Initialize the performance tracker.

        Args:
            metrics_store_path: Optional path to store metrics data. If None, metrics are only stored in memory.
        """
        self.metrics_history = []
        self.metrics_store_path = metrics_store_path

        # Create the metrics store directory if it doesn't exist
        if self.metrics_store_path:
            os.makedirs(os.path.dirname(self.metrics_store_path), exist_ok=True)

            # Load existing metrics if file exists
            if os.path.exists(self.metrics_store_path):
                try:
                    with open(self.metrics_store_path, 'r') as f:
                        self.metrics_history = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    # Initialize with empty list if file is invalid or doesn't exist
                    self.metrics_history = []

    def add_result(self, metrics: Dict[str, float], model_id: Optional[str] = None,
                   task_type: Optional[str] = None, timestamp: Optional[datetime] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new result to the performance tracking history.

        Args:
            metrics: Dictionary of metric names and values
            model_id: Identifier for the model that produced these metrics
            task_type: Type of task (e.g., 'classification', 'regression')
            timestamp: When the metrics were generated, defaults to current time
            metadata: Any additional information to store with the metrics
        """
        if timestamp is None:
            timestamp: str = datetime.now().isoformat()

        result: Dict[str, Union[str, float, Dict[str, Any]]] = {
            'metrics': metrics,
            'timestamp': timestamp,
        }

        if model_id:
            result['model_id'] = model_id

        if task_type:
            result['task_type'] = task_type

        if metadata:
            result['metadata'] = metadata

        self.metrics_history.append(result)

        # Persist to file if path is provided
        if self.metrics_store_path:
            try:
                with open(self.metrics_store_path, 'w') as f:
                    json.dump(self.metrics_history, f, indent=2)
            except (IOError, OSError) as e:
                print(f"Warning: Could not save metrics to {self.metrics_store_path}: {e}")

    def get_history(self, model_id: Optional[str] = None,
                    task_type: Optional[str] = None,
                    metric_name: Optional[str] = None,
                    start_time: Optional[str] = None,
                    end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve historical performance data with optional filtering.

        Args:
            model_id: Filter by model identifier
            task_type: Filter by task type
            metric_name: Filter to include only entries with this metric
            start_time: Filter entries after this timestamp (ISO format)
            end_time: Filter entries before this timestamp (ISO format)

        Returns:
            Filtered list of performance records
        """
        filtered_history = self.metrics_history.copy()

        if model_id:
            filtered_history = [r for r in filtered_history if r.get('model_id') == model_id]

        if task_type:
            filtered_history = [r for r in filtered_history if r.get('task_type') == task_type]

        if metric_name:
            filtered_history = [r for r in filtered_history if metric_name in r.get('metrics', {})]

        if start_time:
            filtered_history = [r for r in filtered_history if r.get('timestamp', '') >= start_time]

        if end_time:
            filtered_history = [r for r in filtered_history if r.get('timestamp', '') <= end_time]

        return filtered_history

    def compute_statistics(self, metric_name: str, model_id: Optional[str] = None,
                           task_type: Optional[str] = None) -> Dict[str, Optional[float]]:
        """
        Compute statistics for a specific metric across historical data.

        Args:
            metric_name: The name of the metric to analyze
            model_id: Optional filter by model
            task_type: Optional filter by task type

        Returns:
            Dictionary of statistics (mean, median, min, max, std)
        """
        history = self.get_history(model_id=model_id, task_type=task_type, metric_name=metric_name)

        if not history:
            return {
                'count': 0,
                'mean': None,
                'median': None,
                'min': None,
                'max': None,
                'std': None
            }

        values = [entry['metrics'][metric_name] for entry in history if metric_name in entry['metrics']]

        if not values:
            return {
                'count': 0,
                'mean': None,
                'median': None,
                'min': None,
                'max': None,
                'std': None
            }

        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'std': float(np.std(values))
        }

    def get_statistical_metrics(self, values: List[float]) -> Dict[str, Optional[float]]:
        """
        Calculate statistical metrics for a list of values.

        Args:
            values: List of numeric values to analyze

        Returns:
            Dictionary of statistical metrics (mean, median, min, max, std)
        """
        if not values:
            return {
                'mean': None,
                'median': None,
                'min': None,
                'max': None,
                'std': None
            }

        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'std': float(np.std(values))
        }

    def get_trend(self, metric_name: str, model_id: Optional[str] = None,
                  task_type: Optional[str] = None,
                  window_size: int = 5) -> List[Tuple[str, float]]:
        """
        Calculate the trend of a metric over time with optional moving average.

        Args:
            metric_name: The name of the metric to analyze
            model_id: Optional filter by model
            task_type: Optional filter by task type
            window_size: Size of the moving average window

        Returns:
            List of (timestamp, value) tuples showing the metric trend
        """
        history = self.get_history(model_id=model_id, task_type=task_type, metric_name=metric_name)

        if not history:
            return []

        # Sort by timestamp
        history.sort(key=lambda x: x.get('timestamp', ''))

        # Extract timestamps and metric values
        data_points = [
            (entry.get('timestamp', ''), entry['metrics'].get(metric_name))
            for entry in history
            if metric_name in entry.get('metrics', {})
        ]

        if window_size > 1 and len(data_points) >= window_size:
            # Apply moving average
            values = [v for _, v in data_points]
            smoothed_values = []

            for i in range(len(values)):
                if i < window_size - 1:
                    # For the first few points, use smaller window
                    window = values[:i + 1]
                else:
                    window = values[i - window_size + 1:i + 1]

                smoothed_values.append(sum(window) / len(window))

            # Return timestamps with smoothed values
            return [(data_points[i][0], smoothed_values[i]) for i in range(len(data_points))]
        else:
            # Return raw data points if window_size is 1 or not enough data
            return data_points

    def best_result(self, metric_name: str, higher_is_better: bool = True,
                    model_id: Optional[str] = None,
                    task_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Find the result with the best performance for a given metric.

        Args:
            metric_name: The metric to optimize
            higher_is_better: Whether higher values of the metric are better
            model_id: Optional filter by model
            task_type: Optional filter by task type

        Returns:
            The result entry with the best performance or None if no results
        """
        history = self.get_history(model_id=model_id, task_type=task_type, metric_name=metric_name)

        if not history:
            return None

        # Filter to only include entries with the specified metric
        valid_entries = [entry for entry in history if metric_name in entry.get('metrics', {})]

        if not valid_entries:
            return None

        if higher_is_better:
            return max(valid_entries, key=lambda x: x['metrics'][metric_name])
        else:
            return min(valid_entries, key=lambda x: x['metrics'][metric_name])

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export the metrics history to a pandas DataFrame for analysis.

        Returns:
            DataFrame containing all metrics history
        """
        if not self.metrics_history:
            return pd.DataFrame()

        # Normalize the nested metrics dictionary
        rows = []

        for entry in self.metrics_history:
            base_row = {k: v for k, v in entry.items() if k != 'metrics'}
            metrics = entry.get('metrics', {})

            for metric_name, value in metrics.items():
                row = base_row.copy()
                row['metric_name'] = metric_name
                row['value'] = value
                rows.append(row)

        return pd.DataFrame(rows)
