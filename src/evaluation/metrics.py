"""
Metrics for evaluating model performance.
"""
from typing import Dict, List, Any, Callable, Union, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
            timestamp = datetime.now().isoformat()
            
        result = {
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
                          task_type: Optional[str] = None) -> Dict[str, float]:
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
                    window = values[:i+1]
                else:
                    window = values[i-window_size+1:i+1]
                    
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
