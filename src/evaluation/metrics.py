"""
Metrics for evaluating model performance.
"""
from typing import Dict, List, Any, Callable, Union, Optional
import numpy as np
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
        

