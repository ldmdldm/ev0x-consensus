from typing import Dict, List, Optional, Any, Union
from web3 import Web3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FlareValidator:
    """Simple validator implementation for Flare network integration."""
    
    def __init__(self, web3_provider: str):
        """Initialize the validator with Web3 connection."""
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.model_weights: Dict[str, float] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        
    def update_model_weights(self, weights: Dict[str, float]) -> None:
        """Update weights for different models."""
        self.model_weights = weights
        logger.info(f"Updated model weights: {weights}")
        
    def register_result(self, result_id: str, data: Dict[str, Any]) -> None:
        """Register a result in the validator."""
        self.results[result_id] = {
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.info(f"Registered result {result_id}")
        
    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get a registered result by ID."""
        return self.results.get(result_id)
        
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.model_weights.copy()
        
    def check_connection(self) -> bool:
        """Check if connected to Flare network."""
        try:
            return self.w3.is_connected()
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

