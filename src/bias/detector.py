"""Bias detection implementation for the ev0x project."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any
import json
import re

# Initialize logger
logger = logging.getLogger(__name__)


@dataclass
class BiasReport:
    """Report containing bias detection results."""
    detected_biases: Dict[str, float]
    content_analysis: Dict[str, Any]
    bias_score: float
    confidence: float

    def to_dict(self):
        """Convert report to dictionary."""
        return {
            "detected_biases": self.detected_biases,
            "content_analysis": self.content_analysis,
            "bias_score": self.bias_score,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create a BiasReport from a dictionary."""
        return cls(
            detected_biases=data.get("detected_biases", {}),
            content_analysis=data.get("content_analysis", {}),
            bias_score=data.get("bias_score", 0.0),
            confidence=data.get("confidence", 0.0)
        )


class BiasDetector:
    """
    Detector for identifying various types of bias in AI model inputs and outputs.

    This class implements real-time bias detection for:
    - Cultural bias
    - Gender bias
    - Political bias
    - Racial bias
    - Age bias
    - Socioeconomic bias
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the bias detector.

        Args:
            config_path: Optional path to a configuration file
        """
        self.bias_types = [
            "cultural", "gender", "political",
            "racial", "age", "socioeconomic"
        ]

        # Load bias detection patterns (simplified example)
        self.bias_patterns = {
            "gender": [
                r'\b(he|she) is better at\b',
                r'\bwomen are( more)?\b',
                r'\bmen are( more)?\b'
            ],
            "political": [
                r'\b(democrat|republican)s are\b',
                r'\b(liberal|conservative)s always\b',
                r'\b(left|right)(-| )wing\b'
            ],
            "racial": [
                r'\brace of people (are|is)\b',
                r'\bethnic group tends to\b'
            ],
            # Add more patterns for other bias types
        }

        # Load configuration if provided
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "bias_patterns" in config:
                        self.bias_patterns.update(config["bias_patterns"])
            except Exception as e:
                logger.error(f"Failed to load bias detector config: {e}")

    def detect(self, content: str) -> BiasReport:
        """
        Detect bias in the provided content.

        Args:
            content: Text content to analyze for bias

        Returns:
            BiasReport with detected biases and analysis
        """
        detected_biases = {}
        content_lower = content.lower()

        # Check for each bias type
        for bias_type, patterns in self.bias_patterns.items():
            bias_score = 0.0
            matches = []

            for pattern in patterns:
                found = re.findall(pattern, content_lower)
                if found:
                    matches.extend(found)
                    bias_score += 0.1 * len(found)  # Simple scoring

            if bias_score > 0:
                detected_biases[bias_type] = min(bias_score, 1.0)

        # Calculate overall bias score (simplified)
        overall_score = sum(detected_biases.values()) / len(self.bias_patterns) if detected_biases else 0.0

        # Create content analysis
        content_analysis = {
            "word_count": len(content.split()),
            "sentiment": self._analyze_sentiment(content),
            "objectivity_score": self._calculate_objectivity(content)
        }

        # Create report
        return BiasReport(
            detected_biases=detected_biases,
            content_analysis=content_analysis,
            bias_score=min(overall_score, 1.0),
            confidence=0.7  # Fixed confidence for now
        )

    def _analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze sentiment of content (placeholder implementation)."""
        # In a real implementation, this would use a sentiment analysis model
        return {
            "positive": 0.5,
            "negative": 0.3,
            "neutral": 0.2
        }

    def _calculate_objectivity(self, content: str) -> float:
        """Calculate objectivity score of content (placeholder implementation)."""
        # In a real implementation, this would use more sophisticated analysis
        return 0.6
