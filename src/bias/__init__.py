"""
Bias detection and neutralization package for the ev0x project.

This package contains modules for detecting and neutralizing different types of biases
in AI model outputs, including gender, political, racial, cultural, age,
and socioeconomic biases.
"""

from src.bias.detector import BiasDetector, BiasReport
from src.bias.neutralizer import BiasNeutralizer

__all__ = ['BiasDetector', 'BiasReport', 'BiasNeutralizer']

