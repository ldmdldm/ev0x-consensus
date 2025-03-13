"""
Test cases for evaluating LLM consensus performance.

This package provides a collection of test cases and utilities for benchmarking
and evaluating the performance of Large Language Models (LLMs) and consensus mechanisms.
"""

# Package metadata
__version__ = "0.1.0"
__author__ = "Ev0x Team"
__license__ = "MIT"

# Export main types and enums
from .base_types import (
    TestCase,
    Category,
    DifficultyLevel,
    EvaluationMetric,
    
    # Export utility functions
    save_test_cases,
    load_test_cases,
    filter_test_cases,
)

# Constants
PACKAGE_DIR = __path__[0]

