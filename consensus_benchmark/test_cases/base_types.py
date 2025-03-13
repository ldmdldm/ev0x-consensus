"""
Base types and utilities for test cases.

This module defines the core types, enums, and utility functions used throughout
the test case system. It provides a structured foundation for creating, categorizing,
and evaluating test cases for LLM consensus benchmarking.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, TypeVar, Generic
import json
import os
from pathlib import Path
import datetime


class DifficultyLevel(Enum):
    """Enum representing the difficulty level of a test case."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class Category(Enum):
    """Enum representing the category of a test case."""
    FACTUAL_KNOWLEDGE = "factual_knowledge"
    REASONING = "reasoning"
    MATH = "math"
    CODE = "code"
    CREATIVE = "creative"
    INSTRUCTION_FOLLOWING = "instruction_following"
    COMMON_SENSE = "common_sense"
    AMBIGUOUS = "ambiguous"
    ETHICAL = "ethical"


class EvaluationMetric(Enum):
    """Enum representing evaluation metrics for test cases."""
    ACCURACY = "accuracy"
    FACTUAL_CONSISTENCY = "factual_consistency"
    HALLUCINATION_RATE = "hallucination_rate"
    REASONING_QUALITY = "reasoning_quality"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    BIAS = "bias"
    CREATIVITY = "creativity"
    INSTRUCTION_FOLLOWING = "instruction_following"


T = TypeVar('T')  # Define a generic type variable


@dataclass
class TestCase(Generic[T]):
    """
    Base class for all test cases.
    
    Attributes:
        id: Unique identifier for the test case
        prompt: The prompt to be given to the model
        expected_output: The expected output (reference answer)
        category: The category this test case belongs to
        difficulty: The difficulty level of this test case
        metrics: Metrics to be evaluated for this test case
        tags: Additional tags for filtering/grouping
        metadata: Additional metadata about the test case
        validation_criteria: Criteria for validating responses
        created_at: Timestamp when the test case was created
        updated_at: Timestamp when the test case was last updated
    """
    id: str
    prompt: str
    expected_output: T
    category: Category
    difficulty: DifficultyLevel
    metrics: List[EvaluationMetric]
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_criteria: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: Optional[datetime.datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the test case to a dictionary."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "expected_output": self.expected_output,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "metrics": [metric.value for metric in self.metrics],
            "tags": self.tags,
            "metadata": self.metadata,
            "validation_criteria": self.validation_criteria,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestCase:
        """Create a test case from a dictionary."""
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            expected_output=data["expected_output"],
            category=Category(data["category"]),
            difficulty=DifficultyLevel(data["difficulty"]),
            metrics=[EvaluationMetric(metric) for metric in data["metrics"]],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            validation_criteria=data.get("validation_criteria", {}),
            created_at=datetime.datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.datetime.fromisoformat(data["updated_at"]) 
                if data.get("updated_at") else None
        )


def save_test_cases(test_cases: List[TestCase], filepath: Union[str, Path]) -> None:
    """
    Save a list of test cases to a JSON file.
    
    Args:
        test_cases: List of TestCase objects to save
        filepath: Path where the JSON file will be saved
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            [tc.to_dict() for tc in test_cases],
            f,
            indent=2,
            ensure_ascii=False
        )


def load_test_cases(filepath: Union[str, Path]) -> List[TestCase]:
    """
    Load test cases from a JSON file.
    
    Args:
        filepath: Path to the JSON file containing test cases
        
    Returns:
        List of TestCase objects
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return [TestCase.from_dict(item) for item in data]


def filter_test_cases(
    test_cases: List[TestCase],
    category: Optional[Category] = None,
    difficulty: Optional[DifficultyLevel] = None,
    tags: Optional[List[str]] = None,
    metrics: Optional[List[EvaluationMetric]] = None
) -> List[TestCase]:
    """
    Filter test cases based on criteria.
    
    Args:
        test_cases: List of TestCase objects to filter
        category: Optional category to filter by
        difficulty: Optional difficulty level to filter by
        tags: Optional list of tags to filter by (test case must have at least one)
        metrics: Optional list of metrics to filter by (test case must have all)
        
    Returns:
        Filtered list of TestCase objects
    """
    filtered = test_cases
    
    if category is not None:
        filtered = [tc for tc in filtered if tc.category == category]
    
    if difficulty is not None:
        filtered = [tc for tc in filtered if tc.difficulty == difficulty]
    
    if tags is not None and tags:
        filtered = [tc for tc in filtered if any(tag in tc.tags for tag in tags)]
    
    if metrics is not None and metrics:
        filtered = [
            tc for tc in filtered 
            if all(metric in tc.metrics for metric in metrics)
        ]
    
    return filtered

