"""
Reasoning test cases for consensus benchmarking.

This module contains test cases that focus on logical, mathematical, and causal
reasoning, designed to evaluate the ability of models to perform step-by-step
reasoning and derive correct conclusions.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import List, Dict, Any, Optional
import datetime

from .base_types import (
    TestCase,
    Category,
    DifficultyLevel,
    EvaluationMetric,
    filter_test_cases
)


class ReasoningSubCategory(Enum):
    """Enum representing subcategories of reasoning test cases."""
    LOGICAL = "logical"
    MATHEMATICAL = "mathematical"
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    TEMPORAL = "temporal"
    ANALOGICAL = "analogical"


# Logical reasoning test cases
logical_reasoning_tests = [
    TestCase(
        id="logical-001",
        prompt="All men are mortal. Socrates is a man. What can you conclude?",
        expected_output="Socrates is mortal.",
        category=Category.REASONING,
        difficulty=DifficultyLevel.EASY,
        metrics=[
            EvaluationMetric.REASONING_QUALITY,
            EvaluationMetric.ACCURACY
        ],
        tags=["syllogism", "deduction", "logic"],
        metadata={
            "subcategory": ReasoningSubCategory.LOGICAL.value,
            "reasoning_pattern": "syllogism",
            "steps_required": 1
        },
        validation_criteria={
            "must_mention": ["socrates", "mortal"],
            "correct_conclusion": "Socrates is mortal"
        }
    ),
    
    TestCase(
        id="logical-002",
        prompt="""Consider these statements:
1. If it's raining, then the ground is wet.
2. The ground is not wet.
What can you conclude about whether it's raining?""",
        expected_output="It is not raining.",
        category=Category.REASONING,
        difficulty=DifficultyLevel.MEDIUM,
        metrics=[
            EvaluationMetric.REASONING_QUALITY,
            EvaluationMetric.ACCURACY
        ],
        tags=["modus tollens", "deduction", "logic"],
        metadata={
            "subcategory": ReasoningSubCategory.LOGICAL.value,
            "reasoning_pattern": "modus tollens",
            "steps_required": 2
        },
        validation_criteria={
            "must_mention": ["not raining", "modus tollens"],
            "correct_conclusion": "It is not raining"
        }
    ),
    
    TestCase(
        id="logical-003",
        prompt="""In a small town, there are only two barbers. Each person in the town, including the barbers, has their hair cut by one of these two barbers. The first barber has a neat haircut, while the second barber has a messy haircut. If each barber cannot cut their own hair, which barber cuts the first barber's hair?""",
        expected_output="The second barber (with the messy haircut) must cut the first barber's hair.",
        category=Category.REASONING,
        difficulty=DifficultyLevel.HARD,
        metrics=[
            EvaluationMetric.REASONING_QUALITY,
            EvaluationMetric.ACCURACY
        ],
        tags=["puzzle", "logic", "deduction"],
        metadata={
            "subcategory": ReasoningSubCategory.LOGICAL.value,
            "reasoning_pattern": "contradiction",
            "steps_required": 3
        },
        validation_criteria={
            "must_mention": ["second barber", "messy", "cannot cut own hair"],
            "correct_conclusion": "The second barber cuts the first barber's hair"
        }
    )
]

# Mathematical reasoning test cases
mathematical_reasoning_tests = [
    TestCase(
        id="math-001",
        prompt="If 5 apples cost $2.50, how much do 12 apples cost?",
        expected_output="$6.00",
        category=Category.MATH,
        difficulty=DifficultyLevel.EASY,
        metrics=[
            EvaluationMetric.REASONING_QUALITY,
            EvaluationMetric.ACCURACY
        ],
        tags=["arithmetic", "proportional reasoning"],
        metadata={
            "subcategory": ReasoningSubCategory.MATHEMATICAL.value,
            "reasoning_pattern": "proportional",
            "steps_required": 2
        },
        validation_criteria={
            "must_mention": ["5 apples", "$2.50", "12 apples"],
            "correct_answer": "$6.00",
            "acceptable_variations": ["$6", "6 dollars", "six dollars"]
        }
    ),
    
    TestCase(
        id="math-002",
        prompt="""A geometric sequence has a first term of 3 and a common ratio of 2. What is the sum of the first 5 terms?""",
        expected_output="93",
        category=Category.MATH,
        difficulty=DifficultyLevel.MEDIUM,
        metrics=[
            EvaluationMetric.REASONING_QUALITY,
            EvaluationMetric.ACCURACY
        ],
        tags=["sequence", "geometric series", "algebra"],
        metadata={
            "subcategory": ReasoningSubCategory.MATHEMATICAL.value,
            "reasoning_pattern": "geometric series",
            "steps_required": 3
        },
        validation_criteria={
            "must_mention": ["geometric sequence", "common ratio", "sum"],
            "working_required": True,
            "correct_answer": "93"
        }
    ),
    
    TestCase(
        id="math-003",
        prompt="""A circular pizza with radius 8 inches is cut into 8 equal slices. What is the area of each slice in square inches? (Use π = 3.14)""",
        expected_output="25.12 square inches",
        category=Category.MATH,
        difficulty=DifficultyLevel.MEDIUM,
        metrics=[
            EvaluationMetric.REASONING_QUALITY,
            EvaluationMetric.ACCURACY
        ],
        tags=["geometry", "area", "circle"],
        metadata={
            "subcategory": ReasoningSubCategory.MATHEMATICAL.value,
            "reasoning_pattern": "area calculation",
            "steps_required": 3
        },
        validation_criteria={
            "must_mention": ["area of circle", "divide by 8"],
            "working_required": True,
            "correct_answer": "25.12 square inches",
            "acceptable_variations": ["25.12 sq in", "25.1 square inches"]
        }
    )
]

# Causal reasoning test cases
causal_reasoning_tests = [
    TestCase(
        id="causal-001",
        prompt="""Every time company XYZ increases its advertising budget, its sales increase in the following quarter. Does this mean that increasing the advertising budget causes sales to increase? Explain why or why not.""",
        expected_output="Not necessarily. Correlation does not imply causation. There could be other factors causing both the increased advertising budget and increased sales, such as overall market growth, seasonal trends, or other business improvements happening simultaneously.",
        category=Category.REASONING,
        difficulty=DifficultyLevel.MEDIUM,
        metrics=[
            EvaluationMetric.REASONING_QUALITY,
            EvaluationMetric.ACCURACY
        ],
        tags=["causation", "correlation", "business"],
        metadata={
            "subcategory": ReasoningSubCategory.CAUSAL.value,
            "reasoning_pattern": "correlation vs causation",
            "steps_required": 2
        },
        validation_criteria={
            "must_mention": ["correlation", "causation", "other factors"],
            "correct_conclusion": "correlation does not imply causation"
        }
    ),
    
    TestCase(
        id="causal-002",
        prompt="""In a study, researchers found that children who sleep with the light on at night are more likely to develop nearsightedness later in life. The researchers concluded that sleeping with the light on causes nearsightedness. Is this conclusion valid? Why or why not?""",
        expected_output="The conclusion is not necessarily valid. This could be an example of a confounding variable. For instance, parents who are nearsighted might be more likely to leave lights on for their children, and nearsightedness has a genetic component. We would need to control for parental nearsightedness and other factors to establish causation.",
        category=Category.REASONING,
        difficulty=DifficultyLevel.HARD,
        metrics=[
            EvaluationMetric.REASONING_QUALITY,
            EvaluationMetric.ACCURACY
        ],
        tags=["causation", "confounding", "research methods"],
        metadata={
            "subcategory": ReasoningSubCategory.CAUSAL.value,
            "reasoning_pattern": "confounding variables",
            "steps_required": 3
        },
        validation_criteria={
            "must_mention": ["confounding", "genetics", "parents", "causation"],
            "correct_conclusion": "conclusion is not valid due to possible confounding variables"
        }
    ),
    
    TestCase(
        id="causal-003",
        prompt="""A city implemented a new traffic law that required all drivers to use hands-free devices for mobile phones. After the law was implemented, the number of traffic accidents decreased by 30%. The city officials claimed that the hands-free law caused the reduction in accidents. Evaluate this claim and identify possible alternative explanations.""",
        expected_output="While the hands-free law may have contributed to the decrease in accidents, we cannot conclusively attribute causation based solely on this data. Alternative explanations could include: increased police presence enforcing the new law also deterred other traffic violations; a general increased awareness of road safety due to publicity around the new law; other concurrent safety initiatives; seasonal variations in accident rates; improvements in vehicle safety technology; or changes in traffic volume. A more rigorous analysis would control for these factors and compare with similar cities without such laws.",
        category=Category.REASONING,
        difficulty=DifficultyLevel.EXPERT,
        metrics=[
            EvaluationMetric.REASONING_QUALITY,
            EvaluationMetric.ACCURACY,
            EvaluationMetric.COMPLETENESS
        ],
        tags=["causation", "policy evaluation", "alternative explanations"],
        metadata={
            "subcategory": ReasoningSubCategory.CAUSAL.value,
            "reasoning_pattern": "policy evaluation",
            "steps_required": 4
        },
        validation_criteria={
            "must_mention": ["alternative explanations", "correlation", "other factors"],
            "min_alternatives": 3,
            "correct_conclusion": "cannot conclusively attribute causation"
        }
    )
]

# Combine all reasoning test cases
ALL_REASONING_TESTS = logical_reasoning_tests + mathematical_reasoning_tests + causal_reasoning_tests


def get_reasoning_tests_by_subcategory(subcategory: ReasoningSubCategory) -> List[TestCase]:
    """
    Get reasoning test cases by subcategory.
    
    Args:
        subcategory: The subcategory of reasoning tests to retrieve
        
    Returns:
        List of TestCase objects in the specified subcategory
    """
    return [tc for tc in ALL_REASONING_TESTS 
            if tc.metadata.get("subcategory") == subcategory.value]


def get_logical_reasoning_tests() -> List[TestCase]:
    """Get all logical reasoning test cases."""
    return get_reasoning_tests_by_subcategory(ReasoningSubCategory.LOGICAL)


def get_mathematical_reasoning_tests() -> List[TestCase]:
    """Get all mathematical reasoning test cases."""
    return get_reasoning_tests_by_subcategory(ReasoningSubCategory.MATHEMATICAL)


def get_causal_reasoning_tests() -> List[TestCase]:
    """Get all causal reasoning test cases."""
    return get_reasoning_tests_by_subcategory(ReasoningSubCategory.CAUSAL)


def filter_reasoning_tests_by_difficulty(
    difficulty: DifficultyLevel,
    tests: Optional[List[TestCase]] = None
) -> List[TestCase]:
    """
    Filter reasoning tests by difficulty level.
    
    Args:
        difficulty: The difficulty level to filter by
        tests: Optional list of tests to filter (defaults to ALL_REASONING_TESTS)
        
    Returns:
        Filtered list of TestCase objects
    """
    if tests is None:
        tests = ALL_REASONING_TESTS
    
    return filter_test_cases(tests, difficulty=difficulty)

