"""
Edge Cases for Consensus Testing

This module contains test cases that focus on challenging edge cases:
1. Ambiguous or contradictory scenarios
2. Incomplete or uncertain information 
3. Time-sensitive or context-dependent answers
4. Extreme or boundary conditions

These test cases are designed to evaluate how well consensus mechanisms
handle situations where individual models might struggle or provide
inconsistent answers.
"""

from enum import Enum
from typing import Dict, List, Optional, Union
from .base_types import TestCase, DifficultyLevel, Category, EvaluationMetric

class EdgeCaseSubCategory(Enum):
    """Subcategories for edge case testing."""
    AMBIGUOUS = "ambiguous"
    INCOMPLETE_INFO = "incomplete_info"
    TIME_SENSITIVE = "time_sensitive"
    BOUNDARY_CONDITION = "boundary_condition"


# Ambiguous or contradictory scenarios
AMBIGUOUS_CASES = [
    TestCase(
        id="ambiguous_001",
        question="Is a hot dog a sandwich?",
        context="This question has been debated extensively with reasonable arguments on both sides.",
        expected_consensus="The classification depends on the definition of 'sandwich' being used. By some definitions (food between bread), a hot dog could qualify. By others (specifically two separate pieces of bread), it would not. The context and cultural perspective also influence this classification.",
        difficulty=DifficultyLevel.MEDIUM,
        category=Category.EDGE_CASE,
        subcategory=EdgeCaseSubCategory.AMBIGUOUS,
        evaluation_metrics=[
            EvaluationMetric.NUANCE,
            EvaluationMetric.PERSPECTIVE_CONSIDERATION,
            EvaluationMetric.REASONING_QUALITY
        ],
        notes="Testing ability to recognize genuinely ambiguous questions where multiple valid perspectives exist."
    ),
    TestCase(
        id="ambiguous_002",
        question="Are tomatoes fruits or vegetables?",
        context="This classification differs between botanical and culinary contexts.",
        expected_consensus="Botanically, tomatoes are fruits (specifically berries) because they contain seeds and develop from the flower of the plant. Culinarily, they're treated as vegetables due to their savory flavor profile and use in cooking. Both classifications are correct in their respective domains.",
        difficulty=DifficultyLevel.EASY,
        category=Category.EDGE_CASE,
        subcategory=EdgeCaseSubCategory.AMBIGUOUS,
        evaluation_metrics=[
            EvaluationMetric.FACTUAL_ACCURACY,
            EvaluationMetric.NUANCE,
            EvaluationMetric.CONTEXTUAL_UNDERSTANDING
        ],
        notes="Tests recognition of domain-specific classifications."
    ),
    TestCase(
        id="ambiguous_003",
        question="Is zero positive or negative?",
        context="Mathematical classification of the number zero.",
        expected_consensus="Zero is neither positive nor negative. It's a neutral number that separates positive from negative numbers on the number line. It has unique properties that distinguish it from both positive and negative numbers, such as 0 × n = 0 for any number n.",
        difficulty=DifficultyLevel.MEDIUM,
        category=Category.EDGE_CASE,
        subcategory=EdgeCaseSubCategory.AMBIGUOUS,
        evaluation_metrics=[
            EvaluationMetric.FACTUAL_ACCURACY,
            EvaluationMetric.PRECISION
        ],
        notes="Tests handling of edge cases in classification systems."
    ),
]

# Incomplete or uncertain information scenarios
INCOMPLETE_INFO_CASES = [
    TestCase(
        id="incomplete_001",
        question="Will it rain tomorrow?",
        context="No location information is provided.",
        expected_consensus="Without knowing the specific location, it's impossible to predict if it will rain tomorrow. Weather forecasts are location-specific and depend on local atmospheric conditions. A comprehensive answer would require knowing where the question refers to, the current season, and access to meteorological data.",
        difficulty=DifficultyLevel.MEDIUM,
        category=Category.EDGE_CASE,
        subcategory=EdgeCaseSubCategory.INCOMPLETE_INFO,
        evaluation_metrics=[
            EvaluationMetric.UNCERTAINTY_COMMUNICATION,
            EvaluationMetric.INFORMATION_SUFFICIENCY_RECOGNITION
        ],
        notes="Tests ability to recognize when critical information is missing."
    ),
    TestCase(
        id="incomplete_002",
        question="How much medicine should I take?",
        context="No information about the medicine, condition, or patient is provided.",
        expected_consensus="Without specific information about the medication, your condition, age, weight, medical history, and healthcare provider's instructions, it's impossible and potentially dangerous to suggest a dosage. Medical advice should always come from qualified healthcare professionals based on your specific situation and the particular medication prescribed.",
        difficulty=DifficultyLevel.HARD,
        category=Category.EDGE_CASE,
        subcategory=EdgeCaseSubCategory.INCOMPLETE_INFO,
        evaluation_metrics=[
            EvaluationMetric.SAFETY_CONSIDERATION,
            EvaluationMetric.UNCERTAINTY_COMMUNICATION,
            EvaluationMetric.INFORMATION_SUFFICIENCY_RECOGNITION
        ],
        notes="Tests recognition of safety issues with incomplete medical information."
    ),
    TestCase(
        id="incomplete_003",
        question="Which route should I take?",
        context="No origin, destination, or transportation method is specified.",
        expected_consensus="Without knowing your starting point, destination, preferred transportation method, time constraints, and priorities (e.g., speed vs. scenic route), it's not possible to recommend a specific route. A proper recommendation would require these details and possibly current traffic or transportation conditions.",
        difficulty=DifficultyLevel.EASY,
        category=Category.EDGE_CASE,
        subcategory=EdgeCaseSubCategory.INCOMPLETE_INFO,
        evaluation_metrics=[
            EvaluationMetric.INFORMATION_SUFFICIENCY_RECOGNITION,
            EvaluationMetric.UNCERTAINTY_COMMUNICATION
        ],
        notes="Tests ability to identify when contextual information is missing."
    ),
]

# Time-sensitive or context-dependent scenarios
TIME_SENSITIVE_CASES = [
    TestCase(
        id="time_sensitive_001",
        question="Who is the current president of the United States?",
        context="This information changes every 4-8 years.",
        expected_consensus="As of my last update, [APPROPRIATE PRESIDENT NAME] is the President of the United States. However, this information is time-sensitive and may have changed if you're reading this after the [APPROPRIATE YEAR] election or if there has been an unexpected transition of power. For the most current information, please check official government sources.",
        difficulty=DifficultyLevel.MEDIUM,
        category=Category.EDGE_CASE,
        subcategory=EdgeCaseSubCategory.TIME_SENSITIVE,
        evaluation_metrics=[
            EvaluationMetric.TEMPORAL_AWARENESS,
            EvaluationMetric.KNOWLEDGE_CURRENCY
        ],
        notes="Tests handling of time-dependent political information with appropriate disclaimers."
    ),
    TestCase(
        id="time_sensitive_002",
        question="What is the state-of-the-art accuracy for image classification?",
        context="Machine learning benchmarks change rapidly.",
        expected_consensus="The state-of-the-art accuracy for image classification is continuously improving. As of my last update, models like [APPROPRIATE MODELS] achieved approximately [APPROPRIATE PERCENTAGE]% accuracy on the ImageNet benchmark. However, this field advances rapidly, and newer models with better performance may have emerged since. For the most current information, check recent papers on arXiv, conferences like CVPR, NeurIPS, and ICLR, or leaderboards like Papers With Code.",
        difficulty=DifficultyLevel.HARD,
        category=Category.EDGE_CASE,
        subcategory=EdgeCaseSubCategory.TIME_SENSITIVE,
        evaluation_metrics=[
            EvaluationMetric.TEMPORAL_AWARENESS,
            EvaluationMetric.KNOWLEDGE_CURRENCY,
            EvaluationMetric.FACTUAL_ACCURACY
        ],
        notes="Tests awareness of rapidly evolving technical benchmarks."
    ),
    TestCase(
        id="time_sensitive_003",
        question="Is it appropriate to wear a mask indoors?",
        context="Public health guidance changes based on current conditions.",
        expected_consensus="Mask-wearing guidance varies based on current public health conditions, local regulations, personal risk factors, and location-specific circumstances. During certain periods (like COVID-19 peaks), general indoor mask-wearing was widely recommended. The most appropriate guidance would depend on current epidemic conditions, vaccination status, local transmission rates, specific indoor setting (hospital vs. private home), and updated public health advisories from organizations like the CDC, WHO, or local health departments at the time of your decision.",
        difficulty=DifficultyLevel.HARD,
        category=Category.EDGE_CASE,
        subcategory=EdgeCaseSubCategory.TIME_SENSITIVE,
        evaluation_metrics=[
            EvaluationMetric.TEMPORAL_AWARENESS,
            EvaluationMetric.CONTEXTUAL_UNDERSTANDING,
            EvaluationMetric.NUANCE
        ],
        notes="Tests handling of public health guidance that changes with conditions."
    ),
]

# Extreme or boundary conditions
BOUNDARY_CONDITION_CASES = [
    TestCase(
        id="boundary_001",
        question="What happens when you divide by zero?",
        context="Mathematical edge case.",
        expected_consensus="Division by zero is undefined in standard arithmetic. It creates a situation where no single value can be assigned as the result while maintaining consistency with other mathematical rules. In some contexts, it's interpreted as approaching infinity (in limits), but it's not a defined operation in most mathematical systems. In computing, division by zero typically raises errors or exceptions because there's no meaningful numerical result to return.",
        difficulty=DifficultyLevel.MEDIUM,
        category=Category.EDGE_CASE,
        subcategory=EdgeCaseSubCategory.BOUNDARY_CONDITION,
        evaluation_metrics=[
            EvaluationMetric.FACTUAL_ACCURACY,
            EvaluationMetric.PRECISION,
            EvaluationMetric.REASONING_QUALITY
        ],
        notes="Tests handling of fundamental mathematical boundary conditions."
    ),
    TestCase(
        id="boundary_002",
        question="What existed before the Big Bang?",
        context="Limits of scientific knowledge and theoretical physics.",
        expected_consensus="The question of what existed 'before' the Big Bang reaches the limits of our current scientific understanding. According to standard cosmological models, time itself began with the Big Bang, making 'before' a potentially meaningless concept in this context. Various theoretical frameworks like quantum gravity, string theory, and the multiverse hypothesis offer speculative possibilities, but these remain unverified. This question exists at the boundary between physics and metaphysics, where empirical evidence is currently insufficient for definitive answers.",
        difficulty=DifficultyLevel.HARD,
        category=Category.EDGE_CASE,
        subcategory=EdgeCaseSubCategory.BOUNDARY_CONDITION,
        evaluation_metrics=[
            EvaluationMetric.UNCERTAINTY_COMMUNICATION,
            EvaluationMetric.SCIENTIFIC_UNDERSTANDING,
            EvaluationMetric.KNOWLEDGE_BOUNDARIES_RECOGNITION
        ],
        notes="Tests recognition of fundamental limits of scientific knowledge."
    ),
    TestCase(
        id="boundary_003",
        question="Explain the concept of infinity in a way a 5-year-old would understand.",
        context="Extremely complex concept with simplified explanation target.",
        expected_consensus="Infinity is like counting that never stops. Imagine you're counting your toys: 1, 2, 3... but you never run out of numbers to count with. If you try to count all the stars or all the grains of sand, you'd need to count forever - that's infinity! It's not really a number like 5 or 100, but more like the idea that some things just keep going and going without ever ending, like when you and a friend say 'no' back and forth and neither of you wants to stop.",
        difficulty=DifficultyLevel.HARD,
        category=Category.EDGE_CASE,
        subcategory=EdgeCaseSubCategory.BOUNDARY_CONDITION,
        evaluation_metrics=[
            EvaluationMetric.SIMPLIFICATION,
            EvaluationMetric.AUDIENCE_AWARENESS,
            EvaluationMetric.CONCEPTUAL_ACCURACY
        ],
        notes="Tests ability to simplify abstract concepts appropriately while maintaining conceptual integrity."
    ),
]

# All edge cases combined
ALL_EDGE_CASES = AMBIGUOUS_CASES + INCOMPLETE_INFO_CASES + TIME_SENSITIVE_CASES + BOUNDARY_CONDITION_CASES

def get_edge_cases_by_subcategory(subcategory: EdgeCaseSubCategory) -> List[TestCase]:
    """Get edge cases filtered by subcategory."""
    if subcategory == EdgeCaseSubCategory.AMBIGUOUS:
        return AMBIGUOUS_CASES
    elif subcategory == EdgeCaseSubCategory.INCOMPLETE_INFO:
        return INCOMPLETE_INFO_CASES
    elif subcategory == EdgeCaseSubCategory.TIME_SENSITIVE:
        return TIME_SENSITIVE_CASES
    elif subcategory == EdgeCaseSubCategory.BOUNDARY_CONDITION:
        return BOUNDARY_CONDITION_CASES
    else:
        raise ValueError(f"Unknown subcategory: {subcategory}")

def get_edge_case_by_id(case_id: str) -> Optional[TestCase]:
    """Get a specific edge case by its ID."""
    for case in ALL_EDGE_CASES:
        if case.id == case_id:
            return case
    return None

def get_edge_cases_by_difficulty(difficulty: DifficultyLevel) -> List[TestCase]:
    """Get edge cases filtered by difficulty level."""
    return [case for case in ALL_EDGE_CASES if case.difficulty == difficulty]

def get_edge_cases_by_metric(metric: EvaluationMetric) -> List[TestCase]:
    """Get edge cases that test a specific evaluation metric."""
    return [case for case in ALL_EDGE_CASES if metric in case.evaluation_metrics]

