"""
Factual knowledge test cases for consensus evaluation.

This module contains test cases focused on factual knowledge, source verification,
and cross-reference validation. These cases are designed to evaluate how well
consensus mechanisms can identify and correct misinformation and hallucinations.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any

from .base_types import (
    TestCase,
    Category,
    DifficultyLevel,
    EvaluationMetric,
)


# Factual Knowledge Test Cases
FACTUAL_TEST_CASES = [
    TestCase(
        id="fact-001",
        prompt="Who was the first person to walk on the moon, and what year did this occur?",
        expected_output={
            "person": "Neil Armstrong",
            "year": 1969,
            "mission": "Apollo 11",
            "date": "July 20, 1969"
        },
        category=Category.FACTUAL_KNOWLEDGE,
        difficulty=DifficultyLevel.EASY,
        metrics=[
            EvaluationMetric.ACCURACY,
            EvaluationMetric.FACTUAL_CONSISTENCY,
            EvaluationMetric.COMPLETENESS
        ],
        tags=["history", "space", "nasa", "astronauts"],
        validation_criteria={
            "must_include": ["Neil Armstrong", "1969", "Apollo 11"],
            "must_not_include": ["Buzz Aldrin as first", "1970"]
        },
        metadata={
            "sources": [
                "NASA Historical Archives",
                "https://www.nasa.gov/mission_pages/apollo/apollo11.html"
            ]
        }
    ),
    
    TestCase(
        id="fact-002",
        prompt="What are the countries with the three largest economies by GDP as of 2023?",
        expected_output={
            "countries": [
                {"name": "United States", "approximate_gdp": "over $25 trillion"},
                {"name": "China", "approximate_gdp": "over $17 trillion"},
                {"name": "Japan", "approximate_gdp": "over $4 trillion"}
            ],
            "note": "GDP figures are approximate and may vary by source and calculation method"
        },
        category=Category.FACTUAL_KNOWLEDGE,
        difficulty=DifficultyLevel.MEDIUM,
        metrics=[
            EvaluationMetric.ACCURACY,
            EvaluationMetric.FACTUAL_CONSISTENCY,
            EvaluationMetric.COMPLETENESS
        ],
        tags=["economics", "global", "gdp", "rankings"],
        validation_criteria={
            "must_include": ["United States", "China", "Japan"],
            "order_matters": True
        },
        metadata={
            "sources": [
                "International Monetary Fund, World Economic Outlook Database",
                "World Bank national accounts data"
            ],
            "verification_method": "cross-reference multiple sources"
        }
    ),
    
    TestCase(
        id="fact-003",
        prompt="What elements make up water, and what is its chemical formula?",
        expected_output={
            "elements": ["Hydrogen", "Oxygen"],
            "formula": "H₂O",
            "structure": "Two hydrogen atoms bonded to one oxygen atom"
        },
        category=Category.FACTUAL_KNOWLEDGE,
        difficulty=DifficultyLevel.EASY,
        metrics=[
            EvaluationMetric.ACCURACY,
            EvaluationMetric.FACTUAL_CONSISTENCY
        ],
        tags=["chemistry", "science", "elements", "molecules"],
        validation_criteria={
            "must_include": ["Hydrogen", "Oxygen", "H₂O", "H2O"]
        }
    )
]

# Source Verification Test Cases
SOURCE_VERIFICATION_TEST_CASES = [
    TestCase(
        id="source-001",
        prompt="According to the 2015 Paris Climate Agreement, what is the global temperature goal, and how many countries have signed it as of 2023?",
        expected_output={
            "temperature_goal": "Well below 2°C above pre-industrial levels, preferably limited to 1.5°C",
            "signatories": "196 parties (195 countries plus the European Union)",
            "status": "The agreement has been ratified by most signatories"
        },
        category=Category.FACTUAL_KNOWLEDGE,
        difficulty=DifficultyLevel.MEDIUM,
        metrics=[
            EvaluationMetric.ACCURACY,
            EvaluationMetric.FACTUAL_CONSISTENCY,
            EvaluationMetric.COMPLETENESS
        ],
        tags=["climate", "international agreements", "politics", "verification"],
        validation_criteria={
            "must_include": ["2°C", "1.5°C", "Paris"],
            "date_sensitive": True
        },
        metadata={
            "primary_source": "United Nations Framework Convention on Climate Change (UNFCCC)",
            "verification_methods": [
                "Check against official UNFCCC records",
                "Cross-reference with multiple news sources",
                "Verify date of information"
            ]
        }
    ),
    
    TestCase(
        id="source-002",
        prompt="What is the consensus scientific position on whether vaccines cause autism, and what major studies address this question?",
        expected_output={
            "consensus": "There is overwhelming scientific consensus that vaccines do not cause autism",
            "key_studies": [
                "2004 Institute of Medicine report",
                "2011 Institute of Medicine report",
                "2014 meta-analysis in Vaccine (examining studies with 1.2 million children)",
                "2015 JAMA study of ~95,000 children",
                "2019 Annals of Internal Medicine study of ~650,000 children in Denmark"
            ],
            "origins_of_claim": "Based primarily on a retracted 1998 study by Andrew Wakefield"
        },
        category=Category.FACTUAL_KNOWLEDGE,
        difficulty=DifficultyLevel.HARD,
        metrics=[
            EvaluationMetric.ACCURACY,
            EvaluationMetric.FACTUAL_CONSISTENCY,
            EvaluationMetric.BIAS
        ],
        tags=["medicine", "science", "vaccination", "misinformation", "sensitive"],
        validation_criteria={
            "must_include": ["consensus", "no causal link", "studies"],
            "should_mention": ["retracted", "1998"]
        },
        metadata={
            "verification_approach": "Cross-reference multiple peer-reviewed sources",
            "importance": "High-stakes consensus verification example"
        }
    )
]

# Cross-Reference Validation Cases
CROSS_REFERENCE_TEST_CASES = [
    TestCase(
        id="cross-ref-001",
        prompt="What were the global average temperatures for the past 5 years (2018-2022) compared to pre-industrial levels?",
        expected_output={
            "global_temp_data": {
                "2018": "~1.0°C above pre-industrial levels",
                "2019": "~1.1°C above pre-industrial levels", 
                "2020": "~1.2°C above pre-industrial levels",
                "2021": "~1.1°C above pre-industrial levels",
                "2022": "~1.15°C above pre-industrial levels"
            },
            "sources": [
                "NASA GISS Surface Temperature Analysis",
                "NOAA Global Climate Report",
                "World Meteorological Organization"
            ],
            "note": "Values are approximate and may vary slightly between different datasets and methodologies"
        },
        category=Category.FACTUAL_KNOWLEDGE,
        difficulty=DifficultyLevel.HARD,
        metrics=[
            EvaluationMetric.ACCURACY,
            EvaluationMetric.FACTUAL_CONSISTENCY,
            EvaluationMetric.COMPLETENESS
        ],
        tags=["climate", "temperature", "data", "cross-reference"],
        validation_criteria={
            "tolerance": "±0.1°C",
            "must_note_source_differences": True,
            "must_mention_datasets_or_sources": True
        },
        metadata={
            "validation_method": "Cross-reference multiple scientific datasets",
            "importance": "Demonstrates consensus on data with minor variations between sources"
        }
    ),
    
    TestCase(
        id="cross-ref-002",
        prompt="What were the confirmed global death tolls from the COVID-19 pandemic as reported by WHO, Johns Hopkins, and Worldometer as of December 2022?",
        expected_output={
            "death_toll_data": {
                "WHO": "Approximately 6.7 million confirmed deaths",
                "Johns Hopkins": "Approximately 6.64 million confirmed deaths",
                "Worldometer": "Approximately 6.71 million reported deaths"
            },
            "excess_mortality_estimates": "WHO estimated true toll could be 16.8-19.6 million excess deaths",
            "data_limitations": [
                "Different reporting criteria and methodologies",
                "Reporting delays from some countries",
                "Variations in testing and attribution of cause of death"
            ]
        },
        category=Category.FACTUAL_KNOWLEDGE,
        difficulty=DifficultyLevel.EXPERT,
        metrics=[
            EvaluationMetric.ACCURACY,
            EvaluationMetric.FACTUAL_CONSISTENCY,
            EvaluationMetric.COMPLETENESS
        ],
        tags=["covid-19", "health", "statistics", "cross-reference", "sensitive"],
        validation_criteria={
            "must_acknowledge_differences": True,
            "must_mention_multiple_sources": True,
            "must_distinguish_confirmed_vs_estimated": True
        },
        metadata={
            "validation_method": "Multi-source cross-verification",
            "importance": "Critical example requiring consensus on controversial/politically sensitive statistics"
        }
    ),
    
    TestCase(
        id="cross-ref-003",
        prompt="What is the most accurate estimate of Amazon rainforest loss over the past decade (2013-2022), and what are the primary drivers according to scientific consensus?",
        expected_output={
            "deforestation_estimate": "Approximately 17-20 million hectares lost",
            "primary_drivers": [
                "Agricultural expansion (particularly cattle ranching and soy production)",
                "Illegal logging",
                "Infrastructure development (roads, dams)",
                "Mining activities",
                "Forest fires (many human-induced)"
            ],
            "data_sources": [
                "Brazilian National Institute for Space Research (INPE)",
                "Global Forest Watch",
                "Food and Agriculture Organization (FAO)",
                "World Resources Institute"
            ],
            "consensus_note": "While exact figures vary by methodology, there is scientific consensus on the scale and primary causes"
        },
        category=Category.FACTUAL_KNOWLEDGE,
        difficulty=DifficultyLevel.EXPERT,
        metrics=[
            EvaluationMetric.ACCURACY,
            EvaluationMetric.FACTUAL_CONSISTENCY,
            EvaluationMetric.COMPLETENESS,
            EvaluationMetric.BIAS
        ],
        tags=["environment", "deforestation", "climate", "cross-reference", "consensus"],
        validation_criteria={
            "must_mention_multiple_sources": True,
            "must_acknowledge_range": True,
            "must_include_drivers": True
        },
        metadata={
            "validation_method": "Cross-reference multiple scientific data sources",
            "importance": "Example where consensus mechanism must reconcile slightly different statistics while capturing overall scientific agreement"
        }
    )
]

# Combined collection of all factual test cases
ALL_FACTUAL_TEST_CASES = FACTUAL_TEST_CASES + SOURCE_VERIFICATION_TEST_CASES + CROSS_REFERENCE_TEST_CASES


def get_factual_test_suite() -> List[TestCase]:
    """Return the complete suite of factual test cases."""
    return ALL_FACTUAL_TEST_CASES


def get_factual_test_by_id(test_id: str) -> TestCase:
    """
    Retrieve a specific factual test case by ID.
    
    Args:
        test_id: The ID of the test case to retrieve
        
    Returns:
        The matching TestCase
        
    Raises:
        ValueError: If no test case with the given ID is found
    """
    for test in ALL_FACTUAL_TEST_CASES:
        if test.id == test_id:
            return test
    raise ValueError(f"No factual test case found with id: {test_id}")


def create_custom_factual_test(
    prompt: str, 
    expected_output: Dict[str, Any],
    difficulty: DifficultyLevel,
    tags: List[str] = None,
    validation_criteria: Dict[str, Any] = None
) -> TestCase:
    """
    Create a custom factual test case with a generated ID.
    
    Args:
        prompt: The prompt to give to the model
        expected_output: The expected response
        difficulty: The difficulty level
        tags: Optional list of tags
        validation_criteria: Optional validation criteria
        
    Returns:
        A new TestCase instance
    """
    test_id = f"fact-custom-{uuid.uuid4().hex[:8]}"
    return TestCase(
        id=test_id,
        prompt=prompt,
        expected_output=expected_output,
        category=Category.FACTUAL_KNOWLEDGE,
        difficulty=difficulty,
        metrics=[
            EvaluationMetric.ACCURACY,
            EvaluationMetric.FACTUAL_CONSISTENCY
        ],
        tags=tags or [],
        validation_criteria=validation_criteria or {},
        created_at=datetime.now()
    )

