#!/usr/bin/env python3
"""
Benchmark script for comparing consensus vs single model approaches.

This script evaluates the performance of individual models against
consensus-based approaches using various metrics and test cases.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from dotenv import load_dotenv

from metrics import (
    calculate_accuracy,
    calculate_factual_consistency,
    calculate_hallucination_rate,
    calculate_response_consistency,
    calculate_reasoning_quality,
)

# Load environment variables from .env file
load_dotenv()

# Default models to test
DEFAULT_MODELS = [
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "claude-3-opus",
    "claude-3-sonnet",
]

# Test categories
TEST_CATEGORIES = [
    "factual_qa",
    "reasoning",
    "scientific",
    "planning",
]


def load_test_cases(category: Optional[str] = None) -> List[Dict]:
    """
    Load test cases from JSON file.
    
    Args:
        category: Optional category to filter test cases
        
    Returns:
        List of test case dictionaries
    """
    data_dir = Path(__file__).parent / "data"
    with open(data_dir / "test_cases.json", "r") as f:
        all_cases = json.load(f)
    
    if category:
        return [case for case in all_cases if case.get("category") == category]
    return all_cases


def run_single_model(
    model: str, 
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Dict:
    """
    Run inference with a single model.
    
    Args:
        model: Model identifier
        prompt: Input prompt
        temperature: Sampling temperature
        max_tokens: Maximum response length
        
    Returns:
        Dict containing model response and metadata
    """
    # In a real implementation, this would call the appropriate API
    # For now, we'll simulate responses
    print(f"Running model: {model} on prompt: {prompt[:50]}...")
    
    # Simulate API call and processing time
    time.sleep(0.5)
    
    # Placeholder for actual API integration
    response = f"This is a simulated response from {model}."
    
    return {
        "model": model,
        "prompt": prompt,
        "response": response,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    }


def run_consensus(
    models: List[str],
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    consensus_method: str = "weighted_voting",
) -> Dict:
    """
    Run consensus inference using multiple models.
    
    Args:
        models: List of model identifiers
        prompt: Input prompt
        temperature: Sampling temperature
        max_tokens: Maximum response length
        consensus_method: Method for combining model outputs
        
    Returns:
        Dict containing consensus response and metadata
    """
    # Get individual model responses
    model_responses = []
    for model in models:
        result = run_single_model(model, prompt, temperature, max_tokens)
        model_responses.append(result)
    
    # Apply consensus method
    if consensus_method == "weighted_voting":
        consensus = apply_weighted_voting(model_responses)
    elif consensus_method == "synthesis":
        consensus = apply_synthesis(model_responses)
    else:
        consensus = apply_ensemble(model_responses)
    
    return {
        "models": models,
        "prompt": prompt,
        "consensus_response": consensus,
        "individual_responses": model_responses,
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "consensus_method": consensus_method,
        }
    }


def apply_weighted_voting(responses: List[Dict]) -> str:
    """Apply weighted voting to combine model responses."""
    # This is a placeholder for actual implementation
    return "Consensus response via weighted voting."


def apply_synthesis(responses: List[Dict]) -> str:
    """Apply synthesis to combine model responses."""
    # This is a placeholder for actual implementation
    return "Consensus response via synthesis."


def apply_ensemble(responses: List[Dict]) -> str:
    """Apply ensemble method to combine model responses."""
    # This is a placeholder for actual implementation
    return "Consensus response via ensemble method."


def evaluate_responses(
    single_results: Dict[str, List[Dict]],
    consensus_results: List[Dict],
    ground_truth: List[Dict],
) -> Dict[str, Dict]:
    """
    Evaluate and compare single model vs consensus results.
    
    Args:
        single_results: Dictionary of single model results by model name
        consensus_results: List of consensus results
        ground_truth: List of ground truth answers
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = {"single_models": {}, "consensus": {}}
    
    # Evaluate single model results
    for model, responses in single_results.items():
        results["single_models"][model] = {
            "accuracy": calculate_accuracy(responses, ground_truth),
            "factual_consistency": calculate_factual_consistency(responses),
            "hallucination_rate": calculate_hallucination_rate(responses, ground_truth),
            "response_consistency": calculate_response_consistency(responses),
            "reasoning_quality": calculate_reasoning_quality(responses),
        }
    
    # Evaluate consensus results
    results["consensus"] = {
        "accuracy": calculate_accuracy(consensus_results, ground_truth),
        "factual_consistency": calculate_factual_consistency(consensus_results),
        "hallucination_rate": calculate_hallucination_rate(consensus_results, ground_truth),
        "response_consistency": calculate_response_consistency(consensus_results),
        "reasoning_quality": calculate_reasoning_quality(consensus_results),
    }
    
    return results


def save_results(results: Dict, output_dir: str = "results") -> None:
    """
    Save benchmark results to files.
    
    Args:
        results: Dictionary of benchmark results
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results as JSON
    with open(f"{output_dir}/benchmark_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create summary CSV
    summary = []
    # Add single model results
    for model, metrics in results["single_models"].items():
        row = {"model": model, "type": "single"}
        row.update(metrics)
        summary.append(row)
    
    # Add consensus results
    row = {"model": "consensus", "type": "consensus"}
    row.update(results["consensus"])
    summary.append(row)
    
    # Save summary as CSV
    df = pd.DataFrame(summary)
    df.to_csv(f"{output_dir}/benchmark_summary_{timestamp}.csv", index=False)
    
    print(f"Results saved to {output_dir}/")


def main():
    """Run the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark consensus vs single model approaches"
    )
    parser.add_argument(
        "--models", 
        type=str, 
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of models to test"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=TEST_CATEGORIES,
        help="Test category to run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--consensus-method",
        type=str,
        default="weighted_voting",
        choices=["weighted_voting", "synthesis", "ensemble"],
        help="Method for consensus generation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Parse models from command line
    models = args.models.split(",")
    
    # Load test cases
    test_cases = load_test_cases(args.category)
    print(f"Loaded {len(test_cases)} test cases" + 
          (f" for category: {args.category}" if args.category else ""))
    
    # Run single model benchmarks
    single_results = {}
    for model in models:
        single_results[model] = []
        for test_case in test_cases:
            result = run_single_model(model, test_case["prompt"])
            single_results[model].append(result)
    
    # Run consensus benchmarks
    consensus_results = []
    for test_case in test_cases:
        result = run_consensus(
            models, 
            test_case["prompt"],
            consensus_method=args.consensus_method
        )
        consensus_results.append(result)
    
    # Evaluate results
    eval_results = evaluate_responses(
        single_results, 
        consensus_results,
        test_cases
    )
    
    # Save results
    save_results(eval_results, args.output_dir)
    
    # Print summary
    print("\nBenchmark Results Summary:")
    print("--------------------------")
    for model, metrics in eval_results["single_models"].items():
        print(f"{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nConsensus:")
    for metric, value in eval_results["consensus"].items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()

