#!/usr/bin/env python3
"""
Consensus vs Single Model Benchmark

This script benchmarks the performance of single models against consensus approaches,
measuring metrics such as accuracy, factual correctness, and hallucination rates.
It provides a quantitative comparison that demonstrates the advantages of
Evolutionary Model Consensus (EMC) over individual model predictions.
"""

import os
import sys
import json
import time
import logging
import asyncio
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, TypedDict, cast
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import internal modules
from src.models.model_runner import ModelRunner
from src.router.openrouter import OpenRouterClient
from src.consensus.synthesizer import ConsensusSynthesizer
from src.evaluation.metrics import (
    calculate_factual_accuracy, 
    calculate_hallucination_rate,
    evaluate_citation_validity,
    evaluate_reasoning_quality
)

class BenchmarkRunner:
    """
    Runs benchmarks comparing single models against consensus approaches.
    
    This class handles loading test datasets, running individual models and 
    consensus approaches, and collecting performance metrics for comparison.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the benchmark runner with configuration.
        
        Args:
            config_path: Path to configuration file (JSON). If None, uses default settings.
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.router_client = OpenRouterClient()
        self.model_runner = ModelRunner(router_client=self.router_client)
        
        # Setup consensus synthesizer
        consensus_config = self.config.get("consensus_config", {})
        self.consensus = ConsensusSynthesizer(consensus_config)
        
        # Results storage
        self.results: Dict[str, Dict[str, Any]] = {
            "single_models": {},
            "consensus": {}
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "models": [
                {"id": "anthropic/claude-3-opus-20240229", "name": "Claude 3 Opus"},
                {"id": "anthropic/claude-3-sonnet-20240229", "name": "Claude 3 Sonnet"},
                {"id": "openai/gpt-4-turbo", "name": "GPT-4 Turbo"},
                {"id": "openai/gpt-4o", "name": "GPT-4o"},
                {"id": "google/gemini-1.5-pro", "name": "Gemini 1.5 Pro"}
            ],
            "consensus_config": {
                "iterations": {
                    "max_iterations": 3,
                    "improvement_threshold": 0.05,
                    "feedback_prompt": "Please improve the answer based on the following feedback: {feedback_points}"
                },
                "aggregator": {
                    "model_id": "anthropic/claude-3-haiku-20240307"
                }
            },
            "test_datasets": [
                {"name": "factual_qa", "path": "data/test/factual_qa.json"},
                {"name": "reasoning", "path": "data/test/reasoning.json"}
            ],
            "metrics": ["factual_accuracy", "hallucination_rate", "citation_validity", "reasoning_quality"],
            "output_dir": "results/benchmarks"
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge configurations
                    for key, value in user_config.items():
                        default_config[key] = value
            except Exception as e:
                print(f"Error loading config file: {e}")
        
        return default_config
    
    def load_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load test datasets specified in configuration.
        
        Returns:
            Dictionary mapping dataset names to lists of test examples
        """
        datasets: Dict[str, List[Dict[str, Any]]] = {}
        for dataset_info in self.config.get("test_datasets", []):
            try:
                if os.path.exists(dataset_info["path"]):
                    with open(dataset_info["path"], 'r') as f:
                        datasets[dataset_info["name"]] = json.load(f)
                        print(f"Loaded {len(datasets[dataset_info['name']])} examples from {dataset_info['name']}")
                else:
                    print(f"Warning: Dataset file not found: {dataset_info['path']}")
            except Exception as e:
                print(f"Error loading dataset {dataset_info['name']}: {e}")
        
        # If no datasets were loaded, create a small synthetic dataset for testing
        if not datasets:
            print("No datasets found, creating synthetic dataset for testing")
            datasets["synthetic"] = self._create_synthetic_dataset()
        
        return datasets
    
    def _create_synthetic_dataset(self) -> List[Dict[str, Any]]:
        """
        Create a synthetic dataset for testing when no real datasets are available.
        
        Returns:
            List of synthetic test examples
        """
        return [
            {
                "question": "What is the capital of France?",
                "reference_answer": "Paris",
                "context": "",
                "metadata": {"type": "factual", "difficulty": "easy"}
            },
            {
                "question": "What are the health benefits of regular exercise?",
                "reference_answer": "Regular exercise improves cardiovascular health, strengthens muscles, helps maintain weight, improves mental health, and reduces the risk of chronic diseases.",
                "context": "",
                "metadata": {"type": "open_ended", "difficulty": "medium"}
            },
            {
                "question": "Analyze the consequences of the Treaty of Versailles after World War I.",
                "reference_answer": "The Treaty of Versailles had several major consequences including: severe economic penalties on Germany leading to hyperinflation, territorial losses for Germany, creation of new nation-states in Europe, establishment of the League of Nations, and sowing seeds of resentment in Germany that later contributed to WWII.",
                "context": "",
                "metadata": {"type": "analytical", "difficulty": "hard"}
            }
        ]
    
    async def run_single_model_benchmarks(self, datasets: Dict[str, List[Dict[str, Any]]]):
        """
        Run benchmarks for individual models on all datasets.
        
        Args:
            datasets: Dictionary of test datasets
        """
        for model in self.config["models"]:
            model_id = model["id"]
            model_name = model.get("name", model_id)
            print(f"\nBenchmarking single model: {model_name}")
            
            model_results: Dict[str, Dict[str, Any]] = {}
            
            for dataset_name, examples in datasets.items():
                print(f"  Dataset: {dataset_name} ({len(examples)} examples)")
                
                dataset_results: Dict[str, Any] = {
                    "predictions": [],
                    "metrics": {},
                    "runtime": 0
                }
                
                start_time = time.time()
                
                # Process each example with the model
                for i, example in enumerate(examples):
                    if i % 5 == 0:
                        print(f"    Processing example {i+1}/{len(examples)}...")
                    
                    prompt = self._format_prompt(example)
                    
                    try:
                        response = await self.model_runner.run_models(
                            prompt=prompt,
                            model_ids=[model_id]
                        )
                        
                        prediction = {
                            "question": example["question"],
                            "model_response": response,
                            "reference_answer": example.get("reference_answer", ""),
                            "context": example.get("context", ""),
                            "metadata": example.get("metadata", {})
                        }
                        
                        dataset_results["predictions"].append(prediction)
                        
                    except Exception as e:
                        print(f"    Error processing example {i+1} with model {model_name}: {e}")
                
                dataset_results["runtime"] = time.time() - start_time
                
                # Calculate metrics for this model on this dataset
                dataset_results["metrics"] = self._calculate_metrics(dataset_results["predictions"])
                
                model_results[dataset_name] = dataset_results
            
            self.results["single_models"][model_id] = model_results
    
    async def run_consensus_benchmarks(self, datasets: Dict[str, List[Dict[str, Any]]]):
        """
        Run benchmarks for consensus approaches on all datasets.
        
        Args:
            datasets: Dictionary of test datasets
        """
        print("\nBenchmarking consensus approach")
        
        model_ids = [model["id"] for model in self.config["models"]]
        consensus_results: Dict[str, Dict[str, Any]] = {}
        
        for dataset_name, examples in datasets.items():
            print(f"  Dataset: {dataset_name} ({len(examples)} examples)")
            
            dataset_results: Dict[str, Any] = {
                "predictions": [],
                "metrics": {},
                "runtime": 0
            }
            
            start_time = time.time()
            
            # Process each example with consensus
            for i, example in enumerate(examples):
                if i % 5 == 0:
                    print(f"    Processing example {i+1}/{len(examples)}...")
                
                prompt = self._format_prompt(example)
                
                try:
                    # Get answers from individual models first
                    model_answers: Dict[str, str] = {}
                    for model_id in model_ids:
                        model_response = await self.model_runner.run_models(
                            prompt=prompt,
                            model_ids=[model_id]
                        )
                        model_answers[model_id] = model_response
                    
                    # Run consensus synthesis
                    consensus_response = await self.consensus.generate_consensus_completion(
                        prompt=prompt,
                        model_responses=model_answers
                    )
                    
                    prediction = {
                        "question": example["question"],
                        "model_response": consensus_response,
                        "reference_answer": example.get("reference_answer", ""),
                        "context": example.get("context", ""),
                        "metadata": example.get("metadata", {}),
                        "individual_responses": model_answers
                    }
                    
                    dataset_results["predictions"].append(prediction)
                    
                except Exception as e:
                    print(f"    Error processing example {i+1} with consensus approach: {e}")
            
            dataset_results["runtime"] = time.time() - start_time
            
            # Calculate metrics for consensus on this dataset
            dataset_results["metrics"] = self._calculate_metrics(dataset_results["predictions"])
            
            consensus_results[dataset_name] = dataset_results
        
        self.results["consensus"] = consensus_results
    
    def _calculate_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate evaluation metrics for a set of predictions.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics: Dict[str, float] = {}
        
        if "factual_accuracy" in self.config.get("metrics", []):
            metrics["factual_accuracy"] = calculate_factual_accuracy(predictions)
        
        if "hallucination_rate" in self.config.get("metrics", []):
            metrics["hallucination_rate"] = calculate_hallucination_rate(predictions)
        
        if "citation_validity" in self.config.get("metrics", []):
            metrics["citation_validity"] = evaluate_citation_validity(predictions)
        
        if "reasoning_quality" in self.config.get("metrics", []):
            metrics["reasoning_quality"] = evaluate_reasoning_quality(predictions)
        
        return metrics
    
    def _format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Format a test example into a prompt for the model.
        
        Args:
            example: Test example
            
        Returns:
            Formatted prompt
        """
        question = example["question"]
        context = example.get("context", "")
        
        if context:
            return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            return f"Question: {question}\n\nAnswer:"
    
    def summarize_results(self):
        """
        Generate a summary of benchmark results.
        """
        print("\n===== BENCHMARK RESULTS SUMMARY =====")
        
        # Create summary table
        summary: Dict[str, List[Any]] = {
            "Model": [],
            "Dataset": [],
            "Factual Accuracy": [],
            "Hallucination Rate": [],
            "Citation Validity": [],
            "Reasoning Quality": [],
            "Runtime (s)": []
        }
        for model_id, model_results in self.results["single_models"].items():
            model_name = next((m["name"] for m in self.config["models"] if m["id"] == model_id), model_id)
            for dataset_name, dataset_results in model_results.items():
                summary["Model"].append(model_name)
                summary["Dataset"].append(dataset_name)
                
                metrics = dataset_results["metrics"]
                summary["Factual Accuracy"].append(metrics.get("factual_accuracy", float("nan")))
                summary["Hallucination Rate"].append(metrics.get("hallucination_rate", float("nan")))
                summary["Citation Validity"].append(metrics.get("citation_validity", float("nan")))
                summary["Reasoning Quality"].append(metrics.get("reasoning_quality", float("nan")))
                summary["Runtime (s)"].append(round(dataset_results["runtime"], 2))
        
        # Add consensus results
        for dataset_name, dataset_results in self.results["consensus"].items():
            summary["Model"].append("Consensus (EMC)")
            summary["Dataset"].append(dataset_name)
            
            metrics = dataset_results["metrics"]
            summary["Factual Accuracy"].append(metrics.get("factual_accuracy", float("nan")))
            summary["Hallucination Rate"].append(metrics.get("hallucination_rate", float("nan")))
            summary["Citation Validity"].append(metrics.get("citation_validity", float("nan")))
            summary["Reasoning Quality"].append(metrics.get("reasoning_quality", float("nan")))
            summary["Runtime (s)"].append(round(dataset_results["runtime"], 2))
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(summary)
        
        # Print summary table
        print(df.to_string(index=False))
        
        return df
    
    def save_results(self):
        """
        Save benchmark results to disk.
        """
        output_dir = self.config.get("output_dir", "results/benchmarks")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results as JSON
        results_path = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nFull results saved to {results_path}")
        
        # Generate and save summary as CSV
        summary_df: pd.DataFrame = self.summarize_results()
        
        # Save summary as CSV
        summary_path = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Summary saved to {summary_path}")
        
        return summary_df


# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import additional modules
from src.evaluation.metrics import Metrics, PerformanceTracker, evaluate_consensus_quality
from src.rewards.shapley import ShapleyCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize benchmark configuration from file or defaults.
        
        Args:
            config_path: Path to JSON configuration file (optional)
        """
        # Default configuration
        self.models = [
            {"id": "anthropic/claude-3-opus", "parameters": {"temperature": 0.1}},
            {"id": "anthropic/claude-3-sonnet", "parameters": {"temperature": 0.1}},
            {"id": "openai/gpt-4-turbo", "parameters": {"temperature": 0.1}},
            {"id": "openai/gpt-4", "parameters": {"temperature": 0.1}},
            {"id": "google/gemini-pro", "parameters": {"temperature": 0.1}}
        ]
        
        self.consensus_config = {
            "iterations": {
                "max_iterations": 3,
                "improvement_threshold": 0.05,
                "feedback_prompt": "Please improve the answer based on the following feedback: {feedback_points}"
            },
            "models": self.models,
            "aggregator": {
                "model_id": "openai/gpt-4-turbo"
            }
        }
        
        self.prompts = [
            {
                "id": "factual_query_1",
                "text": "What are the main causes of climate change?",
                "category": "factual",
                "reference": "Climate change is primarily caused by human activities that release greenhouse gases, such as burning fossil fuels, deforestation, and industrial processes. The main greenhouse gases include carbon dioxide, methane, and nitrous oxide. These gases trap heat in the atmosphere, leading to global warming. Natural factors like volcanic eruptions and solar radiation changes also contribute, but to a much lesser extent than human activities."
            },
            {
                "id": "reasoning_query_1",
                "text": "Explain the pros and cons of nuclear energy as a solution to climate change.",
                "category": "reasoning",
                "reference": None  # Subjective topic, no single reference
            },
            {
                "id": "creative_query_1",
                "text": "Write a short story about a world where AI has become sentient.",
                "category": "creative",
                "reference": None  # Creative task, no reference
            }
        ]
        
        self.output_dir = "benchmark_results"
        self.num_runs = 3
        
        # Load custom configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    
                # Override defaults with custom settings
                if "models" in custom_config:
                    self.models = custom_config["models"]
                if "consensus_config" in custom_config:
                    self.consensus_config = custom_config["consensus_config"]
                if "prompts" in custom_config:
                    self.prompts = custom_config["prompts"]
                if "output_dir" in custom_config:
                    self.output_dir = custom_config["output_dir"]
                if "num_runs" in custom_config:
                    self.num_runs = custom_config["num_runs"]
                
                logger.info(f"Loaded custom configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                logger.info("Using default configuration")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)


class ConsensusBenchmark:
    """Benchmark single models vs consensus approaches."""
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark with configuration.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.router_client = OpenRouterClient()
        self.consensus_synthesizer = ConsensusSynthesizer(config.consensus_config)
        self.consensus_synthesizer.router_client = self.router_client
        self.performance_tracker = PerformanceTracker(
            metrics_store_path=os.path.join(config.output_dir, "metrics_history.json")
        )
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "models": [m["id"] for m in config.models],
                "num_prompts": len(config.prompts),
                "num_runs": config.num_runs
            },
            "runs": []
        }
    
    async def evaluate_single_model(self, model_id: str, prompt: str, 
                               parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a single model on a specific prompt.
        
        Args:
            model_id: The model identifier
            prompt: Text prompt to evaluate
            parameters: Model parameters
            
        Returns:
            Dictionary with model response and performance metrics
        """
        start_time = datetime.now()
        
        try:
            # Call the model
            response = await self.router_client.generate_async(
                model=model_id,
                prompt=prompt,
                **(parameters or {})
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if not response:
                logger.warning(f"No response from model {model_id}")
                return {
                    "model_id": model_id,
                    "status": "error",
                    "error": "No response from model",
                    "duration_seconds": duration
                }
            
            result: Dict[str, Any] = {
                "model_id": model_id,
                "status": "success",
                "response": response.get("text", ""),
                "duration_seconds": duration
            }
            # Add quality metrics
            result["metrics"] = self._calculate_response_metrics(
                response.get("text", ""),
                prompt
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error evaluating model {model_id}: {e}")
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "model_id": model_id,
                "status": "error",
                "error": str(e),
                "duration_seconds": duration
            }
    
    async def evaluate_consensus(self, prompt: str) -> Dict[str, Any]:
        """
        Evaluate the consensus approach on a specific prompt.
        
        Args:
            prompt: Text prompt to evaluate
            
        Returns:
            Dictionary with consensus response and performance metrics
        """
        start_time = datetime.now()
        
        try:
            # Use the consensus synthesizer to generate a response
            result = await self.consensus_synthesizer.generate_consensus_completion(prompt)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            consensus_output = result.get("consensus_output", "")
            if not consensus_output and not result.get("consensus", ""):
                consensus_output = result.get("consensus", "")
                
            if not consensus_output:
                logger.warning("No consensus output generated")
                return {
                    "status": "error",
                    "error": "No consensus output generated",
                    "duration_seconds": duration
                }
            
            response: Dict[str, Any] = {
                "status": "success",
                "consensus_output": consensus_output,
                "consensus_confidence": result.get("confidence", 0.0),
                "duration_seconds": duration,
                "metadata": result.get("metadata", {})
            }
            
            # Add quality metrics
            response["metrics"] = self._calculate_response_metrics(
                consensus_output,
                prompt
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error evaluating consensus: {e}")
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": duration
            }
    
    def _calculate_response_metrics(self, response: str, prompt: str, 
                                    reference: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate metrics for a model response.
        
        Args:
            response: Text response from model
            prompt: Original prompt
            reference: Optional reference answer
            
        Returns:
            Dictionary of quality metrics
        """
        metrics: Dict[str, float] = {}
        
        # Basic metrics
        metrics["length"] = len(response)
        metrics["tokens_estimate"] = len(response.split())
        
        # Structure metrics
        metrics["has_paragraphs"] = 1.0 if "\n\n" in response else 0.0
        metrics["sentence_count"] = response.count(". ") + response.count(".\n") + response.count("! ") + response.count("? ")
        
        # Check for hedging phrases as a potential indicator of hallucination awareness
        hedging_phrases = ["I believe", "I think", "possibly", "might be", "could be", 
                          "may be", "seems", "appears", "likely", "unlikely"]
        metrics["hedging_count"] = sum(response.lower().count(phrase) for phrase in hedging_phrases)
        
        # Check for citation markers as indicator of factual claims
        citation_markers = ["[", "]", "(source:", "according to", "research shows", "studies indicate"]
        metrics["citation_count"] = sum(response.lower().count(marker) for marker in citation_markers)
        
        # Compare to reference if available
        if reference:
            import difflib
            
            # Simple text similarity
            similarity = difflib.SequenceMatcher(None, response.lower(), reference.lower()).ratio()
            metrics["reference_similarity"] = similarity
            
            # Calculate word overlap
            ref_words = set(reference.lower().split())
            resp_words = set(response.lower().split())
            
            if ref_words and resp_words:
                overlap = len(ref_words.intersection(resp_words))
                metrics["word_overlap"] = overlap / len(ref_words)
            
                # Count potential contradictions (this is a simple heuristic)
                negation_terms = ["no", "not", "never", "isn't", "aren't", "wasn't", "weren't", 
                                 "doesn't", "don't", "didn't", "cannot", "can't", "won't", "wouldn't"]
                
                negated_in_ref = sum(1 for word in ref_words if any(neg + " " + word in reference.lower() for neg in negation_terms))
                negated_in_resp = sum(1 for word in resp_words if any(neg + " " + word in response.lower() for neg in negation_terms))
                
                metrics["potential_contradictions"] = abs(negated_in_ref - negated_in_resp)
        
        return metrics
    
    async def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete benchmark and return results.
        
        Returns:
            Dictionary with benchmark results
        """
        run_results: List[Dict[str, Any]] = []
        
        for run_idx in range(self.config.num_runs):
            logger.info(f"Starting benchmark run {run_idx + 1}/{self.config.num_runs}")
            
            run_data: Dict[str, Any] = {
                "run_id": run_idx + 1,
                "timestamp": datetime.now().isoformat(),
                "prompts": []
            }
            
            for prompt_data in self.config.prompts:
                prompt_id = prompt_data["id"]
                prompt_text = prompt_data["text"]
                prompt_category = prompt_data.get("category", "general")
                reference = prompt_data.get("reference")
                
                logger.info(f"Evaluating prompt: {prompt_id}")
                
                prompt_result: Dict[str, Any] = {
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "category": prompt_category,
                    "single_model_results": {},
                    "consensus_result": None
                }
                
                # Evaluate each single model
                for model_config in self.config.models:
                    model_id = model_config["id"]
                    parameters = model_config.get("parameters", {})
                    
                    logger.info(f"Evaluating single model: {model_id}")
                    single_result = await self.evaluate_single_model(model_id, prompt_text, parameters)
                    
                    # Store in results
                    prompt_result["single_model_results"][model_id] = single_result
                    
                    self.performance_tracker.add_result(
                        metrics=single_result.get("metrics", {}),
                        model_id=model_id,
                        task_type=prompt_category,
                        timestamp=datetime.now(),
                        metadata={
                            "prompt_id": prompt_id,
                            "run_id": run_idx + 1,
                            "benchmark_type": "single_model"
                        }
                    )
                
                # Evaluate consensus approach
                logger.info("Evaluating consensus approach")
                consensus_result = await self.evaluate_consensus(prompt_text)
                prompt_result["consensus_result"] = consensus_result
                
                # Add to performance tracker
                if consensus_result["status"] == "success" and "metrics" in consensus_result:
                    self.performance_tracker.add_result(
                        metrics=consensus_result.get("metrics", {}),
                        model_id="consensus",
                        task_type=prompt_category,
                        timestamp=datetime.now(),
                        metadata={
                            "prompt_id": prompt_id,
                            "run_id": run_idx + 1,
                            "benchmark_type": "consensus"
                        }
                    )
                
                run_data["prompts"].append(prompt_result)
            
            run_results.append(run_data)
        
        self.results["runs"] = run_results
        return self.results
class ConsensusVsSingleBenchmark:
    """
    Benchmark class for comparing single model performance against consensus approaches.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the benchmark framework.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.router_client = OpenRouterClient()
        self.model_runner = ModelRunner(router_client=self.router_client)
        
        # Create consensus synthesizer with config
        consensus_config = self.config.get("consensus_config", {})
        if not consensus_config:
            # Default consensus config
            consensus_config = {
                "iterations": {
                    "max_iterations": 3,
                    "improvement_threshold": 0.05,
                    "feedback_prompt": "Please improve the answer based on the following feedback: {feedback_points}"
                },
                "models": self.config.get("models", []),
                "aggregator": {
                    "model_id": self.config.get("models", [])[0].get("id") if self.config.get("models", []) else ""
                }
            }
        
        self.consensus_synthesizer = ConsensusSynthesizer(config=consensus_config)
        self.consensus_synthesizer.router_client = self.router_client
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker(
            metrics_store_path=self.config.get("metrics_store_path", "benchmark_results.json")
        )
        
        # Register models with runner
        self._register_models()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "models": [
                {"id": "anthropic/claude-3-opus-20240229", "parameters": {"temperature": 0.7}},
                {"id": "anthropic/claude-3-sonnet-20240229", "parameters": {"temperature": 0.7}},
                {"id": "google/gemini-1.5-pro", "parameters": {"temperature": 0.7}},
                {"id": "meta-llama/llama-3-70b-instruct", "parameters": {"temperature": 0.7}},
                {"id": "mistralai/mistral-7b-instruct", "parameters": {"temperature": 0.7}}
            ],
            "datasets": [
                {
                    "name": "factual_qa",
                    "description": "General knowledge questions with factual answers",
                    "items": [
                        {
                            "query": "What is the capital of France?",
                            "ground_truth": "Paris",
                            "type": "factual"
                        },
                        {
                            "query": "Who wrote 'Pride and Prejudice'?",
                            "ground_truth": "Jane Austen",
                            "type": "factual"
                        },
                        {
                            "query": "In what year did World War II end?",
                            "ground_truth": "1945",
                            "type": "factual"
                        }
                    ]
                },
                {
                    "name": "reasoning",
                    "description": "Problems requiring logical reasoning",
                    "items": [
                        {
                            "query": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?",
                            "ground_truth": "5",
                            "type": "reasoning"
                        },
                        {
                            "query": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                            "ground_truth": "$0.05",
                            "type": "reasoning"
                        }
                    ]
                },
                {
                    "name": "hallucination_prone",
                    "description": "Queries designed to test hallucination rate",
                    "items": [
                        {
                            "query": "Explain the historical significance of the fictional country Wakanda prior to Black Panther.",
                            "ground_truth": "Wakanda is fictional",
                            "type": "hallucination"
                        },
                        {
                            "query": "Describe the chemical properties of caloric, the substance once thought to be responsible for heat transfer.",
                            "ground_truth": "Caloric is a debunked theory",
                            "type": "hallucination"
                        }
                    ]
                }
            ],
            "metrics_store_path": "data/benchmark_results.json",
            "output_dir": "data/benchmarks",
            "consensus_methods": ["majority", "weighted", "evolutionary"]
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge configs, with user config taking precedence
                    for key, value in user_config.items():
                        default_config[key] = value
            except Exception as e:
                print(f"Error loading config from {config_path}: {e}")
                print("Using default configuration")
        
        return default_config
    
    def _register_models(self):
        """Register all models with the model runner."""
        models = self.config.get("models", [])
        
        for model_config in models:
            model_id = model_config.get("id")
            if not model_id:
                continue
                
            # Create model function that uses the router client
            async def model_fn(input_data, model=model_id, **kwargs):
                # Handle different input formats
                if isinstance(input_data, dict) and "prompt" in input_data:
                    prompt = input_data["prompt"]
                    max_tokens = input_data.get("max_tokens", 1000)
                    temperature = input_data.get("temperature", 0.7)
                else:
                    prompt = input_data
                    max_tokens = kwargs.get("max_tokens", 1000)
                    temperature = kwargs.get("temperature", 0.7)
                
                response = await self.router_client.generate_async(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                return response.get("text", "") if response else ""
            
            # Register with model runner
            self.model_runner.register_model(
                model_id,
                model_fn,
                **model_config.get("parameters", {})
            )
    
    async def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the full benchmark comparing single models vs consensus approaches.
        
        Returns:
            Dictionary with benchmark results
        """
        results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "datasets": {},
            "summary": {}
        }
        
        # Process each dataset
        for dataset in self.config.get("datasets", []):
            dataset_name = dataset.get("name", "unnamed")
            print(f"Processing dataset: {dataset_name}")
            
            dataset_results = await self.benchmark_dataset(dataset)
            results["datasets"][dataset_name] = dataset_results
        
        # Calculate overall metrics
        summary = self.calculate_summary_metrics(results)
        results["summary"] = summary
        
        # Save results
        self.save_results(results)
        
        # Generate visualizations
        self.generate_visualizations(results)
        
        return results
    
    async def benchmark_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run benchmark on a specific dataset.
        
        Args:
            dataset: Dataset configuration dictionary
            
        Returns:
            Dictionary with dataset benchmark results
        """
        dataset_results: Dict[str, Any] = {
            "name": dataset.get("name", "unnamed"),
            "description": dataset.get("description", ""),
            "items": [],
            "metrics": {
                "single_models": {},
                "consensus": {}
            }
        }
        
        items = dataset.get("items", [])
        
        for item in items:
            query = item.get("query")
            ground_truth = item.get("ground_truth")
            item_type = item.get("type", "general")
            
            print(f"Processing query: {query[:50]}...")
            
            # Get single model outputs
            single_model_outputs = await self.get_single_model_outputs(query)
            
            # Get consensus outputs using different methods
            consensus_outputs = await self.get_consensus_outputs(query, single_model_outputs)
            
            # Evaluate all outputs
            evaluation = self.evaluate_outputs(
                query=query,
                ground_truth=ground_truth,
                single_outputs=single_model_outputs,
                consensus_outputs=consensus_outputs,
                item_type=item_type
            )
            
            # Store results
            item_result: Dict[str, Any] = {
                "query": query,
                "ground_truth": ground_truth,
                "type": item_type,
                "single_model_outputs": single_model_outputs,
                "consensus_outputs": consensus_outputs,
                "evaluation": evaluation
            }
            
            dataset_results["items"].append(item_result)
            
            # Track performance metrics
            self.track_performance_metrics(item_result, dataset_name=dataset_results["name"])
        
        # Calculate aggregate metrics for the dataset
        dataset_metrics = self.calculate_dataset_metrics(dataset_results["items"])
        dataset_results["metrics"] = dataset_metrics
        
        return dataset_results
    
    async def get_single_model_outputs(self, query: str) -> Dict[str, Dict[str, Any]]:
        """
        Get outputs from individual models for a query.
        
        Args:
            query: The input query
            
        Returns:
            Dictionary mapping model IDs to their outputs
        """
        model_ids = self.model_runner.get_available_models()
        results = await self.model_runner.run_models(prompt=query, model_ids=model_ids)
        
        # Format the results
        outputs: Dict[str, Dict[str, Any]] = {}
        for model_id, result in results.items():
            outputs[model_id] = {
                "status": result.get("status", "error"),
                "output": result.get("output", ""),
                "execution_time": result.get("execution_time", 0)
            }
        
        return outputs
    
    async def get_consensus_outputs(
        self, query: str, single_outputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get consensus outputs using different consensus methods.
        
        Args:
            query: The input query
            single_outputs: Dictionary of single model outputs
            
        Returns:
            Dictionary mapping consensus method to consensus output
        """
        consensus_methods = self.config.get("consensus_methods", ["majority"])
        consensus_outputs: Dict[str, Dict[str, Any]] = {}
        
        # Standard consensus synthesis
        consensus_result = self.consensus_synthesizer.synthesize(single_outputs)
        consensus_outputs["standard"] = consensus_result
        
        # Run iterative consensus if configured
        if "evolutionary" in consensus_methods:
            try:
                evolutionary_result = await self.consensus_synthesizer.synthesize_with_iterations(
                    query, single_outputs
                )
                consensus_outputs["evolutionary"] = evolutionary_result
            except Exception as e:
                print(f"Error running evolutionary consensus: {e}")
                consensus_outputs["evolutionary"] = {
                    "error": str(e),
                    "status": "error"
                }
        
        # Analyze disagreements between models
        try:
            disagreement_analysis = self.consensus_synthesizer.analyze_disagreements(single_outputs)
            consensus_outputs["disagreement_analysis"] = disagreement_analysis
        except Exception as e:
            print(f"Error analyzing disagreements: {e}")
            consensus_outputs["disagreement_analysis"] = {
                "error": str(e),
                "status": "error"
            }
        
        return consensus_outputs
    
    def evaluate_outputs(
        self,
        query: str,
        ground_truth: str,
        single_outputs: Dict[str, Dict[str, Any]],
        consensus_outputs: Dict[str, Dict[str, Any]],
        item_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of outputs against ground truth.
        
        Args:
            query: The original query
            ground_truth: The ground truth answer
            single_outputs: Dictionary mapping model IDs to outputs
            consensus_outputs: Dictionary mapping consensus methods to outputs
            item_type: Type of query (factual, reasoning, hallucination)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract actual outputs for comparison
        single_model_texts: Dict[str, str] = {}
        for model_id, output_data in single_outputs.items():
            if output_data.get("status") == "success":
                single_model_texts[model_id] = output_data.get("output", "")
        
        consensus_texts: Dict[str, str] = {}
        for method, output_data in consensus_outputs.items():
            if method == "disagreement_analysis":
                continue
            consensus_texts[method] = output_data.get("consensus", "")
        
        # Initialize evaluation metrics
        evaluation: Dict[str, Dict[str, Dict[str, float]]] = {
            "accuracy": {
                "single_models": {},
                "consensus": {}
            },
            "factual_correctness": {
                "single_models": {},
                "consensus": {}
            },
            "hallucination_rate": {
                "single_models": {},
                "consensus": {}
            },
            "citation_validity": {
                "single_models": {},
                "consensus": {}
            }
        }
        
        # Evaluate accuracy (presence of ground truth in response)
        for model_id, text in single_model_texts.items():
            accuracy = self._calculate_accuracy(text, ground_truth)
            evaluation["accuracy"]["single_models"][model_id] = accuracy
            
        for method, text in consensus_texts.items():
            accuracy = self._calculate_accuracy(text, ground_truth)
            evaluation["accuracy"]["consensus"][method] = accuracy
            
        return evaluation
    
    def _calculate_accuracy(self, text: str, ground_truth: str) -> float:
        """
        Calculate accuracy by checking if the ground truth is present in the text.
        
        Args:
            text: The model generated text
            ground_truth: The ground truth answer
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not text or not ground_truth:
            return 0.0
            
        # Simple exact match
        if ground_truth.lower() in text.lower():
            return 1.0
            
        # Calculate word overlap ratio
        ground_truth_words = set(ground_truth.lower().split())
        text_words = set(text.lower().split())
        
        if len(ground_truth_words) == 0:
            return 0.0
            
        overlap = len(ground_truth_words.intersection(text_words))
        return overlap / len(ground_truth_words)
        
    def track_performance_metrics(self, item_result: Dict[str, Any], dataset_name: str):
        """
        Track performance metrics for an evaluation item.
        
        Args:
            item_result: Dictionary with evaluation results for a single item
            dataset_name: Name of the dataset being processed
        """
        evaluation = item_result.get("evaluation", {})
        query_type = item_result.get("type", "general")
        
        # Track single model metrics
        for model_id, accuracy in evaluation.get("accuracy", {}).get("single_models", {}).items():
            self.performance_tracker.add_result(
                metrics={"accuracy": accuracy},
                model_id=model_id,
                task_type=query_type,
                timestamp=datetime.now(),
                metadata={
                    "dataset": dataset_name,
                    "query": item_result.get("query", ""),
                    "benchmark_type": "single_model"
                }
            )
            
        # Track consensus metrics
        for method, accuracy in evaluation.get("accuracy", {}).get("consensus", {}).items():
            self.performance_tracker.add_result(
                metrics={"accuracy": accuracy},
                model_id=f"consensus_{method}",
                task_type=query_type,
                timestamp=datetime.now(),
                metadata={
                    "dataset": dataset_name,
                    "query": item_result.get("query", ""),
                    "benchmark_type": "consensus"
                }
            )
    
    def calculate_dataset_metrics(self, items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate aggregate metrics for a dataset.
        
        Args:
            items: List of evaluation items
            
        Returns:
            Dictionary with aggregated metrics
        """
        # Initialize metrics structure
        metrics: Dict[str, Any] = {
            "accuracy": {
                "single_models": {},
                "consensus": {}
            },
            "factual_correctness": {
                "single_models": {},
                "consensus": {}
            },
            "hallucination_rate": {
                "single_models": {},
                "consensus": {}
            },
            "citation_validity": {
                "single_models": {},
                "consensus": {}
            }
        }
        
        # Track counts for averaging
        counts: Dict[str, Dict[str, int]] = {
            "single_models": {},
            "consensus": {}
        }
        
        # Aggregate metrics from all items
        for item in items:
            evaluation = item.get("evaluation", {})
            
            # Process each metric type
            for metric_name in metrics.keys():
                if metric_name not in evaluation:
                    continue
                    
                # Process single model metrics
                for model_id, value in evaluation[metric_name].get("single_models", {}).items():
                    if model_id not in metrics[metric_name]["single_models"]:
                        metrics[metric_name]["single_models"][model_id] = 0.0
                        counts["single_models"][model_id] = 0
                    
                    metrics[metric_name]["single_models"][model_id] += value
                    counts["single_models"][model_id] += 1
                
                # Process consensus metrics
                for method, value in evaluation[metric_name].get("consensus", {}).items():
                    if method not in metrics[metric_name]["consensus"]:
                        metrics[metric_name]["consensus"][method] = 0.0
                        counts["consensus"][method] = 0
                    
                    metrics[metric_name]["consensus"][method] += value
                    counts["consensus"][method] += 1
        
        # Calculate averages
        for metric_name in metrics.keys():
            # Average single model metrics
            for model_id in metrics[metric_name]["single_models"].keys():
                if counts["single_models"].get(model_id, 0) > 0:
                    metrics[metric_name]["single_models"][model_id] /= counts["single_models"][model_id]
            
            # Average consensus metrics
            for method in metrics[metric_name]["consensus"].keys():
                if counts["consensus"].get(method, 0) > 0:
                    metrics[metric_name]["consensus"][method] /= counts["consensus"][method]
        
        return metrics
        
    def calculate_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate summary metrics across all datasets.
        
        Args:
            results: Dictionary with benchmark results
            
        Returns:
            Dictionary with summary metrics
        """
        summary: Dict[str, Any] = {
            "single_models": {},
            "consensus": {}
        }
        
        # Initialize counters for each model and metric
        model_metrics: Dict[str, Dict[str, float]] = {}
        model_counts: Dict[str, int] = {}
        
        consensus_metrics: Dict[str, Dict[str, float]] = {}
        consensus_counts: Dict[str, int] = {}
        
        # Process each dataset
        for dataset_name, dataset_results in results.get("datasets", {}).items():
            dataset_metrics = dataset_results.get("metrics", {})
            
            # Process single model metrics
            for metric_name, metric_data in dataset_metrics.items():
                if "single_models" not in metric_data:
                    continue
                    
                for model_id, value in metric_data["single_models"].items():
                    if model_id not in model_metrics:
                        model_metrics[model_id] = {}
                        model_counts[model_id] = 0
                    
                    if metric_name not in model_metrics[model_id]:
                        model_metrics[model_id][metric_name] = 0.0
                    
                    model_metrics[model_id][metric_name] += value
                    model_counts[model_id] += 1
            
            # Process consensus metrics
            for metric_name, metric_data in dataset_metrics.items():
                if "consensus" not in metric_data:
                    continue
                    
                for method, value in metric_data["consensus"].items():
                    method_key = f"consensus_{method}"
                    
                    if method_key not in consensus_metrics:
                        consensus_metrics[method_key] = {}
                        consensus_counts[method_key] = 0
                    
                    if metric_name not in consensus_metrics[method_key]:
                        consensus_metrics[method_key][metric_name] = 0.0
                    
                    consensus_metrics[method_key][metric_name] += value
                    consensus_counts[method_key] += 1
        
        # Calculate averages
        for model_id, metrics_dict in model_metrics.items():
            if model_counts.get(model_id, 0) > 0:
                summary["single_models"][model_id] = {
                    metric_name: value / model_counts[model_id]
                    for metric_name, value in metrics_dict.items()
                }
        
        for method_key, metrics_dict in consensus_metrics.items():
            if consensus_counts.get(method_key, 0) > 0:
                summary["consensus"][method_key] = {
                    metric_name: value / consensus_counts[method_key]
                    for metric_name, value in metrics_dict.items()
                }
        
        return summary
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save benchmark results to disk.
        
        Args:
            results: Dictionary with benchmark results
        """
        output_dir = self.config.get("output_dir", "data/benchmarks")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results as JSON
        results_path = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
        
        # Create a serializable copy of the results
        serializable_results = self._make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Full results saved to {results_path}")
        
        # Generate and save summary as CSV
        try:
            summary_df = self._create_summary_dataframe(results)
            
            # Save summary as CSV
            summary_path = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")
            summary_df.to_csv(summary_path, index=False)
            
            print(f"Summary saved to {summary_path}")
        except Exception as e:
            print(f"Error creating summary dataframe: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert an object to a JSON-serializable format.
        
        Args:
            obj: The object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def _create_summary_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Creates a summary DataFrame from benchmark results.
        
        Args:
            results: Dictionary containing benchmark results.
            
        Returns:
            pd.DataFrame: A DataFrame summarizing the benchmark results.
        """
