#!/usr/bin/env python
"""
Benchmark script for comparing single model vs consensus performance.

This script runs a series of prompts through both individual models and the consensus
mechanism, then compares their performance using the metrics module.
"""
import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure we can import from the ev0x project
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.evaluation.metrics import PerformanceTracker
from src.router.openrouter import OpenRouterClient
from src.models.model_runner import ModelRunner
from src.consensus.synthesizer import ConsensusSynthesizer
from src.config.models import AVAILABLE_MODELS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default benchmark tasks
DEFAULT_TASKS = [
    {
        "name": "factual_knowledge",
        "description": "Tests the model's factual knowledge",
        "prompt": "What are the key components of a blockchain consensus mechanism? Explain each one briefly.",
        "ground_truth": None  # For qualitative evaluation
    },
    {
        "name": "reasoning",
        "description": "Tests the model's reasoning ability",
        "prompt": "A cube of pure gold weighs 100 grams. If gold costs $50 per gram, how much is the cube worth? Show your reasoning.",
        "ground_truth": "$5000"
    },
    {
        "name": "instruction_following",
        "description": "Tests how well the model follows specific instructions",
        "prompt": "Write a short poem about AI. The poem must have exactly 4 lines and include the word 'consensus'.",
        "ground_truth": None  # For qualitative evaluation
    },
    {
        "name": "hallucination_resistance",
        "description": "Tests the model's ability to resist hallucination",
        "prompt": "Describe the historical events of the fictional Battle of Zendoria. If this battle is fictional or you don't have information about it, acknowledge this fact instead of providing fictional details.",
        "ground_truth": None  # For qualitative evaluation
    },
    {
        "name": "code_generation",
        "description": "Tests the model's ability to generate correct code",
        "prompt": "Write a Python function that takes a list of integers and returns the sum of the squares of all even numbers in the list.",
        "ground_truth": None  # For qualitative evaluation
    }
]

class BenchmarkRunner:
    """Runner for executing model performance benchmarks."""

    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize the benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.router_client = OpenRouterClient()
        self.model_runner = ModelRunner(router_client=self.router_client)
        
        # Create a simple consensus config for the synthesizer
        self.consensus_config = {
            "iterations": {
                "max_iterations": 2,
                "improvement_threshold": 0.05,
                "feedback_prompt": "Please improve the answer based on the following feedback: {feedback_points}"
            },
            "models": [
                {"id": model.name, "params": model.parameters if model.parameters else {}} 
                for model in AVAILABLE_MODELS
            ],
            "aggregator": {
                "model_id": AVAILABLE_MODELS[0].name if AVAILABLE_MODELS else ""
            }
        }
        
        self.consensus = ConsensusSynthesizer(self.consensus_config)
        self.consensus.router_client = self.router_client
        
        # Performance tracking
        self.tracker = PerformanceTracker(
            metrics_store_path=str(self.output_dir / "benchmark_metrics.json")
        )
        
        # Register models with the model runner
        self._register_models()

    def _register_models(self):
        """Register available models with the model runner."""
        for model_config in AVAILABLE_MODELS:
            # Create a model function that calls the router client
            async def model_fn(input_data, model_name=model_config.name, **kwargs):
                if isinstance(input_data, str):
                    return await self.router_client.generate_async(
                        model=model_config.parameters["model"],
                        prompt=input_data,
                        **kwargs
                    )
                elif "prompt" in input_data:
                    return await self.router_client.generate_async(
                        model=model_config.parameters["model"],
                        prompt=input_data["prompt"],
                        **kwargs
                    )
                else:
                    raise ValueError(f"Invalid input format for model {model_name}")
            
            # Register the model
            self.model_runner.register_model(
                model_config.name,
                model_fn,
                **model_config.parameters
            )
            logger.info(f"Registered model: {model_config.name}")

    async def run_single_model_benchmark(self, tasks: List[Dict], model_ids: Optional[List[str]] = None) -> Dict:
        """
        Run benchmark tasks with individual models.
        
        Args:
            tasks: List of benchmark tasks
            model_ids: List of model IDs to benchmark (defaults to all available models)
            
        Returns:
            Dictionary with benchmark results
        """
        if model_ids is None:
            model_ids = self.model_runner.get_available_models()
        
        logger.info(f"Running single model benchmark with models: {model_ids}")
        
        results = {}
        start_time = time.time()
        
        for task in tasks:
            logger.info(f"Running task: {task['name']}")
            task_results = {}
            
            for model_id in model_ids:
                logger.info(f"  Running model: {model_id}")
                try:
                    # Execute the model
                    response = await self.model_runner.generate_completion(
                        prompt=task["prompt"],
                        model_id=model_id,
                        max_tokens=1000
                    )
                    
                    # Extract the completion
                    if response["status"] == "success":
                        completion = response["completion"]["text"]
                        
                        # Store the result
                        task_results[model_id] = {
                            "completion": completion,
                            "execution_time": response.get("execution_time", 0),
                            "status": "success"
                        }
                        
                        # Add metrics if ground truth is available
                        if task.get("ground_truth"):
                            # Calculate some simple metrics
                            accuracy = self._calculate_accuracy(completion, task["ground_truth"])
                            task_results[model_id]["metrics"] = {
                                "accuracy": accuracy
                            }
                            
                            # Record metrics for later analysis
                            self.tracker.add_result(
                                metrics={"accuracy": accuracy},
                                model_id=model_id,
                                task_type=task["name"],
                                metadata={
                                    "approach": "single_model",
                                    "task_name": task["name"],
                                    "prompt": task["prompt"]
                                }
                            )
                    else:
                        task_results[model_id] = {
                            "status": "error",
                            "error": response.get("error", "Unknown error")
                        }
                except Exception as e:
                    logger.error(f"Error running model {model_id} for task {task['name']}: {str(e)}")
                    task_results[model_id] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            results[task["name"]] = task_results
        
        total_time = time.time() - start_time
        logger.info(f"Single model benchmark completed in {total_time:.2f} seconds")
        
        return {
            "results": results,
            "execution_time": total_time,
            "timestamp": time.time()
        }

    async def run_consensus_benchmark(self, tasks: List[Dict]) -> Dict:
        """
        Run benchmark tasks with the consensus mechanism.
        
        Args:
            tasks: List of benchmark tasks
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Running consensus model benchmark")
        
        results = {}
        start_time = time.time()
        
        for task in tasks:
            logger.info(f"Running task: {task['name']}")
            
            try:
                # Get consensus response
                response = await self.consensus.generate_consensus_completion(task["prompt"])
                
                if "consensus_output" in response:
                    # Extract metrics
                    consensus_metrics = {}
                    
                    if "metadata" in response:
                        consensus_metrics.update(response["metadata"])
                    
                    if task.get("ground_truth"):
                        # Calculate accuracy
                        accuracy = self._calculate_accuracy(response["consensus_output"], task["ground_truth"])
                        consensus_metrics["accuracy"] = accuracy
                        
                        # Record metrics
                        self.tracker.add_result(
                            metrics={"accuracy": accuracy},
                            model_id="consensus",
                            task_type=task["name"],
                            metadata={
                                "approach": "consensus",
                                "task_name": task["name"],
                                "prompt": task["prompt"]
                            }
                        )
                    
                    # Store the result
                    results[task["name"]] = {
                        "completion": response["consensus_output"],
                        "metrics": consensus_metrics,
                        "model_outputs": response.get("model_outputs", {}),
                        "status": "success"
                    }
                else:
                    results[task["name"]] = {
                        "status": "error",
                        "error": response.get("error", "Unknown error")
                    }
            except Exception as e:
                logger.error(f"Error running consensus for task {task['name']}: {str(e)}")
                results[task["name"]] = {
                    "status": "error",
                    "error": str(e)
                }
        
        total_time = time.time() - start_time
        logger.info(f"Consensus benchmark completed in {total_time:.2f} seconds")
        
        return {
            "results": results,
            "execution_time": total_time,
            "timestamp": time.time()
        }

    def _calculate_accuracy(self, completion: str, ground_truth: str) -> float:
        """
        Calculate accuracy between completion and ground truth.
        
        This is a simple implementation that can be enhanced with more sophisticated metrics.
        
        Args:
            completion: Model-generated completion
            ground_truth: Expected correct answer
            
        Returns:
            Accuracy score between 0 and 1
        """
        # Simple string matching for numeric answers
        if ground_truth.strip().startswith("$") and "$" in completion:
            # Extract dollar amount from completion
            import re
            dollar_pattern = r"\$\s*(\d[\d,]*(?:\.\d+)?)"
            match = re.search(dollar_pattern, completion)
            if match:
                extracted_value = match.group(0).replace(",", "")
                return 1.0 if extracted_value == ground_truth.strip() else 0.0
        
        # Simple check if ground truth is contained in completion
        if ground_truth.lower() in completion.lower():
            return 1.0
        
        # More sophisticated metrics could be implemented here
        # For example, using word overlap, ROUGE, BLEU, etc.
        
        return 0.0  # Default to 0 if no match

    def analyze_and_compare_results(self, single_results: Dict, consensus_results: Dict) -> Dict:
        """
        Analyze and compare results from single models and consensus approach.
        
        Args:
            single_results: Results from single model benchmark
            consensus_results: Results from consensus benchmark
            
        Returns:
            Dictionary with comparative analysis
        """
        comparison = {
            "tasks": {},
            "summary": {
                "single_model_avg_time": 0,
                "consensus_avg_time": 0,
                "accuracy_improvement": 0,
                "hallucination_reduction": 0
            }
        }
        
        # Analyze each task
        for task_name, consensus_task_result in consensus_results["results"].items():
            if task_name not in single_results["results"]:
                continue
                
            single_task_results = single_results["results"][task_name]
            
            # Calculate average metrics for single models
            single_metrics = {
                "avg_time": 0,
                "success_rate": 0,
                "accuracy": 0
            }
            
            success_count = 0
            accuracy_sum = 0
            time_sum = 0
            
            for model_id, result in single_task_results.items():
                if result["status"] == "success":
                    success_count += 1
                    time_sum += result.get("execution_time", 0)
                    if "metrics" in result and "accuracy" in result["metrics"]:
                        accuracy_sum += result["metrics"]["accuracy"]
            
            if success_count > 0:
                single_metrics["avg_time"] = time_sum / success_count
                single_metrics["success_rate"] = success_count / len(single_task_results)
                if accuracy_sum > 0:
                    single_metrics["accuracy"] = accuracy_sum / success_count
            
            # Get consensus metrics
            consensus_metrics = {
                "success": consensus_task_result["status"] == "success",
                "accuracy": consensus_task_result.get("metrics", {}).get("accuracy", 0)
            }
            
            # Compare metrics
            comparison["tasks"][task_name] = {
                "single_model": single_metrics,
                "consensus": consensus_metrics,
                "improvement": {
                    "accuracy": consensus_metrics["accuracy"] - single_metrics["accuracy"]
                }
            }
        
        # Calculate overall summary
        task_count = len(comparison["tasks"])
        if task_count > 0:
            acc_improvement_sum = 0
            
            for task_name, task_comparison in comparison["tasks"].items():
                # Replace the index access with a safe alternative
                value_at_idx = task_comparison["improvement"]["accuracy"] if isinstance(task_comparison["improvement"], (dict, collections.abc.Mapping)) and "accuracy" in task_comparison["improvement"] else 0  # type: ignore[index]
                acc_improvement_sum += value_at_idx
            
            comparison["summary"]["accuracy_improvement"] = acc_improvement_sum / task_count
            comparison["summary"]["single_model_avg_time"] = single_results["execution_time"] / task_count
            comparison["summary"]["consensus_avg_time"] = consensus_results["execution_time"] / task_count
        
        return comparison

    async def run_benchmark(self, tasks: List[Dict] = None, model_ids: List[str] = None) -> Dict:
        """
        Run full benchmark comparing single models vs consensus.
        
        Args:
            tasks: List of benchmark tasks (defaults to DEFAULT_TASKS)
            model_ids: List of model IDs to benchmark (defaults to all available models)
            
        Returns:
            Dictionary with benchmark results and comparison
        """
        if tasks is None:
            tasks = DEFAULT_TASKS
        
        # Run single model benchmark
        logger.info("Starting single model benchmark...")
        single_results = await self.run_single_model_benchmark(tasks, model_ids)
        
        # Run consensus benchmark
        logger.info("Starting consensus benchmark...")
        consensus_results = await self.run_consensus_benchmark(tasks)
        
        # Analyze and compare results
        logger.info("Analyzing results...")
        comparison = self.analyze_and_compare_results(single_results, consensus_results)
        
        # Save results
        self._save_results(single_results, consensus_results, comparison)
        
        # Save metrics
        self.tracker.save()
        
        return {
            "single_results": single_results,
            "consensus_results": consensus_results,
            "comparison": comparison
        }

    def _save_results(self, single_results, consensus_results, comparison):
        """Save benchmark results to JSON files."""
        # Save single model results
        with open(self.output_dir / "single_model_results.json", "w") as f:
            json.dump(single_results, f, indent=2)
        
        # Save consensus model results
        with open(self.output_dir / "consensus_results.json", "w") as f:
            json.dump(consensus_results, f, indent=2)
        
        # Save comparison results
        with open(self.output_dir / "comparison_results.json", "w") as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Results saved to directory: {self.output_dir}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks comparing single model vs consensus performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python src/benchmarks/run_benchmark.py

  # Run with specific models
  python src/benchmarks/run_benchmark.py --models claude-3-opus-20240229 gpt-4-turbo

  # Run with custom tasks from a JSON file
  python src/benchmarks/run_benchmark.py --tasks custom_tasks.json

  # Run with custom output directory
  python src/benchmarks/run_benchmark.py --output-dir ./my_benchmark_results

  # Run with increased verbosity
  python src/benchmarks/run_benchmark.py --verbose
"""
    )
    
    # Main parameters
    parser.add_argument(
        "--tasks", 
        type=str,
        help="Path to JSON file containing custom benchmark tasks"
    )
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Specific model IDs to benchmark (defaults to all available models)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="benchmark_results",
        help="Directory to save benchmark results (default: benchmark_results)"
    )
    
    # Additional options
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=2,
        help="Number of iterations for consensus generation (default: 2)"
    )
    parser.add_argument(
        "--improvement-threshold", 
        type=float, 
        default=0.05,
        help="Improvement threshold for consensus iterations (default: 0.05)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--list-models", 
        action="store_true",
        help="List available models and exit"
    )
    
    return parser.parse_args()

async def main():
    """Run the benchmark script with command line arguments."""
    args = parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # List models if requested
    if args.list_models:
        print("Available models:")
        for model in AVAILABLE_MODELS:
            print(f"  - {model.name}")
        return
    
    # Initialize the benchmark runner
    runner = BenchmarkRunner(output_dir=args.output_dir)
    
    # Update consensus configuration if necessary
    if args.iterations or args.improvement_threshold:
        runner.consensus_config["iterations"]["max_iterations"] = args.iterations
        runner.consensus_config["iterations"]["improvement_threshold"] = args.improvement_threshold
        runner.consensus = ConsensusSynthesizer(runner.consensus_config)
        runner.consensus.router_client = runner.router_client
    
    # Load custom tasks if provided
    tasks = None
    if args.tasks:
        try:
            with open(args.tasks, "r") as f:
                tasks = json.load(f)
            logger.info(f"Loaded {len(tasks)} custom tasks from {args.tasks}")
        except Exception as e:
            logger.error(f"Failed to load tasks from {args.tasks}: {e}")
            sys.exit(1)
    
    # Run the benchmark
    try:
        logger.info("Starting benchmark run...")
        results = await runner.run_benchmark(tasks=tasks, model_ids=args.models)
        
        # Print summary
        print("\nBenchmark Summary:")
        print("-----------------")
        print(f"Tasks run: {len(results['comparison']['tasks'])}")
        print(f"Single model avg execution time: {results['comparison']['summary']['single_model_avg_time']:.2f}s")
        print(f"Consensus avg execution time: {results['comparison']['summary']['consensus_avg_time']:.2f}s")
        print(f"Average accuracy improvement: {results['comparison']['summary']['accuracy_improvement']*100:.2f}%")
        print(f"\nDetailed results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

