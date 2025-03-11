#!/usr/bin/env python3
"""
Comprehensive test script for the ev0x consensus system.

This script tests the entire flow of the consensus system, including:
1. OpenRouter integration
2. Model consensus across multiple models
3. Iterative feedback loop
4. Citation verification
5. Both completion and chat completion modes
6. Error handling and recovery
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Union

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.router.openrouter import OpenRouterClient
from src.consensus.synthesizer import ConsensusSynthesizer
from src.factual.citation import verify_citations
from src.models.model_runner import ModelRunner
from src.tee.confidential_vm import verify_attestation
from src.evaluation.metrics import evaluate_consensus_quality

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_consensus_flow.log")
    ]
)

logger = logging.getLogger("consensus_test")

# Test prompts
TEST_PROMPTS = {
    "factual": "What are the largest cities in California by population? Please provide population figures and cite your sources.",
    "creative": "Write a short poem about artificial intelligence and human creativity.",
    "reasoning": "Explain the prisoner's dilemma and how it relates to game theory.",
    "multi_step": "Design a simple algorithm to find the nth Fibonacci number and analyze its time complexity.",
    "controversial": "Discuss the ethical implications of using AI in autonomous weapons systems."
}

# Test conversation for chat mode
TEST_CONVERSATION = [
    {"role": "system", "content": "You are a helpful, factual assistant that provides accurate information with citations when appropriate."},
    {"role": "user", "content": "Tell me about quantum computing. What are qubits and how do they work?"},
    {"role": "assistant", "content": "Quantum computing is a type of computing that uses quantum mechanics to process information. The basic unit of quantum information is the qubit."},
    {"role": "user", "content": "Can you elaborate on quantum entanglement and its role in quantum computing?"}
]

class ConsensusSystemTester:
    """Test harness for the consensus system."""
    
    def __init__(self, config_path: str, api_key: Optional[str] = None):
        """Initialize the tester with configuration."""
        self.config_path = config_path
        self.api_key = api_key or os.getenv("OPEN_ROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided and not found in environment")
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
        
        # Initialize components
        self.router_client = OpenRouterClient(self.api_key)
        self.model_runner = ModelRunner(self.router_client)
        self.synthesizer = ConsensusSynthesizer(self.config)
        
        # Test state
        self.results = {
            "completion": {},
            "chat": {},
            "iterations": {},
            "errors": [],
            "attestation": None
        }

    async def test_attestation(self):
        """Test the TEE attestation mechanism."""
        logger.info("Testing TEE attestation...")
        try:
            attestation_result = await verify_attestation()
            self.results["attestation"] = attestation_result
            logger.info(f"Attestation result: {attestation_result}")
            return attestation_result
        except Exception as e:
            logger.error(f"Attestation verification failed: {e}")
            self.results["errors"].append(f"Attestation error: {str(e)}")
            return False

    async def test_models_availability(self):
        """Test if all configured models are available."""
        logger.info("Testing model availability...")
        available_models = []
        unavailable_models = []
        
        for model in self.config.get("models", []):
            model_id = model.get("id")
            try:
                is_available = await self.router_client.check_model_availability(model_id)
                if is_available:
                    available_models.append(model_id)
                    logger.info(f"Model {model_id} is available")
                else:
                    unavailable_models.append(model_id)
                    logger.warning(f"Model {model_id} is not available")
            except Exception as e:
                logger.error(f"Error checking model {model_id}: {e}")
                unavailable_models.append(model_id)
                self.results["errors"].append(f"Model availability error ({model_id}): {str(e)}")
        
        self.results["available_models"] = available_models
        self.results["unavailable_models"] = unavailable_models
        
        return len(unavailable_models) == 0

    async def test_completion(self, prompt: str, model_id: Optional[str] = None):
        """Test the completion mode with a specific prompt."""
        logger.info(f"Testing completion mode with prompt: {prompt[:50]}...")
        try:
            if model_id:
                # Test single model
                completion = await self.model_runner.generate_completion(
                    prompt=prompt,
                    model=model_id,
                    max_tokens=self.config.get("max_tokens", 1000),
                    temperature=self.config.get("temperature", 0.7)
                )
                logger.info(f"Single model completion successful: {model_id}")
                return completion
            else:
                # Test consensus completion
                consensus_result = await self.synthesizer.generate_consensus_completion(prompt)
                logger.info(f"Consensus completion successful with {len(consensus_result.get('model_outputs', []))} models")
                
                # Store result
                short_prompt = prompt[:30] + "..." if len(prompt) > 30 else prompt
                self.results["completion"][short_prompt] = {
                    "result": consensus_result.get("consensus_output"),
                    "models_used": [m.get("id") for m in consensus_result.get("model_outputs", [])],
                    "elapsed_time": consensus_result.get("elapsed_time")
                }
                
                return consensus_result
        except Exception as e:
            logger.error(f"Completion test failed: {e}")
            self.results["errors"].append(f"Completion error: {str(e)}")
            return None

    async def test_chat_completion(self, messages: List[Dict], model_id: Optional[str] = None):
        """Test the chat completion mode with a conversation."""
        logger.info(f"Testing chat completion mode with {len(messages)} messages...")
        try:
            if model_id:
                # Test single model
                chat_completion = await self.model_runner.generate_chat_completion(
                    messages=messages,
                    model=model_id,
                    max_tokens=self.config.get("max_tokens", 1000),
                    temperature=self.config.get("temperature", 0.7)
                )
                logger.info(f"Single model chat completion successful: {model_id}")
                return chat_completion
            else:
                # Test consensus chat completion
                consensus_result = await self.synthesizer.generate_consensus_chat_completion(messages)
                logger.info(f"Consensus chat completion successful with {len(consensus_result.get('model_outputs', []))} models")
                
                # Store result
                last_message = messages[-1]["content"]
                short_message = last_message[:30] + "..." if len(last_message) > 30 else last_message
                self.results["chat"][short_message] = {
                    "result": consensus_result.get("consensus_output"),
                    "models_used": [m.get("id") for m in consensus_result.get("model_outputs", [])],
                    "elapsed_time": consensus_result.get("elapsed_time")
                }
                
                return consensus_result
        except Exception as e:
            logger.error(f"Chat completion test failed: {e}")
            self.results["errors"].append(f"Chat completion error: {str(e)}")
            return None

    async def test_iterative_feedback(self, prompt: str, max_iterations: int = 3):
        """Test the iterative feedback loop."""
        logger.info(f"Testing iterative feedback loop with {max_iterations} iterations...")
        try:
            iterations = []
            current_prompt = prompt
            
            for i in range(max_iterations):
                logger.info(f"Starting iteration {i+1}/{max_iterations}")
                start_time = time.time()
                
                # Generate consensus
                consensus_result = await self.synthesizer.generate_consensus_completion(current_prompt)
                
                # Get consensus output
                consensus_output = consensus_result.get("consensus_output", "")
                
                # Prepare for next iteration if needed
                if i < max_iterations - 1:
                    # Create feedback prompt for next iteration
                    feedback_prompt = (
                        f"Previous response: {consensus_output}\n\n"
                        f"Please improve the above response by:\n"
                        f"1. Adding more specific details\n"
                        f"2. Ensuring factual accuracy\n"
                        f"3. Providing proper citations\n"
                        f"4. Making the explanation clearer\n\n"
                        f"Original query: {prompt}"
                    )
                    current_prompt = feedback_prompt
                
                # Measure improvement
                if i > 0:
                    improvement = evaluate_consensus_quality(
                        iterations[i-1]["output"],
                        consensus_output
                    )
                else:
                    improvement = None
                
                # Store iteration result
                iterations.append({
                    "iteration": i+1,
                    "output": consensus_output,
                    "elapsed_time": time.time() - start_time,
                    "improvement": improvement
                })
                
                logger.info(f"Completed iteration {i+1} in {time.time() - start_time:.2f} seconds")
            
            # Store all iterations
            short_prompt = prompt[:30] + "..." if len(prompt) > 30 else prompt
            self.results["iterations"][short_prompt] = iterations
            
            return iterations
        except Exception as e:
            logger.error(f"Iterative feedback test failed: {e}")
            self.results["errors"].append(f"Iterative feedback error: {str(e)}")
            return None

    async def test_citation_verification(self, response: str):
        """Test the citation verification mechanism."""
        logger.info("Testing citation verification...")
        try:
            verification_result = verify_citations(response)
            logger.info(f"Citation verification completed with {len(verification_result.get('citations', []))} citations")
            return verification_result
        except Exception as e:
            logger.error(f"Citation verification failed: {e}")
            self.results["errors"].append(f"Citation verification error: {str(e)}")
            return None

    async def test_error_handling(self):
        """Test error handling by intentionally causing errors."""
        logger.info("Testing error handling...")
        error_tests = [
            # Test with invalid model
            self.test_completion(TEST_PROMPTS["factual"], "nonexistent-model"),
            # Test with empty prompt
            self.test_completion(""),
            # Test with excessive token request
            self.test_completion("Generate a 10,000 word essay on quantum physics", 
                               model_id=self.config["models"][0]["id"])
        ]
        
        results = await asyncio.gather(*error_tests, return_exceptions=True)
        
        recovery_successful = True
        for result in results:
            if not isinstance(result, Exception):
                recovery_successful = False
                logger.warning("Error test did not raise an exception as expected")
        
        if recovery_successful:
            logger.info("Error handling tests passed - all errors were properly caught")
        else:
            logger.warning("Some error tests did not behave as expected")
        
        return recovery_successful

    async def run_all_tests(self):
        """Run all tests and generate a comprehensive report."""
        start_time = time.time()
        logger.info("Starting comprehensive consensus system test")
        
        # Test attestation
        await self.test_attestation()
        
        # Test model availability
        await self.test_models_availability()
        
        # Test completions with different prompts
        for prompt_type, prompt in TEST_PROMPTS.items():
            logger.info(f"Testing {prompt_type} prompt")
            await self.test_completion(prompt)
        
        # Test chat completion
        await self.test_chat_completion(TEST_CONVERSATION)
        
        # Test iterative feedback
        await self.test_iterative_feedback(TEST_PROMPTS["factual"])
        
        # Test citation verification with a factual response
        factual_result = await self.test_completion(TEST_PROMPTS["factual"])
        if factual_result:
            await self.test_citation_verification(factual_result.get("consensus_output", ""))
        
        # Test error handling
        await self.test_error_handling()
        
        # Generate report
        self.results["overall_duration"] = time.time() - start_time
        self.results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.results["success"] = len(self.results["errors"]) == 0
        
        # Save report
        report_path = "consensus_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Test completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Test report saved to {report_path}")
        
        return self.results


async def interactive_mode():
    """Run the consensus test in interactive mode."""
    print("=== Consensus System Interactive Test ===")
    
    # Get configuration path
    config_path = input("Enter path to configuration file (default: input.json): ").strip() or "input.json"
    
    # Get API key
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        api_key = input("Enter OpenRouter API key: ").strip()
    
    # Initialize tester
    tester = ConsensusSystemTester(config_path, api_key)
    
    while True:
        print("\nTest Options:")
        print("1. Test completion")
        print("2. Test chat completion")
        print("3. Test iterative feedback")
        print("4. Test citation verification")
        print("5

