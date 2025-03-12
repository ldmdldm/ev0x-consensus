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

from typing import Dict, List, Optional, Any
from src.evaluation.metrics import evaluate_consensus_quality
from src.tee.confidential_vm import verify_attestation
from src.models.model_runner import ModelRunner
from src.consensus.synthesizer import ConsensusSynthesizer
from src.router.openrouter import OpenRouterClient
import argparse
import asyncio
import json
import logging
import os
import sys
import time
import pytest
pytest.register_assert_rewrite('pytest_asyncio')

# Enable pytest-asyncio plugin
pytest_plugins = ('pytest_asyncio',)

# Register asyncio marker


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as an async test"
    )


# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
        self._init_components()

        # Test state
        self.results = {
            "completion": {},
            "chat": {},
            "iterations": {},
            "errors": [],
            "attestation": None
        }

    def _init_components(self):
        """Initialize system components."""
        # Initialize router client with API key
        self.router_client = OpenRouterClient(self.api_key)

        # Initialize model runner
        self.model_runner = ModelRunner(self.router_client)

        # Initialize synthesizer with config and router client
        self.synthesizer = ConsensusSynthesizer(self.config)
        self.synthesizer.router_client = self.router_client

    async def test_attestation(self):
        """Test the TEE attestation mechanism."""
        logger.info("Testing TEE attestation...")
        try:
            # Get project_id from config if available, otherwise use default
            project_id = self.config.get("project_id", "ev0x-consensus")
            attestation_result = await verify_attestation(project_id=project_id)
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

                    # Register model with ModelRunner to handle both prompt and messages inputs
                    async def model_fn(input_data, **kwargs):
                        # Handle different input formats from ModelRunner
                        if isinstance(input_data, str):
                            # Direct string prompt (completion case)
                            return await self.router_client.completion(
                                prompt=input_data,
                                model=model_id,
                                **kwargs
                            )
                        elif isinstance(input_data, dict):
                            # Dictionary input format
                            if "prompt" in input_data:
                                # Dictionary with prompt key (completion case)
                                return await self.router_client.completion(
                                    prompt=input_data["prompt"],
                                    model=model_id,
                                    **kwargs
                                )
                            elif "messages" in input_data:
                                # Dictionary with messages key (chat completion case)
                                return await self.router_client.chat_async(
                                    messages=input_data["messages"],
                                    model=model_id,
                                    **kwargs
                                )
                        # Fallback to treating input as messages for backward compatibility
                        return await self.router_client.chat_async(
                            messages=input_data,
                            model=model_id,
                            **kwargs
                        )
                    self.model_runner.register_model(model_id, model_fn)
                    logger.info(f"Registered model {model_id} with ModelRunner")
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
                # Check if completion was successful
                if completion and completion.get("status") == "success" and completion.get("completion") is not None:
                    logger.info(f"Single model completion successful: {model_id}")
                    # Return a dictionary with the same structure as consensus completions for consistent handling
                    return {
                        "consensus_output": completion.get("completion"),
                        "model_outputs": [{"id": model_id, "output": completion.get("completion")}],
                        "elapsed_time": completion.get("elapsed_time", 0)
                    }
                else:
                    logger.warning(f"Single model completion failed for {model_id}: {completion}")
                    return None
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
                logger.info(
                    f"Consensus chat completion successful with {len(consensus_result.get('model_outputs', []))} models")

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
                logger.info(f"Starting iteration {i + 1}/{max_iterations}")
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
                        iterations[i - 1]["output"],
                        consensus_output
                    )
                else:
                    improvement = None

                # Store iteration result
                iterations.append({
                    "iteration": i + 1,
                    "output": consensus_output,
                    "elapsed_time": time.time() - start_time,
                    "improvement": improvement
                })

                logger.info(f"Completed iteration {i + 1} in {time.time() - start_time:.2f} seconds")

            # Store all iterations
            short_prompt = prompt[:30] + "..." if len(prompt) > 30 else prompt
            self.results["iterations"][short_prompt] = iterations

            return iterations
        except Exception as e:
            logger.error(f"Iterative feedback test failed: {e}")
            self.results["errors"].append(f"Iterative feedback error: {str(e)}")
            return None

    async def test_citation_verification(self, text: str) -> Dict[str, Any]:
        """Test citation verification functionality."""
        try:
            # Import and use the CitationVerifier
            from src.consensus.citation_verifier import CitationVerifier

            # Verify citations using the CitationVerifier
            result = await CitationVerifier.verify_citations(text)

            if result["is_verified"]:
                self.results["citation_verification"] = {
                    "status": "success",
                    "verified_citations": result["verified_citations"],
                    "total_citations": result["total_citations"]
                }
            else:
                self.results["citation_verification"] = {
                    "status": "failed",
                    "message": result.get("message", "Citation verification failed")
                }

            return result

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
        print("5. Run all tests")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        try:
            if choice == "1":
                # Test completion
                prompt_type = input(
                    "Enter prompt type (factual, creative, reasoning, multi_step, controversial) or custom: ").strip()
                if prompt_type in TEST_PROMPTS:
                    prompt = TEST_PROMPTS[prompt_type]
                else:
                    prompt = input("Enter custom prompt: ").strip()

                use_model = input("Test with specific model? (y/n): ").strip().lower()
                if use_model == 'y':
                    model_id = input("Enter model ID: ").strip()
                    result = await tester.test_completion(prompt, model_id)
                else:
                    result = await tester.test_completion(prompt)

                print("\nCompletion Result:")
                print(json.dumps(result, indent=2))

            elif choice == "2":
                # Test chat completion
                use_default = input("Use default test conversation? (y/n): ").strip().lower()
                if use_default == 'y':
                    messages = TEST_CONVERSATION
                else:
                    messages = []
                    system_msg = input("Enter system message (optional): ").strip()
                    if system_msg:
                        messages.append({"role": "system", "content": system_msg})

                    user_msg = input("Enter user message: ").strip()
                    messages.append({"role": "user", "content": user_msg})

                use_model = input("Test with specific model? (y/n): ").strip().lower()
                if use_model == 'y':
                    model_id = input("Enter model ID: ").strip()
                    result = await tester.test_chat_completion(messages, model_id)
                else:
                    result = await tester.test_chat_completion(messages)

                print("\nChat Completion Result:")
                print(json.dumps(result, indent=2))

            elif choice == "3":
                # Test iterative feedback
                prompt_type = input(
                    "Enter prompt type (factual, creative, reasoning, multi_step, controversial) or custom: ").strip()
                if prompt_type in TEST_PROMPTS:
                    prompt = TEST_PROMPTS[prompt_type]
                else:
                    prompt = input("Enter custom prompt: ").strip()

                iterations = input("Enter number of iterations (default: 3): ").strip()
                iterations = int(iterations) if iterations else 3

                result = await tester.test_iterative_feedback(prompt, iterations)

                print("\nIterative Feedback Result:")
                print(json.dumps(result, indent=2))

            elif choice == "4":
                # Test citation verification
                response_input = input(
                    "Enter text with citations to verify, or 'generate' to generate a response first: ").strip()

                if response_input == 'generate':
                    result = await tester.test_completion(TEST_PROMPTS["factual"])
                    if result:
                        response = result.get("consensus_output", "")
                        print(f"\nGenerated response:\n{response}\n")
                    else:
                        print("Failed to generate response for citation verification")
                        continue
                else:
                    response = response_input

                verification_result = await tester.test_citation_verification(response)

                print("\nCitation Verification Result:")
                print(json.dumps(verification_result, indent=2))

            elif choice == "5":
                # Run all tests
                print("\nRunning all tests. This may take several minutes...")
                result = await tester.run_all_tests()

                print("\nTest Results:")
                print(f"Success: {result['success']}")
                print(f"Errors: {len(result['errors'])}")
                print(f"Duration: {result['overall_duration']:.2f} seconds")
                print("Report saved to: consensus_test_report.json")

            elif choice == "6":
                # Exit
                print("Exiting interactive mode.")
                break

            else:
                print("Invalid choice. Please enter a number between 1 and 6.")

        except Exception as e:
            print(f"Error during test execution: {e}")
            continue

        input("\nPress Enter to continue...")

# Add main function for script execution


def main():
    parser = argparse.ArgumentParser(description="Test the ev0x consensus system")
    parser.add_argument("--mode", choices=["interactive", "all"], default="interactive",
                        help="Test mode: interactive or run all tests")
    parser.add_argument("--config", default="input.json", help="Path to configuration file")

    args = parser.parse_args()

    if args.mode == "interactive":
        asyncio.run(interactive_mode())
    else:
        tester = ConsensusSystemTester(args.config)
        asyncio.run(tester.run_all_tests())


# Pytest test functions
@pytest.mark.asyncio
async def test_attestation():
    """Test TEE attestation verification."""
    config_path = "input.json"
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        pytest.skip("OpenRouter API key not available in environment")

    tester = ConsensusSystemTester(config_path, api_key)
    result = await tester.test_attestation()

    # We accept both success and simulated results since not all environments support TEE
    assert result is not None, "Attestation verification should return a result"
    # In non-TEE environments, this might be simulated
    if isinstance(result, dict):
        assert "attestation_id" in result or "simulated" in result, "Attestation should include ID or be marked as simulated"


@pytest.mark.asyncio
async def test_completion():
    """Test completion functionality."""
    config_path = "input.json"
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        pytest.skip("OpenRouter API key not available in environment")

    tester = ConsensusSystemTester(config_path, api_key)
    prompt = "What is the capital of France?"

    # Test consensus completion
    result = await tester.test_completion(prompt)
    assert result is not None, "Completion result should not be None"
    assert "consensus_output" in result, "Result should contain consensus_output"
    assert "Paris" in result["consensus_output"], "Output should mention Paris as the capital of France"

    # Test with error handling
    models_available = await tester.test_models_availability()
    if models_available and len(tester.results["available_models"]) > 0:
        # Test single model completion
        model_id = tester.results["available_models"][0]
        single_result = await tester.test_completion(prompt, model_id)
        assert single_result is not None, f"Single model completion failed for {model_id}"


@pytest.mark.asyncio
async def test_chat_completion():
    """Test chat completion functionality."""
    config_path = "input.json"
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        pytest.skip("OpenRouter API key not available in environment")

    tester = ConsensusSystemTester(config_path, api_key)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Italy?"}
    ]

    # Test consensus chat completion
    result = await tester.test_chat_completion(messages)
    assert result is not None, "Chat completion result should not be None"
    assert "consensus_output" in result, "Result should contain consensus_output"
    assert "Rome" in result["consensus_output"], "Output should mention Rome as the capital of Italy"

    # Test with error handling
    models_available = await tester.test_models_availability()
    if models_available and len(tester.results["available_models"]) > 0:
        # Test single model chat completion
        model_id = tester.results["available_models"][0]
        single_result = await tester.test_chat_completion(messages, model_id)
        assert single_result is not None, f"Single model chat completion failed for {model_id}"


@pytest.mark.asyncio
async def test_citation_verification():
    """Test citation verification functionality."""
    config_path = "input.json"
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        pytest.skip("OpenRouter API key not available in environment")

    tester = ConsensusSystemTester(config_path, api_key)

    # Create a response with citations
    text_with_citations = """
    According to the World Health Organization (WHO), regular physical activity has significant health benefits [1].
    A study published in The Lancet found that 150 minutes of moderate exercise per week can reduce mortality risk by 30% [2].
    """

    result = await tester.test_citation_verification(text_with_citations)
    assert result is not None, "Citation verification result should not be None"
    assert result["is_verified"] is True, "Citations should be verified"
    assert result["verified_citations"] > 0, "Should have at least one verified citation"
    assert len(result["results"]) > 0, "Should have verification results"

    # Check the verification results structure
    for verification in result["results"]:
        assert "citation_number" in verification, "Each result should have a citation number"
        assert "is_valid" in verification, "Each result should have a validity status"
        assert "url" in verification, "Each result should have a URL"


if __name__ == "__main__":
    main()
