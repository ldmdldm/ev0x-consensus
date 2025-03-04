import pytest
import asyncio
import json
import numpy as np
from unittest.mock import MagicMock, patch

from src.models.model_runner import ModelRunner, AIModel
from src.consensus.synthesizer import ConsensusSynthesizer
from src.rewards.shapley import ShapleyCalculator


# Mock AI models for testing
class MockAIModel(AIModel):
    def __init__(self, name, predictions):
        """
        Initialize a mock AI model with predefined predictions
        
        Args:
            name (str): Name of the mock model
            predictions (dict): Dictionary mapping input queries to output predictions
        """
        self.name = name
        self.predictions = predictions
        self.call_count = 0
        
    async def generate(self, prompt):
        """
        Generate a response based on predefined predictions
        
        Args:
            prompt (str): The input prompt
        
        Returns:
            str: The predefined response for this prompt
        """
        self.call_count += 1
        if prompt in self.predictions:
            return self.predictions[prompt]
        return f"Default response from {self.name}"


# Test fixtures
@pytest.fixture
def mock_models():
    """
    Create a set of mock AI models with predefined responses
    """
    model1 = MockAIModel("model1", {
        "What is the capital of France?": "Paris",
        "What is 2+2?": "4",
        "Recommend a stock to buy": "AAPL"
    })
    
    model2 = MockAIModel("model2", {
        "What is the capital of France?": "Paris",
        "What is 2+2?": "4",
        "Recommend a stock to buy": "GOOGL"
    })
    
    model3 = MockAIModel("model3", {
        "What is the capital of France?": "Lyon", # Deliberately incorrect
        "What is 2+2?": "5", # Deliberately incorrect
        "Recommend a stock to buy": "MSFT"
    })
    
    return [model1, model2, model3]


@pytest.fixture
def model_runner(mock_models):
    """
    Create a ModelRunner with mock models
    """
    return ModelRunner(models=mock_models)


@pytest.fixture
def consensus_synthesizer():
    """
    Create a ConsensusSynthesizer for testing
    """
    return ConsensusSynthesizer()


@pytest.fixture
def shapley_calculator():
    """
    Create a ShapleyCalculator for testing
    """
    return ShapleyCalculator()


# Tests for running multiple models simultaneously
@pytest.mark.asyncio
async def test_parallel_model_execution(model_runner):
    """
    Test that models can be run in parallel
    """
    prompt = "What is the capital of France?"
    results = await model_runner.run_all_models(prompt)
    
    # Check that results were returned for all models
    assert len(results) == 3
    
    # Check that each model was called exactly once
    for model in model_runner.models:
        assert model.call_count == 1
    
    # Check the actual predictions
    assert results["model1"] == "Paris"
    assert results["model2"] == "Paris"
    assert results["model3"] == "Lyon"


# Tests for comparing and synthesizing outputs
def test_consensus_basic_agreement(consensus_synthesizer):
    """
    Test consensus when models agree
    """
    model_outputs = {
        "model1": "Paris",
        "model2": "Paris",
        "model3": "Paris"
    }
    
    result = consensus_synthesizer.generate_consensus(model_outputs)
    
    assert result["consensus"] == "Paris"
    assert result["confidence"] > 0.9  # High confidence for complete agreement


def test_consensus_majority_agreement(consensus_synthesizer):
    """
    Test consensus with majority agreement
    """
    model_outputs = {
        "model1": "Paris",
        "model2": "Paris",
        "model3": "Lyon"
    }
    
    result = consensus_synthesizer.generate_consensus(model_outputs)
    
    assert result["consensus"] == "Paris"
    assert 0.6 < result["confidence"] < 0.9  # Medium confidence for majority


def test_consensus_no_agreement(consensus_synthesizer):
    """
    Test consensus with no clear agreement
    """
    model_outputs = {
        "model1": "Paris",
        "model2": "Lyon",
        "model3": "Marseille"
    }
    
    result = consensus_synthesizer.generate_consensus(model_outputs)
    
    # Should select one of the options or indicate low confidence
    assert result["consensus"] in ["Paris", "Lyon", "Marseille", "Uncertain"]
    assert result["confidence"] < 0.5  # Low confidence


# Tests for Shapley value calculation
def test_shapley_values_equal_contribution(shapley_calculator):
    """
    Test Shapley value calculation when all models contribute equally
    """
    model_outputs = {
        "model1": "Paris",
        "model2": "Paris",
        "model3": "Paris"
    }
    ground_truth = "Paris"
    
    # All models agree with ground truth, so should have equal Shapley values
    shapley_values = shapley_calculator.calculate(model_outputs, ground_truth)
    
    # Should have 3 values (one per model)
    assert len(shapley_values) == 3
    
    # All values should be approximately equal and sum to 1
    assert abs(shapley_values["model1"] - shapley_values["model2"]) < 0.01
    assert abs(shapley_values["model2"] - shapley_values["model3"]) < 0.01
    assert 0.99 < sum(shapley_values.values()) < 1.01


def test_shapley_values_unequal_contribution(shapley_calculator):
    """
    Test Shapley value calculation when models contribute differently
    """
    model_outputs = {
        "model1": "Paris",
        "model2": "Paris",
        "model3": "Lyon"  # Incorrect answer
    }
    ground_truth = "Paris"
    
    shapley_values = shapley_calculator.calculate(model_outputs, ground_truth)
    
    # model1 and model2 should have higher Shapley values than model3
    assert shapley_values["model1"] > shapley_values["model3"]
    assert shapley_values["model2"] > shapley_values["model3"]
    
    # The sum should still be approximately 1
    assert 0.99 < sum(shapley_values.values()) < 1.01


# Integration tests
@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """
    Test the entire workflow from running models to calculating rewards
    """
    # Create mock models
    model1 = MockAIModel("model1", {"What is 2+2?": "4"})
    model2 = MockAIModel("model2", {"What is 2+2?": "4"})
    model3 = MockAIModel("model3", {"What is 2+2?": "5"})  # Incorrect
    
    # Set up the components
    model_runner = ModelRunner([model1, model2, model3])
    consensus_synthesizer = ConsensusSynthesizer()
    shapley_calculator = ShapleyCalculator()
    
    # Run the entire pipeline
    prompt = "What is 2+2?"
    ground_truth = "4"
    
    # Step 1: Get predictions from all models
    model_outputs = await model_runner.run_all_models(prompt)
    
    # Step 2: Generate consensus
    consensus_result = consensus_synthesizer.generate_consensus(model_outputs)
    
    # Step 3: Calculate rewards using Shapley values
    shapley_values = shapley_calculator.calculate(model_outputs, ground_truth)
    
    # Assertions
    assert consensus_result["consensus"] == "4"  # Correct consensus
    assert consensus_result["confidence"] > 0.6  # Reasonable confidence
    
    # model1 and model2 should have higher rewards than model3
    assert shapley_values["model1"] > shapley_values["model3"]
    assert shapley_values["model2"] > shapley_values["model3"]


@pytest.mark.asyncio
async def test_multiple_prompts_workflow():
    """
    Test handling multiple prompts in sequence
    """
    # Create mock models
    model1 = MockAIModel("model1", {
        "What is 2+2?": "4",
        "What is the capital of France?": "Paris"
    })
    model2 = MockAIModel("model2", {
        "What is 2+2?": "4",
        "What is the capital of France?": "Paris"
    })
    model3 = MockAIModel("model3", {
        "What is 2+2?": "5",  # Incorrect
        "What is the capital of France?": "Paris"
    })
    
    # Set up components
    model_runner = ModelRunner([model1, model2, model3])
    consensus_synthesizer = ConsensusSynthesizer()
    
    # Run multiple prompts
    prompts = ["What is 2+2?", "What is the capital of France?"]
    results = []
    
    for prompt in prompts:
        # Get model outputs
        model_outputs = await model_runner.run_all_models(prompt)
        
        # Generate consensus
        consensus_result = consensus_synthesizer.generate_consensus(model_outputs)
        
        results.append({
            "prompt": prompt,
            "model_outputs": model_outputs,
            "consensus": consensus_result
        })
    
    # Check results
    assert len(results) == 2
    
    # First prompt results
    assert results[0]["prompt"] == "What is 2+2?"
    assert results[0]["consensus"]["consensus"] == "4"
    
    # Second prompt results
    assert results[1]["prompt"] == "What is the capital of France?"
    assert results[1]["consensus"]["consensus"] == "Paris"
    assert results[1]["consensus"]["confidence"] > 0.9  # High confidence (all agreed)

