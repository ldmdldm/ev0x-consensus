#!/usr/bin/env python3
"""
Test script for checking model availability in OpenRouterClient.
This script will help debug issues with model availability in the consensus system.
"""

from src.router.openrouter import OpenRouterClient
import asyncio
import json
import logging
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from the src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_model_availability():
    """
    Test the OpenRouterClient's model availability.
    Reads the model ID from input.json, fetches all available models,
    and checks if the specified model is available.
    """
    # Read model ID from input.json
    try:
        with open('input.json', 'r') as f:
            config = json.load(f)
            models = config.get('models', [])
            if not models:
                logger.error("No models array found in input.json")
                return

            first_model = models[0]
            target_model = first_model.get('id', '')
            if not target_model:
                logger.error("No model ID specified in the first model of input.json")
                return

            logger.info(f"Looking for model: {target_model}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error reading input.json: {e}")
        return

    # Initialize the OpenRouterClient
    try:
        api_key = os.environ.get('OPEN_ROUTER_API_KEY')
        if not api_key:
            logger.error("OPEN_ROUTER_API_KEY environment variable not set")
            return

        client = OpenRouterClient(api_key=api_key)
        logger.info("OpenRouterClient initialized successfully")

        # Get all available models
        models = await client.get_models()
        logger.info(f"Retrieved {len(models)} models from OpenRouter API")

        # Print all model IDs for debugging
        logger.info("Available models:")
        for i, model in enumerate(models, 1):
            model_id = model.get('id', 'Unknown ID')
            logger.info(f"  {i}. {model_id}")

        # Check if the target model is available
        is_available = await client.check_model_availability(target_model)
        if is_available:
            logger.info(f"SUCCESS: Model '{target_model}' is available!")
        else:
            logger.error(f"ERROR: Model '{target_model}' is NOT available!")

            # Try to find similar models
            logger.info("Checking for similar models...")
            found_similar = False
            for model in models:
                model_id = model.get('id', '')
                if target_model.lower() in model_id.lower():
                    logger.info(f"Found similar model: {model_id}")
                    found_similar = True

            if not found_similar:
                logger.info("No similar models found. Check if the model ID is correct.")

    except Exception as e:
        logger.error(f"Error testing model availability: {e}")


if __name__ == "__main__":
    asyncio.run(test_model_availability())
