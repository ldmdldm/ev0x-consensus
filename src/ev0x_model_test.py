#!/usr/bin/env python3
"""
Model Registration Demo Script

This script demonstrates how to properly register models with the ModelRunner
using the OpenRouterClient.
"""

import os
import asyncio
from dotenv import load_dotenv
from src.router.openrouter import OpenRouterClient
from src.models.model_runner import ModelRunner


async def main():
    # 1. Load environment variables from .env
    print("1. Loading environment variables from .env")
    load_dotenv()

    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("Error: OPEN_ROUTER_API_KEY not found in .env file")
        return

    # 2. Create an OpenRouterClient
    print("\n2. Creating OpenRouterClient")
    router_client = OpenRouterClient(api_key)

    # 3. Get available models from OpenRouter
    print("\n3. Getting available models from OpenRouter")
    try:
        models = await router_client.get_models()
        print(f"Found {len(models)} models from OpenRouter")

        # Print first 5 models as examples
        for i, model in enumerate(models[:5]):
            print(f"  - {model['id']}")
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more")

    except Exception as e:
        print(f"Error getting models: {e}")
        return

    # 4. Create a ModelRunner
    print("\n4. Creating ModelRunner")
    model_runner = ModelRunner(router_client)

    # 5. Register models with the ModelRunner
    print("\n5. Registering models with the ModelRunner")
    # First let's check if our target model is available
    target_model = "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"
    model_available = await router_client.check_model_availability(target_model)

    if model_available:
        print(f"Target model '{target_model}' is available")
        # Register the model with the ModelRunner
        # We need to register a callable function that will use the router_client

        async def model_fn(input_data, **kwargs):
            if "prompt" in input_data:
                response = await router_client.completion(
                    model=target_model,
                    prompt=input_data["prompt"],
                    max_tokens=input_data.get("max_tokens", 1000),
                    temperature=input_data.get("temperature", 0.7),
                )
                # Extract the completion from the response
                if isinstance(response, dict):
                    # Handle standard OpenRouter API format with choices array
                    if "choices" in response and isinstance(response["choices"], list) and len(response["choices"]) > 0:
                        choice = response["choices"][0]
                        # Check for message content format (chat completions API)
                        if isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict):
                            return {"completion": choice["message"]["content"], "status": "success"}
                        # Check for text field format (completions API)
                        elif isinstance(choice, dict) and "text" in choice:
                            return {"completion": choice["text"], "status": "success"}
                    # Fall back to direct text field if present
                    elif "text" in response:
                        return {"completion": response["text"], "status": "success"}

                # If we couldn't extract text in expected format, return the raw response
                return {"completion": str(response), "status": "success"}
            else:
                raise ValueError("Input data must contain a 'prompt' field")

        model_runner.register_model(target_model, model_fn)
        print(f"Registered model '{target_model}' with ModelRunner")
    else:
        print(f"Target model '{target_model}' is NOT available")
        # If our target model isn't available, register the first available model
        if models:
            alternative_model = models[0]['id']
            print(f"Using alternative model '{alternative_model}'")

            # Register a callable function that uses router_client for the alternative model
            async def alt_model_fn(input_data, **kwargs):
                if "prompt" in input_data:
                    response = await router_client.completion(
                        model=alternative_model,
                        prompt=input_data["prompt"],
                        max_tokens=input_data.get("max_tokens", 1000),
                        temperature=input_data.get("temperature", 0.7),
                        **kwargs
                    )
                    # Extract the completion from the response
                    # Extract the completion from the response
                    # Handle standard OpenRouter API format with choices array
                    if "choices" in response and isinstance(response["choices"], list) and len(response["choices"]) > 0:
                        choice = response["choices"][0]
                        # Check for message content format (chat completions API)
                        if isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict):
                            return {"completion": choice["message"]["content"], "status": "success"}
                        # Check for text field format (completions API)
                        # Check for text field format (completions API)
                        elif isinstance(choice, dict) and "text" in choice:
                            return {"completion": choice["text"], "status": "success"}
                        # Fall back to direct text field if present
                        elif "text" in response:
                            return {"completion": response["text"], "status": "success"}
                    # If we couldn't extract text in expected format, return the raw response
                    return {"completion": str(response), "status": "success"}
                else:
                    raise ValueError("Input data must contain a 'prompt' field")

            model_runner.register_model(alternative_model, alt_model_fn)
            print(f"Registered model '{alternative_model}' with ModelRunner")
        else:
            print("No models available to register")
            return

    # 6. Show the list of registered models
    print("\n6. Showing list of registered models")
    registered_models = model_runner.get_available_models()
    print(f"Number of registered models: {len(registered_models)}")
    for model in registered_models:
        print(f"  - {model}")

    # 7. Try to generate a completion with one of the models
    print("\n7. Generating a completion with a registered model")
    model_id = registered_models[0]
    prompt = "Explain in three sentences what consensus means in the context of AI."

    try:
        print(f"Generating completion with model '{model_id}'...")
        print(f"Prompt: '{prompt}'")

        response = await model_runner.generate_completion(
            model_id=model_id,
            prompt=prompt,
            max_tokens=100
        )

        print("\nGeneration result:")
        if response and "status" in response and response["status"] == "success":
            print(f"Completion: {response.get('completion', 'No completion returned')}")
        else:
            print(f"Error in generation: {response}")

    except Exception as e:
        print(f"Error generating completion: {e}")

if __name__ == "__main__":
    asyncio.run(main())
