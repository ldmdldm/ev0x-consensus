#!/usr/bin/env python
import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List, Optional, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.router.openrouter import OpenRouter


async def test_completion(
    prompt: str,
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    interactive: bool = False,
) -> Dict[str, Any]:
    """
    Test the completion endpoint with the given parameters.
    
    Args:
        prompt: The prompt to complete
        model: The model to use for completion
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature parameter for generation
        interactive: Whether to run in interactive mode
    
    Returns:
        The completion response
    """
    try:
        api_key = os.environ.get("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPEN_ROUTER_API_KEY environment variable is not set")
        
        client = OpenRouter(api_key=api_key)
        
        if interactive:
            print(f"\nUsing model: {model}\n")
            while True:
                if not prompt:
                    prompt = input("\nEnter your prompt (type 'exit' to quit): ")
                    if prompt.lower() == "exit":
                        break
                
                print("\nGenerating completion...")
                response = await client.create_completion(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                
                print("\n" + "="*80)
                print("COMPLETION RESPONSE:")
                print("="*80)
                if response.get("choices"):
                    print(response["choices"][0]["text"])
                else:
                    print("No completion text found in response")
                print("="*80)
                
                # Reset prompt for next iteration
                prompt = ""
                
            return {"status": "interactive_session_ended"}
        else:
            # Single completion mode
            response = await client.create_completion(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            print("\n" + "="*80)
            print("COMPLETION RESPONSE:")
            print("="*80)
            if response.get("choices"):
                print(response["choices"][0]["text"])
            else:
                print("No completion text found in response")
            print("="*80)
            
            return response
    
    except Exception as e:
        error_msg = f"Error during completion test: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        return {"error": error_msg}


def main():
    parser = argparse.ArgumentParser(description="Test OpenRouter completion endpoint")
    parser.add_argument("--prompt", type=str, help="Prompt for completion")
    parser.add_argument(
        "--model", 
        type=str, 
        default="google/gemini-1.5-pro", 
        help="Model to use for completion"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=256, 
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="Temperature for generation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "interactive"],
        default="default",
        help="Mode to run the test in"
    )
    
    args = parser.parse_args()
    
    if args.mode == "default" and not args.prompt:
        parser.error("--prompt is required when not in interactive mode")
    
    asyncio.run(test_completion(
        prompt=args.prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        interactive=(args.mode == "interactive"),
    ))


if __name__ == "__main__":
    main()

