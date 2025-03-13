#!/usr/bin/env python
from src.router.openrouter import OpenRouter
import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List, Optional, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_chat_completion(
    messages: Optional[List[Dict[str, str]]] = None,
    model: str = "google/gemini-1.5-pro",
    max_tokens: int = 256,
    temperature: float = 0.7,
    interactive: bool = False,
) -> Dict[str, Any]:
    """
    Test the chat completion endpoint with the given parameters.

    Args:
        messages: List of message dictionaries for chat completion
        model: The model to use for chat completion
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature parameter for generation
        interactive: Whether to run in interactive mode

    Returns:
        The chat completion response
    """
    try:
        api_key = os.environ.get("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPEN_ROUTER_API_KEY environment variable is not set")

        client = OpenRouter(api_key=api_key)

        if interactive:
            conversation = messages or []
            if not conversation:
                # Start with system message if empty
                conversation.append({"role": "system", "content": "You are a helpful assistant."})

            print(f"\nUsing model: {model}")
            print("Interactive chat mode (type 'exit' to quit)")
            print("Current conversation:")
            for msg in conversation:
                if msg["role"] != "system":
                    print(f"{msg['role'].upper()}: {msg['content']}")

            while True:
                user_input = input("\nYOU: ")
                if user_input.lower() == "exit":
                    break

                # Add user message to conversation
                conversation.append({"role": "user", "content": user_input})

                print("\nGenerating response...")
                response = await client.create_chat_completion(
                    model=model,
                    messages=conversation,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                if response.get("choices") and response["choices"][0].get("message"):
                    assistant_message = response["choices"][0]["message"]["content"]
                    print(f"\nASSISTANT: {assistant_message}")

                    # Add assistant response to conversation history
                    conversation.append({"role": "assistant", "content": assistant_message})
                else:
                    print("\n[ERROR] No valid response received")

            return {"status": "interactive_session_ended", "conversation": conversation}
        else:
            # Single chat completion mode
            if not messages:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, can you help me with a task?"}
                ]

            response = await client.create_chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            print("\n" + "=" * 80)
            print("CHAT COMPLETION RESPONSE:")
            print("=" * 80)
            if response.get("choices") and response["choices"][0].get("message"):
                print(response["choices"][0]["message"]["content"])
            else:
                print("No completion text found in response")
            print("=" * 80)

            return response

    except Exception as e:
        error_msg = f"Error during chat completion test: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        return {"error": error_msg}


def parse_messages(messages_json: str) -> List[Dict[str, str]]:
    """Parse JSON string to messages list"""
    try:
        return json.loads(messages_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format for messages: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Test OpenRouter chat completion endpoint")
    parser.add_argument(
        "--messages",
        type=str,
        help="JSON string of messages for chat completion"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-1.5-pro",
        help="Model to use for chat completion"
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

    messages = None
    if args.messages:
        messages = parse_messages(args.messages)

    asyncio.run(test_chat_completion(
        messages=messages,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        interactive=(args.mode == "interactive"),
    ))


if __name__ == "__main__":
    main()
