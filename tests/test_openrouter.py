import os
import pytest
from src.router.openrouter import OpenRouterClient


@pytest.mark.asyncio
async def test_openrouter_basic():
    """Test basic OpenRouter functionality."""
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        pytest.skip("OpenRouter API key not available in environment")

    try:
        client = OpenRouterClient(api_key)
        prompt = "What is 2+2?"
        model = "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"

        response = await client.generate_async(
            model=model,
            prompt=prompt,
            max_tokens=100
        )

        assert response is not None, "Response should not be None"
        assert isinstance(response, dict), "Response should be a dictionary"
        assert "text" in response, "Response should contain 'text' field"

        text = response["text"]
        assert isinstance(text, str), "Response text should be a string"
        assert len(text) > 0, "Response text should not be empty"
        assert "4" in text, "Response should contain the answer '4'"

        print(f"OpenRouter response: {text}")

    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")
