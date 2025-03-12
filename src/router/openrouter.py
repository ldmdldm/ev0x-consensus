import time
import os
import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional, Union
from aiohttp import ClientSession, ClientTimeout
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# OpenRouter API constants
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

# Rate limiting parameters (5 requests per second)
RATE_LIMIT = 5
PERIOD = 1  # seconds

class OpenRouterException(Exception):
    """Custom exception for OpenRouter API errors"""
    def __init__(self, status_code: int, message: str, response_data: Optional[Dict] = None):
        self.status_code = status_code
        self.message = message
        self.response_data = response_data
        super().__init__(f"OpenRouter API Error (Status: {status_code}): {message}")


class OpenRouterClient:
    """
    OpenRouter API client that provides access to 300+ models through a unified interface.
    Includes proper error handling, rate limiting, and async support.
    """
    def __init__(self, api_key: Optional[str] = None, timeout: int = 120):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If not provided, it's read from environment variables.
            timeout: API request timeout in seconds
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPEN_ROUTER_API_KEY environment variable or pass it directly.")
        
        self.timeout = ClientTimeout(total=timeout)
        self.models_cache = None
        self.models_cache_expiry = None
        self.models_cache_duration = timedelta(hours=24)
        
        # Create a semaphore for rate limiting control
        self.semaphore = asyncio.Semaphore(RATE_LIMIT)  # Limit concurrent requests to RATE_LIMIT
        # Track request timestamps for rate limiting
        self.request_timestamps = deque(maxlen=RATE_LIMIT*2)
    
    async def _make_request(self, endpoint: str, data: Dict, method: str = "POST") -> Dict:
        """
        Make a rate-limited request to the OpenRouter API.
        
        Args:
            endpoint: API endpoint to call
            data: Request payload
            method: HTTP method to use
            
        Returns:
            Response data as dictionary
        """
        # Implement rate limiting with the semaphore
        async with self.semaphore:
            # Check if we need to wait to respect the rate limit
            now = time.time()
            if len(self.request_timestamps) >= RATE_LIMIT:
                # Calculate how long to wait if we've reached the rate limit
                oldest_timestamp = self.request_timestamps[0]
                time_since_oldest = now - oldest_timestamp
                if time_since_oldest < PERIOD:
                    # Wait until a full period has passed since the oldest request
                    wait_time = PERIOD - time_since_oldest
                    logger.debug(f"Rate limit hit, waiting for {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                    now = time.time()  # Update current time after waiting
            
            # Record this request's timestamp
            self.request_timestamps.append(now)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://flare-ai-consensus.hackathon",  # Identify your application
                "X-Title": "Flare AI Consensus"  # Application name
            }
            
            url = f"{OPENROUTER_API_URL}/{endpoint}"
            
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    if method == "POST":
                        async with session.post(url, headers=headers, json=data) as response:
                            response_data = await response.json()
                            if response.status >= 400:
                                raise OpenRouterException(
                                    status_code=response.status,
                                    message=response_data.get('error', {}).get('message', 'Unknown error'),
                                    response_data=response_data
                                )
                            return response_data
                    else:  # GET
                        async with session.get(url, headers=headers) as response:
                            response_data = await response.json()
                            if response.status >= 400:
                                raise OpenRouterException(
                                    status_code=response.status,
                                    message=response_data.get('error', {}).get('message', 'Unknown error'),
                                    response_data=response_data
                                )
                            return response_data
                            
            except aiohttp.ClientError as e:
                logger.error(f"Network error during OpenRouter API request: {str(e)}")
                raise OpenRouterException(status_code=500, message=f"Network error: {str(e)}")
            except asyncio.TimeoutError:
                logger.error(f"Request to OpenRouter API timed out after {self.timeout.total} seconds")
                raise OpenRouterException(status_code=408, message=f"Request timed out after {self.timeout.total} seconds")

    async def get_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get a list of all available models from OpenRouter.
        Results are cached for 24 hours to reduce API calls.
        
        Args:
            force_refresh: Force refresh of cached models list
            
        Returns:
            List of available models with their details
        """
        # Return cached models if available and not expired
        if not force_refresh and self.models_cache and self.models_cache_expiry and self.models_cache_expiry > datetime.now():
            return self.models_cache
        
        try:
            result = await self._make_request("models", {}, method="GET")
            
            # Format the models data for easier consumption
            models = result.get('data', [])
            
            # Cache the models list
            self.models_cache = models
            self.models_cache_expiry = datetime.now() + self.models_cache_duration
            
            logger.info(f"Retrieved {len(models)} models from OpenRouter")
            return models
            
        except Exception as e:
            logger.error(f"Error retrieving models from OpenRouter: {str(e)}")
            # If we have a cache, return it even if expired
            if self.models_cache:
                logger.warning("Returning expired models cache due to API error")
                return self.models_cache
            raise

    async def completion(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a text completion using the specified model.
        
        Args:
            prompt: Text prompt to complete
            model: Model ID to use for completion
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (0-2)
            presence_penalty: Presence penalty (0-2)
            stop: Stop sequences
            
        Returns:
            Completion response
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }
        
        if stop:
            payload["stop"] = stop
            
        try:
            return await self._make_request("completions", payload)
        except OpenRouterException as e:
            logger.error(f"Error in completion request: {e.message}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in completion request: {str(e)}")
            raise OpenRouterException(
                status_code=500,
                message=f"Unexpected error: {str(e)}"
            )

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], aiohttp.ClientResponse]:
        """
        Generate a chat completion using the specified model.
        
        Args:
            messages: List of messages in the conversation
            model: Model ID to use for chat completion
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (0-2)
            presence_penalty: Presence penalty (0-2)
            stop: Stop sequences
            stream: Whether to stream the response
            
        Returns:
            Chat completion response or stream
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream
        }
        
        if stop:
            payload["stop"] = stop
            
        # Streaming responses require special handling
        if stream:
            return await self._stream_chat_completion(payload)
            
        try:
            return await self._make_request("chat/completions", payload)
        except OpenRouterException as e:
            logger.error(f"Error in chat completion request: {e.message}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in chat completion request: {str(e)}")
            raise OpenRouterException(
                status_code=500,
                message=f"Unexpected error: {str(e)}"
            )

    async def _stream_chat_completion(self, payload: Dict) -> aiohttp.ClientResponse:
        """
        Stream a chat completion response.
        
        Args:
            payload: Request payload
            
        Returns:
            Streaming response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://flare-ai-consensus.hackathon",
            "X-Title": "Flare AI Consensus"
        }
        
        url = f"{OPENROUTER_API_URL}/chat/completions"
        
        try:
            session = aiohttp.ClientSession(timeout=self.timeout)
            response = await session.post(url, headers=headers, json=payload)
            
            if response.status >= 400:
                response_data = await response.json()
                raise OpenRouterException(
                    status_code=response.status,
                    message=response_data.get('error', {}).get('message', 'Unknown error'),
                    response_data=response_data
                )
                
            # Return the response object for caller to process the stream
            return response
            
        except Exception as e:
            if isinstance(e, OpenRouterException):
                raise
            else:
                logger.error(f"Error in streaming chat completion: {str(e)}")
                raise OpenRouterException(
                    status_code=500,
                    message=f"Streaming error: {str(e)}"
                )

    async def process_stream(self, response: aiohttp.ClientResponse):
        """
        Process a streaming response from the OpenRouter API.
        
        Args:
            response: Streaming response from _stream_chat_completion
            
        Yields:
            Parsed chunks from the stream
        """
        async for line in response.content:
            line = line.strip()
            if line:
                if line.startswith(b"data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == b"[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse stream chunk: {data}")

    async def generate_with_backup(
        self,
        messages: List[Dict[str, str]],
        primary_model: str,
        backup_models: List[str],
        is_chat: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using a primary model with automatic fallback to backup models.
        
        Args:
            messages: List of messages (for chat) or single prompt string (for completion)
            primary_model: Primary model ID to use
            backup_models: List of backup model IDs in priority order
            is_chat: Whether this is a chat completion (True) or text completion (False)
            **kwargs: Additional parameters for the completion/chat completion
            
        Returns:
            Response from the first successful model
        """
        # Combine primary and backup models into one list for iteration
        models = [primary_model] + backup_models
        last_error = None
        
        for model in models:
            try:
                logger.info(f"Attempting generation with model: {model}")
                if is_chat:
                    result = await self.chat_completion(messages=messages, model=model, **kwargs)
                else:
                    # For completion, extract the prompt from messages if needed
                    prompt = messages if isinstance(messages, str) else messages[-1]["content"]
                    result = await self.completion(prompt=prompt, model=model, **kwargs)
                return result
            except OpenRouterException as e:
                logger.warning(f"Error with model {model}: {e.message}. Trying next model...")
                last_error = e
                continue
                
        # If we've exhausted all models, raise the last error
        if last_error:
            raise last_error
        else:
            raise OpenRouterException(
                status_code=500,
                message="All models failed without specific errors"
            )

    async def generate_with_multiple_models(
        self,
        messages: List[Dict[str, str]],
        models: List[str],
        is_chat: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate text using multiple models in parallel.
        
        Args:
            messages: List of messages (for chat) or single prompt string (for completion)
            models: List of model IDs to use
            is_chat: Whether this is a chat completion (True) or text completion (False)
            **kwargs: Additional parameters for the completion/chat completion
            
        Returns:
            List of responses from all models that succeeded
        """
        tasks = []
        
        for model in models:
            if is_chat:
                task = asyncio.create_task(
                    self.chat_completion(messages=messages, model=model, **kwargs)
                )
            else:
                # For completion, extract the prompt from messages if needed
                prompt = messages if isinstance(messages, str) else messages[-1]["content"]
                task = asyncio.create_task(
                    self.completion(prompt=prompt, model=model, **kwargs)
                )
            tasks.append((model, task))
            
        results = []
        
        for model, task in tasks:
            try:
                result = await task
                results.append({
                    "model": model,
                    "result": result
                })
            except Exception as e:
                logger.warning(f"Error with model {model}: {str(e)}")
                
        return results

    async def check_model_availability(self, model_id: str) -> bool:
        """
        Check if a specified model is available on OpenRouter.
        
        Args:
            model_id: The ID of the model to check
            
        Returns:
            Boolean indicating whether the model is available
        """
        try:
            # Get the list of available models
            models = await self.get_models()
            
            # Log all available model IDs for debugging purposes
            all_model_ids = [model.get('id') for model in models]
            logger.info(f"Available models from OpenRouter API: {all_model_ids}")
            
            # Also log the model details to help with debugging format differences
            logger.info(f"Looking for model: '{model_id}'")
            
            # Check if the specified model is in the list
            for model in models:
                model_id_from_api = model.get('id')
                if model_id_from_api == model_id:
                    logger.info(f"Model {model_id} is available")
                    return True
            
            logger.warning(f"Model {model_id} is not available on OpenRouter")
            logger.warning(f"Please ensure the model ID format matches exactly what's returned by the API")
            return False
        except Exception as e:
            logger.error(f"Error checking model availability for {model_id}: {str(e)}")
            # If there's an error, we can't confirm availability, so return False
            return False

    async def get_credits(self) -> Dict[str, Any]:
        """
        Get current credit balance and usage information.
        
        Returns:
            Credit balance information
        """
        try:
            endpoint = "stats/credits"
            return await self._make_request(endpoint, {}, method="GET")
        except Exception as e:
            logger.error(f"Failed to get credits: {str(e)}")
            raise OpenRouterException(status_code=500, message=f"Error fetching credit information: {str(e)}")

    async def generate_async(self, model: str, prompt: str, **kwargs) -> Optional[Dict]:
        """Generate completion asynchronously."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://localhost",  # Required by OpenRouter
                    "X-Title": "Consensus Flow Test",     # Required by OpenRouter
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],  # OpenRouter expects messages format
                    **kwargs
                }
                
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",  # Use chat completions endpoint
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0]:
                            return {"text": result["choices"][0]["message"]["content"]}
                        else:
                            logger.warning(f"Unexpected API response format: {result}")
                            return {"text": str(result), "error": "Unexpected response format"}
                    else:
                        logger.error(f"API request failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error in generate_async: {e}")
            return None
            
    async def chat_async(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Optional[Dict]:
        """Generate chat completion asynchronously."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://localhost",  # Required by OpenRouter
                    "X-Title": "Consensus Flow Test",     # Required by OpenRouter
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": model,
                    "messages": messages,
                    **kwargs
                }
                
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0]:
                            return {"text": result["choices"][0]["message"]["content"]}
                        else:
                            logger.warning(f"Unexpected API response format: {result}")
                            return {"text": str(result), "error": "Unexpected response format"}
                    else:
                        logger.error(f"API request failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error in chat_async: {e}")
            return None
