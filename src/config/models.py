"""Configuration for AI models used in the ev0x project."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


class ModelType(Enum):
    """Types of AI models supported by ev0x."""
    LLM = "llm"
    EMBEDDING = "embedding"
    AQA = "attributed_qa"
    CUSTOM = "custom"


class ModelProvider(Enum):
    """AI model providers supported by ev0x."""
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    name: str
    model_type: ModelType
    provider: ModelProvider
    version: str
    endpoint: str
    api_key_env_var: str
    parameters: Optional[Dict[str, Any]] = None
    capabilities: Optional[List[str]] = None
    enabled: bool = True

    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "provider": self.provider.value,
            "version": self.version,
            "endpoint": self.endpoint,
            "capabilities": self.capabilities,
            "enabled": self.enabled
        }


# Default model configurations
DEFAULT_MODELS = [
    ModelConfig(
        name="gpt-4",
        model_type=ModelType.LLM,
        provider=ModelProvider.OPENROUTER,
        version="gpt-4",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        api_key_env_var="OPEN_ROUTER_API_KEY",
        capabilities=["text_generation", "chat", "reasoning", "code_generation"],
        parameters={
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.95,
            "model": "gpt-4"
        }
    ),
    ModelConfig(
        name="claude-3-opus",
        model_type=ModelType.LLM,
        provider=ModelProvider.OPENROUTER,
        version="claude-3-opus",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        api_key_env_var="OPEN_ROUTER_API_KEY",
        capabilities=["text_generation", "chat", "reasoning", "code_generation"],
        parameters={
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.95,
            "model": "anthropic/claude-3-opus"
        }
    ),
    ModelConfig(
        name="claude-3-sonnet",
        model_type=ModelType.LLM,
        provider=ModelProvider.OPENROUTER,
        version="claude-3-sonnet",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        api_key_env_var="OPEN_ROUTER_API_KEY",
        capabilities=["text_generation", "chat", "reasoning"],
        parameters={
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.95,
            "model": "anthropic/claude-3-sonnet"
        }
    ),
    ModelConfig(
        name="llama-3-70b",
        model_type=ModelType.LLM,
        provider=ModelProvider.OPENROUTER,
        version="llama-3-70b",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        api_key_env_var="OPEN_ROUTER_API_KEY",
        capabilities=["text_generation", "chat", "reasoning"],
        parameters={
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.95,
            "model": "meta-llama/llama-3-70b-instruct"
        }
    ),
    ModelConfig(
        name="gemini-pro",
        model_type=ModelType.LLM,
        provider=ModelProvider.OPENROUTER,
        version="gemini-pro",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        api_key_env_var="OPEN_ROUTER_API_KEY",
        capabilities=["text_generation", "chat", "reasoning"],
        parameters={
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.95,
            "model": "google/gemini-pro"
        }
    )
]

# Alias for DEFAULT_MODELS to maintain compatibility with imports in main.py
AVAILABLE_MODELS = DEFAULT_MODELS


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a specific model by name."""
    for model in DEFAULT_MODELS:
        if model.name == model_name:
            return model
    return None


def get_enabled_models() -> List[ModelConfig]:
    """Get all enabled models."""
    return [model for model in DEFAULT_MODELS if model.enabled]


def get_models_by_capability(capability: str) -> List[ModelConfig]:
    """Get all models that have a specific capability."""
    return [model for model in DEFAULT_MODELS
            if model.enabled and capability in model.capabilities]
