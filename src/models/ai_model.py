from abc import ABC, abstractmethod
from typing import Any, Dict


class AIModel(ABC):
    """
    Abstract base class for AI models used in the consensus system.
    All AI models should inherit from this class and implement the required methods.
    """

    def __init__(self, name: str):
        """
        Initialize an AI model instance.

        Args:
            name: The name of the AI model.
        """
        self.name = name

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response based on the given prompt.

        Args:
            prompt: The input text prompt to generate a response for.
            **kwargs: Additional arguments to customize generation.

        Returns:
            A dictionary containing the generated response and metadata.
        """
