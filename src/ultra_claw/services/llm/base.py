"""
Base LLM provider interface for Ultra-Claw.

This module defines the abstract interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field

from ultra_claw.core.models import Message


class LLMResponse(BaseModel):
    """Response from an LLM provider."""
    
    content: str
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM providers (OpenAI, Anthropic, etc.) must implement this interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2000)
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncGenerator[LLMResponse, None]:
        """
        Generate a chat completion.
        
        Args:
            messages: List of messages in the conversation
            **kwargs: Additional provider-specific parameters
            
        Yields:
            LLMResponse chunks
        """
        pass
    
    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether the provider supports streaming responses."""
        pass
    
    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether the provider supports vision/multimodal inputs."""
        pass
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert internal Message objects to provider format.
        
        Args:
            messages: List of Message objects
            
        Returns:
            List of message dictionaries in provider format
        """
        return [
            {
                "role": msg.role.value,
                "content": msg.content
            }
            for msg in messages
        ]
