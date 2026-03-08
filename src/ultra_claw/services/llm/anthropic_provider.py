"""
Anthropic LLM provider for Ultra-Claw.

This module provides integration with Anthropic's Claude models.
"""

import os
from typing import Any, AsyncGenerator, Dict, List

import anthropic

from ultra_claw.core.models import Message
from ultra_claw.services.llm.base import LLMProvider, LLMResponse
from ultra_claw.utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """
    LLM provider for Anthropic Claude models.
    
    Supports Claude 3 and other Anthropic models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    @property
    def supports_streaming(self) -> bool:
        return True
    
    @property
    def supports_vision(self) -> bool:
        return "claude-3" in self.model
    
    async def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncGenerator[LLMResponse, None]:
        """
        Generate a chat completion using Anthropic.
        
        Args:
            messages: List of messages
            **kwargs: Additional parameters
            
        Yields:
            LLMResponse chunks
        """
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            system_message = None
            
            for msg in messages:
                if msg.role.value == "system":
                    system_message = msg.content
                else:
                    anthropic_messages.append({
                        "role": msg.role.value,
                        "content": msg.content
                    })
            
            # Build parameters
            params = {
                "model": kwargs.get("model", self.model),
                "messages": anthropic_messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": True,
            }
            
            if system_message:
                params["system"] = system_message
            
            if "temperature" in kwargs:
                params["temperature"] = kwargs["temperature"]
            if "top_p" in kwargs:
                params["top_p"] = kwargs["top_p"]
            
            # Make request
            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield LLMResponse(
                        content=text,
                        model=params["model"],
                        metadata={}
                    )
        
        except Exception as e:
            logger.error(f"Anthropic chat error: {e}")
            raise
    
    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings using Anthropic.
        
        Note: Anthropic doesn't provide embeddings API directly.
        This is a placeholder that would use a fallback provider.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional parameters
            
        Returns:
            List of embedding vectors
        """
        # Anthropic doesn't have an embeddings API
        # In production, you'd use OpenAI or another provider for embeddings
        logger.warning("Anthropic doesn't support embeddings, using fallback")
        
        # Return simple hash-based embeddings as fallback
        import hashlib
        embeddings = []
        for text in texts:
            hash_bytes = hashlib.sha256(text.encode()).digest()
            embedding = [float(b) / 255.0 for b in hash_bytes[:128]]
            embeddings.append(embedding)
        return embeddings
