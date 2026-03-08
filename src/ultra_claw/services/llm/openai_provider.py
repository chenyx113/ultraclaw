"""
OpenAI LLM provider for Ultra-Claw.

This module provides integration with OpenAI's GPT models.
"""

import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import openai

from ultra_claw.core.models import Message
from ultra_claw.services.llm.base import LLMProvider, LLMResponse
from ultra_claw.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    LLM provider for OpenAI models.
    
    Supports GPT-4, GPT-3.5, and other OpenAI models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = config.get("base_url")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    @property
    def name(self) -> str:
        return "openai"
    
    @property
    def supports_streaming(self) -> bool:
        return True
    
    @property
    def supports_vision(self) -> bool:
        return "vision" in self.model or "gpt-4" in self.model
    
    async def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncGenerator[LLMResponse, None]:
        """
        Generate a chat completion using OpenAI.
        
        Args:
            messages: List of messages
            **kwargs: Additional parameters
            
        Yields:
            LLMResponse chunks
        """
        try:
            # Convert messages
            openai_messages = self._convert_messages(messages)
            
            # Build parameters
            params = {
                "model": kwargs.get("model", self.model),
                "messages": openai_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": True,
            }
            
            # Add optional parameters
            if "top_p" in kwargs:
                params["top_p"] = kwargs["top_p"]
            if "frequency_penalty" in kwargs:
                params["frequency_penalty"] = kwargs["frequency_penalty"]
            if "presence_penalty" in kwargs:
                params["presence_penalty"] = kwargs["presence_penalty"]
            
            # Make request
            stream = await self.client.chat.completions.create(**params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield LLMResponse(
                        content=chunk.choices[0].delta.content,
                        model=params["model"],
                        metadata={
                            "finish_reason": chunk.choices[0].finish_reason,
                        }
                    )
        
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise
    
    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings using OpenAI.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional parameters
            
        Returns:
            List of embedding vectors
        """
        try:
            model = kwargs.get("embedding_model", "text-embedding-ada-002")
            
            response = await self.client.embeddings.create(
                model=model,
                input=texts
            )
            
            return [item.embedding for item in response.data]
        
        except Exception as e:
            logger.error(f"OpenAI embed error: {e}")
            raise
