"""
Mock LLM provider for Ultra-Claw.

This module provides a mock LLM provider for testing purposes.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List

from ultra_claw.core.models import Message
from ultra_claw.services.llm.base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    """
    Mock LLM provider for testing.
    
    Returns predefined responses for testing without making API calls.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.responses = config.get("responses", [])
        self.response_index = 0
    
    @property
    def name(self) -> str:
        return "mock"
    
    @property
    def supports_streaming(self) -> bool:
        return True
    
    @property
    def supports_vision(self) -> bool:
        return True
    
    async def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncGenerator[LLMResponse, None]:
        """
        Generate a mock chat completion.
        
        Args:
            messages: List of messages
            **kwargs: Additional parameters
            
        Yields:
            LLMResponse chunks
        """
        # Get response
        if self.responses:
            response = self.responses[self.response_index % len(self.responses)]
            self.response_index += 1
        else:
            # Generate a simple response based on the last message
            if messages:
                last_msg = messages[-1].content
                response = f"Mock response to: {last_msg[:50]}..."
            else:
                response = "Mock response"
        
        # Stream the response word by word
        words = response.split()
        for word in words:
            yield LLMResponse(
                content=word + " ",
                model=self.model,
                metadata={}
            )
            await asyncio.sleep(0.01)  # Small delay for realism
    
    async def embed(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Generate mock embeddings.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional parameters
            
        Returns:
            List of embedding vectors
        """
        import hashlib
        
        embeddings = []
        for text in texts:
            # Generate deterministic embedding from text hash
            hash_bytes = hashlib.sha256(text.encode()).digest()
            embedding = [float(b) / 255.0 for b in hash_bytes[:128]]
            embeddings.append(embedding)
        
        return embeddings
