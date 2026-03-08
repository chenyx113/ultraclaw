"""
OpenClaw compatibility layer for Ultra-Claw.

This module provides compatibility with OpenClaw's API and data formats.
"""

from typing import Any, Dict, List, Optional

from ultra_claw.core.agent import UltraAgent
from ultra_claw.core.models import AgentConfig, Message, MessageRole
from ultra_claw.utils.logger import get_logger

logger = get_logger(__name__)


class OpenClawCompatibility:
    """
    Compatibility layer for OpenClaw.
    
    Provides adapters to use Ultra-Claw with OpenClaw-formatted
    requests and responses.
    """
    
    def __init__(self, ultra_agent: UltraAgent):
        self.agent = ultra_agent
    
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        OpenClaw-compatible chat method.
        
        Args:
            messages: OpenClaw-formatted messages
            **kwargs: Additional parameters
            
        Returns:
            OpenClaw-formatted response
        """
        # Convert OpenClaw messages to Ultra-Claw format
        ultra_messages = self._convert_messages(messages)
        
        # Get session and user IDs from kwargs
        session_id = kwargs.get("session_id")
        user_id = kwargs.get("user_id")
        
        # Call Ultra-Claw
        response_chunks = []
        async for response in self.agent.chat(
            messages=ultra_messages,
            session_id=session_id,
            user_id=user_id
        ):
            response_chunks.append(response.content)
        
        full_response = "".join(response_chunks)
        
        # Convert to OpenClaw format
        return {
            "message": {
                "role": "assistant",
                "content": full_response,
            },
            "metadata": {
                "model": self.agent.config.llm.model,
                "memories_used": [],
            }
        }
    
    async def memorize(
        self,
        content: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        OpenClaw-compatible memorize method.
        
        Args:
            content: Content to memorize
            **kwargs: Additional parameters
            
        Returns:
            OpenClaw-formatted response
        """
        user_id = kwargs.get("user_id", "default")
        categories = kwargs.get("categories", [])
        
        memory_item = await self.agent.memorize(
            content=content,
            user_id=user_id,
            categories=categories
        )
        
        return {
            "id": memory_item.id,
            "status": "stored",
        }
    
    async def retrieve(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        OpenClaw-compatible retrieve method.
        
        Args:
            query: Search query
            **kwargs: Additional parameters
            
        Returns:
            OpenClaw-formatted response
        """
        user_id = kwargs.get("user_id")
        categories = kwargs.get("categories")
        top_k = kwargs.get("top_k", 10)
        
        memories = await self.agent.retrieve(
            query=query,
            user_id=user_id,
            categories=categories,
            top_k=top_k
        )
        
        return {
            "results": [
                {
                    "id": m.id,
                    "content": m.content,
                    "categories": m.categories,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in memories
            ],
            "total": len(memories),
        }
    
    def _convert_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Message]:
        """
        Convert OpenClaw messages to Ultra-Claw format.
        
        Args:
            messages: OpenClaw-formatted messages
            
        Returns:
            Ultra-Claw Message objects
        """
        ultra_messages = []
        
        for msg in messages:
            role_str = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map OpenClaw roles to Ultra-Claw roles
            role_map = {
                "user": MessageRole.USER,
                "assistant": MessageRole.ASSISTANT,
                "system": MessageRole.SYSTEM,
                "tool": MessageRole.TOOL,
            }
            
            role = role_map.get(role_str, MessageRole.USER)
            
            ultra_messages.append(Message(
                role=role,
                content=content
            ))
        
        return ultra_messages
    
    @staticmethod
    def convert_config(openclaw_config: Dict[str, Any]) -> AgentConfig:
        """
        Convert OpenClaw config to Ultra-Claw config.
        
        Args:
            openclaw_config: OpenClaw configuration
            
        Returns:
            Ultra-Claw AgentConfig
        """
        return AgentConfig(
            name=openclaw_config.get("name", "Ultra-Claw Agent"),
            llm={
                "provider": openclaw_config.get("llm", {}).get("provider", "openai"),
                "model": openclaw_config.get("llm", {}).get("model", "gpt-4"),
                "api_key": openclaw_config.get("llm", {}).get("api_key"),
                "temperature": openclaw_config.get("llm", {}).get("temperature", 0.7),
            },
            memory={
                "backend": openclaw_config.get("memory", {}).get("backend", "sqlite"),
                "database_url": openclaw_config.get("memory", {}).get("database_url", "sqlite:///memory.db"),
            },
        )
