"""
Agent Engine for Ultra-Claw.

This module provides the core agent functionality, integrating
memory, LLM, and workflow capabilities into a unified agent.
"""

import asyncio
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from uuid import uuid4

from ultra_claw.core.models import (
    AgentConfig,
    AgentResponse,
    MemoryFilter,
    MemoryItem,
    MemoryLevel,
    Message,
    MessageRole,
    ModalityType,
    WorkflowStep,
    WorkflowState,
)
from ultra_claw.core.memory import MemoryService
from ultra_claw.core.session import SessionManager
from ultra_claw.core.workflow import WorkflowEngine
from ultra_claw.services.llm.base import LLMProvider
from ultra_claw.services.llm.mock_provider import MockProvider
from ultra_claw.services.tools import ToolManager
from ultra_claw.utils.logger import get_logger

logger = get_logger(__name__)


class UltraAgent:
    """
    The core agent class for Ultra-Claw.
    
    UltraAgent integrates memory, LLM, and workflow capabilities
to provide a comprehensive intelligent agent with long-term
    memory and multimodal support.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        memory_service: Optional[MemoryService] = None,
        llm_provider: Optional[LLMProvider] = None,
        session_manager: Optional[SessionManager] = None,
        workflow_engine: Optional[WorkflowEngine] = None,
    ):
        """
        Initialize the UltraAgent.
        
        Args:
            config: Agent configuration
            memory_service: Memory service instance
            llm_provider: LLM provider instance
            session_manager: Session manager instance
            workflow_engine: Workflow engine instance
        """
        self.config = config
        self.id = str(uuid4())
        
        # Initialize services
        self.memory = memory_service or MemoryService(config.memory)
        self.llm = llm_provider or self._create_llm_provider(config.llm)
        self.sessions = session_manager or SessionManager()
        self.workflows = workflow_engine or WorkflowEngine(config.workflow)
        self.tools = ToolManager()
        
        self._initialized = False
        logger.info(f"Created UltraAgent {self.id} with name '{config.name}'")
    
    def _create_llm_provider(self, llm_config) -> LLMProvider:
        """Create an LLM provider based on configuration."""
        provider_name = llm_config.provider
        config_dict = llm_config.model_dump()
        
        if provider_name == "openai":
            from ultra_claw.services.llm.openai_provider import OpenAIProvider
            return OpenAIProvider(config_dict)
        elif provider_name == "anthropic":
            from ultra_claw.services.llm.anthropic_provider import AnthropicProvider
            return AnthropicProvider(config_dict)
        elif provider_name == "mock":
            return MockProvider(config_dict)
        else:
            logger.warning(f"Unknown provider '{provider_name}', using mock")
            return MockProvider(config_dict)
    
    async def initialize(self) -> None:
        """Initialize the agent and all its services."""
        if self._initialized:
            return
        
        logger.info(f"Initializing UltraAgent {self.id}")
        
        # Initialize services
        await self.memory.initialize()
        await self.sessions.initialize()
        
        self._initialized = True
        logger.info(f"UltraAgent {self.id} initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the agent and all its services."""
        logger.info(f"Shutting down UltraAgent {self.id}")
        
        await self.memory.shutdown()
        await self.sessions.shutdown()
        
        self._initialized = False
    
    async def chat(
        self,
        messages: List[Message],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        use_memory: bool = True,
        memory_categories: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[AgentResponse, None]:
        """
        Have a conversation with the agent.
        
        Args:
            messages: List of messages
            session_id: Optional session ID for context
            user_id: Optional user ID for personalization
            use_memory: Whether to use memory retrieval
            memory_categories: Categories to filter memories
            **kwargs: Additional parameters for LLM
            
        Yields:
            AgentResponse chunks
        """
        await self.initialize()
        
        # Get or create session
        if session_id:
            session = await self.sessions.get_session(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found, creating new")
                session = await self.sessions.create_session(
                    user_id=user_id or "anonymous",
                    agent_id=self.id
                )
                session_id = session.id
        else:
            session = await self.sessions.create_session(
                user_id=user_id or "anonymous",
                agent_id=self.id
            )
            session_id = session.id
        
        # Retrieve relevant memories
        memories = []
        if use_memory and user_id:
            # Get the last user message for retrieval
            last_user_msg = None
            for msg in reversed(messages):
                if msg.role == MessageRole.USER:
                    last_user_msg = msg.content
                    break
            
            if last_user_msg:
                filters = MemoryFilter(
                    user_id=user_id,
                    agent_id=self.id,
                    categories=memory_categories
                )
                memories = await self.memory.retrieve(
                    query=last_user_msg,
                    filters=filters,
                    top_k=self.config.memory.max_context_memories
                )
        
        # Build context with memories
        context_messages = list(messages)
        
        if memories:
            # Add memory context as a system message
            memory_context = "Relevant information from memory:\n"
            for i, memory in enumerate(memories, 1):
                memory_context += f"{i}. {memory.content}\n"
            
            # Insert memory context at the beginning
            context_messages.insert(0, Message(
                role=MessageRole.SYSTEM,
                content=memory_context
            ))
        
        # Generate response
        full_response = ""
        memories_used = [m.id for m in memories]
        
        async for chunk in self.llm.chat(context_messages, **kwargs):
            full_response += chunk.content
            yield AgentResponse(
                content=chunk.content,
                role=MessageRole.ASSISTANT,
                modality=ModalityType.TEXT,
                metadata=chunk.metadata,
                memories_used=memories_used
            )
        
        # Store the conversation in session
        for msg in messages:
            await self.sessions.add_message(
                session_id=session_id,
                role=msg.role,
                content=msg.content,
                modality=msg.modality.value
            )
        
        await self.sessions.add_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=full_response
        )
        
        # Memorize the conversation if enabled
        if use_memory and user_id:
            await self.memorize(
                content=f"User: {messages[-1].content}\nAssistant: {full_response}",
                user_id=user_id,
                categories=["conversation"] + (memory_categories or []),
                metadata={"session_id": session_id}
            )
    
    async def memorize(
        self,
        content: str,
        user_id: str,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        modality: Union[str, ModalityType] = ModalityType.TEXT,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> MemoryItem:
        """
        Store a memory item.
        
        Args:
            content: The content to memorize
            user_id: The user ID
            categories: Categories for organization
            tags: Tags for the memory
            modality: Content modality
            importance: Importance score (0-1)
            metadata: Additional metadata
            expires_at: Expiration time
            
        Returns:
            The created memory item
        """
        await self.initialize()
        
        if isinstance(modality, str):
            modality = ModalityType(modality)
        
        # Generate embedding
        try:
            embeddings = await self.llm.embed([content])
            embedding = embeddings[0] if embeddings else None
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            embedding = None
        
        item = MemoryItem(
            user_id=user_id,
            agent_id=self.id,
            content=content,
            modality=modality,
            embedding=embedding,
            categories=categories or [],
            tags=tags or [],
            source="agent",
            metadata=metadata or {},
            expires_at=expires_at,
            importance_score=importance,
        )
        
        await self.memory.store(item)
        logger.debug(f"Memorized item {item.id} for user {user_id}")
        
        return item
    
    async def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
        top_k: int = 10,
        **filters
    ) -> List[MemoryItem]:
        """
        Retrieve memories matching a query.
        
        Args:
            query: The search query
            user_id: Optional user ID filter
            categories: Optional category filter
            top_k: Maximum number of results
            **filters: Additional filters
            
        Returns:
            List of matching memory items
        """
        await self.initialize()
        
        memory_filter = MemoryFilter(
            user_id=user_id,
            agent_id=self.id,
            categories=categories,
            **filters
        )
        
        return await self.memory.retrieve(
            query=query,
            filters=memory_filter,
            top_k=top_k
        )
    
    async def forget(self, memory_id: str) -> bool:
        """
        Delete a memory item.
        
        Args:
            memory_id: The memory item ID
            
        Returns:
            True if deleted, False otherwise
        """
        await self.initialize()
        return await self.memory.delete(memory_id)
    
    async def execute_workflow(
        self,
        steps: List[WorkflowStep],
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """
        Execute a workflow.
        
        Args:
            steps: List of workflow steps
            context: Initial context
            
        Returns:
            The final workflow state
        """
        await self.initialize()
        
        workflow_id = str(uuid4())
        self.workflows.register_workflow(workflow_id, steps)
        
        return await self.workflows.execute(
            workflow_id=workflow_id,
            context=context or {}
        )
    
    async def get_session_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get the message history for a session.
        
        Args:
            session_id: The session ID
            limit: Maximum number of messages
            
        Returns:
            List of messages
        """
        return await self.sessions.get_messages(session_id, limit=limit)
    
    async def consolidate_memories(
        self,
        user_id: str,
        strategy: str = "auto"
    ) -> bool:
        """
        Consolidate memories for a user.
        
        Args:
            user_id: The user ID
            strategy: Consolidation strategy
            
        Returns:
            True if successful
        """
        await self.initialize()
        return await self.memory.consolidate(user_id, self.id, strategy)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        memory_stats = await self.memory.get_stats()
        session_stats = await self.sessions.get_stats()
        
        return {
            "agent_id": self.id,
            "agent_name": self.config.name,
            "memory": memory_stats,
            "sessions": session_stats,
            "workflows": {
                "registered": len(self.workflows.list_workflows()),
            },
            "tools": {
                "available": len(self.tools.list_tools()),
            },
        }
