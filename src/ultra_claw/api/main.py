"""
FastAPI application for Ultra-Claw.

This module provides the RESTful API for interacting with Ultra-Claw.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ultra_claw.core.agent import UltraAgent
from ultra_claw.core.models import (
    AgentConfig,
    Message,
    MemoryItem,
    MemorySession,
    WorkflowStep,
)
from ultra_claw.utils.config import load_config
from ultra_claw.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

# Global agent instance
_agent: Optional[UltraAgent] = None


class CreateAgentRequest(BaseModel):
    """Request to create an agent."""
    name: str = "Ultra-Claw Agent"
    config: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Request to chat with the agent."""
    messages: List[Dict[str, Any]]
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    use_memory: bool = True
    memory_categories: Optional[List[str]] = None


class ChatResponse(BaseModel):
    """Response from chat."""
    content: str
    session_id: str
    memories_used: List[str] = Field(default_factory=list)


class MemoryRequest(BaseModel):
    """Request to store a memory."""
    content: str
    user_id: str
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    modality: str = "text"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    """Response for memory operations."""
    id: str
    status: str


class RetrieveRequest(BaseModel):
    """Request to retrieve memories."""
    query: str
    user_id: Optional[str] = None
    categories: Optional[List[str]] = None
    top_k: int = Field(default=10, ge=1, le=100)


class WorkflowRequest(BaseModel):
    """Request to execute a workflow."""
    steps: List[Dict[str, Any]]
    context: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""
    healthy: bool
    version: str
    agent_id: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _agent
    
    # Startup
    configure_logging()
    logger.info("Starting Ultra-Claw API")
    
    # Load config and create agent
    config = load_config()
    _agent = UltraAgent(config=config)
    await _agent.initialize()
    
    logger.info(f"Ultra-Claw API started with agent {_agent.id}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Ultra-Claw API")
    if _agent:
        await _agent.shutdown()


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Ultra-Claw API",
        description="Next-generation intelligent agent framework with long-term memory",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            healthy=_agent is not None,
            version="1.0.0",
            agent_id=_agent.id if _agent else None
        )
    
    @app.post("/api/v1/agents")
    async def create_agent(request: CreateAgentRequest):
        """Create a new agent."""
        config = AgentConfig(
            name=request.name,
            **request.config
        )
        
        global _agent
        if _agent:
            await _agent.shutdown()
        
        _agent = UltraAgent(config=config)
        await _agent.initialize()
        
        return {
            "id": _agent.id,
            "name": _agent.config.name,
            "status": "created"
        }
    
    @app.get("/api/v1/agents/current")
    async def get_current_agent():
        """Get the current agent."""
        if not _agent:
            raise HTTPException(status_code=404, detail="No agent initialized")
        
        return {
            "id": _agent.id,
            "name": _agent.config.name,
            "config": _agent.config.model_dump()
        }
    
    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Chat with the agent."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        # Convert messages
        messages = [
            Message(role=msg.get("role", "user"), content=msg.get("content", ""))
            for msg in request.messages
        ]
        
        # Get response
        response_chunks = []
        session_id = None
        memories_used = []
        
        async for response in _agent.chat(
            messages=messages,
            session_id=request.session_id,
            user_id=request.user_id,
            use_memory=request.use_memory,
            memory_categories=request.memory_categories
        ):
            response_chunks.append(response.content)
            session_id = request.session_id  # Will be updated by agent
            if response.memories_used:
                memories_used = response.memories_used
        
        return ChatResponse(
            content="".join(response_chunks),
            session_id=session_id or "",
            memories_used=memories_used
        )
    
    @app.post("/api/v1/sessions")
    async def create_session(user_id: str, agent_id: Optional[str] = None):
        """Create a new session."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        session = await _agent.sessions.create_session(
            user_id=user_id,
            agent_id=agent_id or _agent.id
        )
        
        return {
            "id": session.id,
            "user_id": session.user_id,
            "agent_id": session.agent_id,
            "start_time": session.start_time.isoformat()
        }
    
    @app.get("/api/v1/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get session details."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        session = await _agent.sessions.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "id": session.id,
            "user_id": session.user_id,
            "agent_id": session.agent_id,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "message_count": len(session.messages),
            "is_active": session.is_active
        }
    
    @app.get("/api/v1/sessions/{session_id}/messages")
    async def get_session_messages(
        session_id: str,
        limit: int = Query(default=50, ge=1, le=1000),
        offset: int = Query(default=0, ge=0)
    ):
        """Get messages from a session."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        messages = await _agent.sessions.get_messages(session_id, limit=limit, offset=offset)
        
        return {
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in messages
            ]
        }
    
    @app.post("/api/v1/memory", response_model=MemoryResponse)
    async def store_memory(request: MemoryRequest):
        """Store a memory."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        item = await _agent.memorize(
            content=request.content,
            user_id=request.user_id,
            categories=request.categories,
            tags=request.tags,
            modality=request.modality,
            importance=request.importance,
            metadata=request.metadata
        )
        
        return MemoryResponse(id=item.id, status="stored")
    
    @app.get("/api/v1/memory")
    async def retrieve_memory(
        query: str,
        user_id: Optional[str] = None,
        categories: Optional[List[str]] = Query(default=None),
        top_k: int = Query(default=10, ge=1, le=100)
    ):
        """Retrieve memories."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        memories = await _agent.retrieve(
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
                    "tags": m.tags,
                    "timestamp": m.timestamp.isoformat(),
                    "importance": m.importance_score
                }
                for m in memories
            ],
            "total": len(memories)
        }
    
    @app.delete("/api/v1/memory/{memory_id}")
    async def delete_memory(memory_id: str):
        """Delete a memory."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        success = await _agent.forget(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return {"id": memory_id, "status": "deleted"}
    
    @app.post("/api/v1/workflows")
    async def execute_workflow(request: WorkflowRequest):
        """Execute a workflow."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        steps = [WorkflowStep(**step) for step in request.steps]
        state = await _agent.execute_workflow(steps, request.context)
        
        return {
            "state_id": state.id,
            "status": state.status,
            "completed_steps": state.completed_steps,
            "failed_steps": state.failed_steps,
            "results": state.results,
            "start_time": state.start_time.isoformat(),
            "end_time": state.end_time.isoformat() if state.end_time else None
        }
    
    @app.get("/api/v1/stats")
    async def get_stats():
        """Get agent statistics."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        stats = await _agent.get_stats()
        return stats
    
    return app


# Create the application instance
app = create_app()
