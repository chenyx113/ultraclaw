"""
Core data models for Ultra-Claw.

This module defines all the Pydantic models used throughout the framework,
including memory items, sessions, messages, and configuration.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModalityType(str, Enum):
    """Supported content modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class MessageRole(str, Enum):
    """Message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MemoryLevel(str, Enum):
    """Memory hierarchy levels."""
    SENSORY = "sensory"  # Short-term sensory memory
    WORKING = "working"   # Working memory (current context)
    LONG_TERM = "long_term"  # Long-term persistent memory


class Message(BaseModel):
    """A message in a conversation."""
    model_config = ConfigDict(extra="allow")
    
    role: MessageRole
    content: Union[str, List[Dict[str, Any]]]
    modality: ModalityType = ModalityType.TEXT
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Union[str, List[Dict[str, Any]]]) -> Union[str, List[Dict[str, Any]]]:
        """Validate message content."""
        if isinstance(v, str) and len(v) == 0:
            raise ValueError("Content cannot be empty")
        return v


class MemoryItem(BaseModel):
    """
    A memory item stored in the system.
    
    This represents any piece of information that the agent can remember,
    including text, images, audio, or video content.
    """
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    agent_id: str
    content: str
    modality: ModalityType = ModalityType.TEXT
    embedding: Optional[List[float]] = None
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    expires_at: Optional[datetime] = None
    memory_level: MemoryLevel = MemoryLevel.LONG_TERM
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    last_accessed: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if the memory item has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class MemoryCategory(BaseModel):
    """
    A category for organizing memories.
    
    Categories help structure memories into logical groups,
    making retrieval more efficient.
    """
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    embedding: Optional[List[float]] = None
    parent_id: Optional[str] = None
    user_id: str
    agent_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemorySession(BaseModel):
    """
    A conversation session.
    
    Sessions track the context of conversations between users and agents,
    including all messages and associated metadata.
    """
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    agent_id: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    messages: List[Message] = Field(default_factory=list)
    summary: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    
    def add_message(self, message: Message) -> None:
        """Add a message to the session."""
        self.messages.append(message)
    
    def close(self) -> None:
        """Close the session."""
        self.end_time = datetime.utcnow()
        self.is_active = False


class MemoryFilter(BaseModel):
    """
    Filter criteria for memory retrieval.
    
    This allows fine-grained control over which memories are retrieved
    based on various criteria like categories, time range, and modality.
    """
    model_config = ConfigDict(extra="allow")
    
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    modality: Optional[ModalityType] = None
    memory_level: Optional[MemoryLevel] = None
    time_range: Optional[tuple[datetime, datetime]] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    min_importance: Optional[float] = None
    max_importance: Optional[float] = None
    source: Optional[str] = None
    
    def matches(self, item: MemoryItem) -> bool:
        """Check if a memory item matches this filter."""
        if self.categories and not any(cat in item.categories for cat in self.categories):
            return False
        if self.tags and not any(tag in item.tags for tag in self.tags):
            return False
        if self.modality and item.modality != self.modality:
            return False
        if self.memory_level and item.memory_level != self.memory_level:
            return False
        if self.time_range:
            start, end = self.time_range
            if not (start <= item.timestamp <= end):
                return False
        if self.user_id and item.user_id != self.user_id:
            return False
        if self.agent_id and item.agent_id != self.agent_id:
            return False
        if self.min_importance is not None and item.importance_score < self.min_importance:
            return False
        if self.max_importance is not None and item.importance_score > self.max_importance:
            return False
        if self.source and item.source != self.source:
            return False
        return True


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    model_config = ConfigDict(extra="allow")
    
    provider: str = "openai"
    api_key: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    base_url: Optional[str] = None
    timeout: int = 60


class MemoryConfig(BaseModel):
    """Configuration for memory backend."""
    model_config = ConfigDict(extra="allow")
    
    backend: str = "sqlite"  # sqlite, postgres, pinecone, weaviate
    database_url: str = "sqlite:///memory.db"
    vector_store: str = "sqlite-vec"
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimensions: int = 1536
    max_context_memories: int = 10
    similarity_threshold: float = 0.7
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    categories: List[Dict[str, str]] = Field(default_factory=list)


class WorkflowConfig(BaseModel):
    """Configuration for workflow engine."""
    model_config = ConfigDict(extra="allow")
    
    max_steps: int = 50
    timeout: int = 300  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds


class SecurityConfig(BaseModel):
    """Configuration for security settings."""
    model_config = ConfigDict(extra="allow")
    
    encryption_enabled: bool = True
    encryption_key: Optional[str] = None
    access_control_enabled: bool = True
    audit_logging_enabled: bool = True
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    rate_limit_requests: int = 100  # per minute
    rate_limit_window: int = 60  # seconds


class AgentConfig(BaseModel):
    """
    Complete configuration for an UltraAgent.
    
    This aggregates all configuration settings for the agent,
    including LLM, memory, workflow, and security settings.
    """
    model_config = ConfigDict(extra="allow")
    
    name: str = "Ultra-Claw Agent"
    version: str = "1.0.0"
    description: str = ""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    plugins: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowStep(BaseModel):
    """
    A single step in a workflow.
    
    Steps define individual actions that can be executed
    as part of a larger workflow.
    """
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    action: str  # The action to execute
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    condition: Optional[str] = None  # Conditional execution
    retry_count: int = Field(default=0, ge=0)
    timeout: Optional[int] = None
    on_error: Optional[str] = None  # Error handling strategy


class WorkflowState(BaseModel):
    """
    The state of a workflow execution.
    
    Tracks the progress and results of a running workflow.
    """
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = "pending"
    current_step: Optional[str] = None
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    results: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None


class RetrievalResult(BaseModel):
    """Result of a memory retrieval operation."""
    model_config = ConfigDict(extra="allow")
    
    item: MemoryItem
    score: float
    retrieval_method: str  # vector, keyword, hybrid


class AgentResponse(BaseModel):
    """Response from an agent."""
    model_config = ConfigDict(extra="allow")
    
    content: str
    role: MessageRole = MessageRole.ASSISTANT
    modality: ModalityType = ModalityType.TEXT
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    memories_used: List[str] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
