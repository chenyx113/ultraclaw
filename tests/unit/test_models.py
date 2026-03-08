"""
Unit tests for core models.
"""

import pytest
from datetime import datetime, timedelta

from ultra_claw.core.models import (
    MemoryItem,
    MemoryCategory,
    MemorySession,
    Message,
    MemoryFilter,
    ModalityType,
    MessageRole,
    MemoryLevel,
    AgentConfig,
)


class TestMemoryItem:
    """Tests for MemoryItem model."""
    
    def test_create_memory_item(self):
        """Test creating a memory item."""
        item = MemoryItem(
            user_id="user-123",
            agent_id="agent-456",
            content="Test memory content",
            categories=["test", "memory"],
            tags=["important"],
        )
        
        assert item.user_id == "user-123"
        assert item.agent_id == "agent-456"
        assert item.content == "Test memory content"
        assert item.categories == ["test", "memory"]
        assert item.tags == ["important"]
        assert item.modality == ModalityType.TEXT
        assert item.id is not None
        assert item.timestamp is not None
    
    def test_memory_item_expiration(self):
        """Test memory item expiration."""
        # Not expired
        item = MemoryItem(
            user_id="user-123",
            agent_id="agent-456",
            content="Test content",
            expires_at=datetime.utcnow() + timedelta(days=1)
        )
        assert not item.is_expired()
        
        # Expired
        item.expires_at = datetime.utcnow() - timedelta(days=1)
        assert item.is_expired()
        
        # No expiration
        item.expires_at = None
        assert not item.is_expired()
    
    def test_memory_item_touch(self):
        """Test memory item touch method."""
        item = MemoryItem(
            user_id="user-123",
            agent_id="agent-456",
            content="Test content"
        )
        
        assert item.access_count == 0
        assert item.last_accessed is None
        
        item.touch()
        
        assert item.access_count == 1
        assert item.last_accessed is not None


class TestMemoryFilter:
    """Tests for MemoryFilter model."""
    
    def test_filter_matches_categories(self):
        """Test category filtering."""
        item = MemoryItem(
            user_id="user-123",
            agent_id="agent-456",
            content="Test",
            categories=["personal", "important"]
        )
        
        # Matching filter
        filter_match = MemoryFilter(categories=["personal"])
        assert filter_match.matches(item)
        
        # Non-matching filter
        filter_no_match = MemoryFilter(categories=["work"])
        assert not filter_no_match.matches(item)
    
    def test_filter_matches_user(self):
        """Test user ID filtering."""
        item = MemoryItem(
            user_id="user-123",
            agent_id="agent-456",
            content="Test"
        )
        
        filter_match = MemoryFilter(user_id="user-123")
        assert filter_match.matches(item)
        
        filter_no_match = MemoryFilter(user_id="user-999")
        assert not filter_no_match.matches(item)
    
    def test_filter_matches_modality(self):
        """Test modality filtering."""
        item = MemoryItem(
            user_id="user-123",
            agent_id="agent-456",
            content="Test",
            modality=ModalityType.IMAGE
        )
        
        filter_match = MemoryFilter(modality=ModalityType.IMAGE)
        assert filter_match.matches(item)
        
        filter_no_match = MemoryFilter(modality=ModalityType.TEXT)
        assert not filter_no_match.matches(item)


class TestMessage:
    """Tests for Message model."""
    
    def test_create_message(self):
        """Test creating a message."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello, world!"
        )
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
        assert msg.modality == ModalityType.TEXT
        assert msg.timestamp is not None
    
    def test_message_validation(self):
        """Test message content validation."""
        with pytest.raises(ValueError):
            Message(role=MessageRole.USER, content="")


class TestMemorySession:
    """Tests for MemorySession model."""
    
    def test_create_session(self):
        """Test creating a session."""
        session = MemorySession(
            user_id="user-123",
            agent_id="agent-456"
        )
        
        assert session.user_id == "user-123"
        assert session.agent_id == "agent-456"
        assert session.is_active
        assert len(session.messages) == 0
    
    def test_add_message(self):
        """Test adding messages to session."""
        session = MemorySession(
            user_id="user-123",
            agent_id="agent-456"
        )
        
        msg = Message(role=MessageRole.USER, content="Hello")
        session.add_message(msg)
        
        assert len(session.messages) == 1
        assert session.messages[0].content == "Hello"
    
    def test_close_session(self):
        """Test closing a session."""
        session = MemorySession(
            user_id="user-123",
            agent_id="agent-456"
        )
        
        session.close()
        
        assert not session.is_active
        assert session.end_time is not None


class TestAgentConfig:
    """Tests for AgentConfig model."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = AgentConfig()
        
        assert config.name == "Ultra-Claw Agent"
        assert config.version == "1.0.0"
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4"
        assert config.memory.backend == "sqlite"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            name="Custom Agent",
            llm={"provider": "anthropic", "model": "claude-3"},
            memory={"backend": "postgres"}
        )
        
        assert config.name == "Custom Agent"
        assert config.llm.provider == "anthropic"
        assert config.llm.model == "claude-3"
        assert config.memory.backend == "postgres"
