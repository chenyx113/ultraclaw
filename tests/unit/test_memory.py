"""
Unit tests for memory service.
"""

import pytest
import asyncio

from ultra_claw.core.memory import (
    MemoryService,
    VectorIndex,
    KeywordIndex,
    TemporalIndex,
    HybridRetriever,
)
from ultra_claw.core.models import MemoryItem, MemoryFilter, ModalityType


class TestVectorIndex:
    """Tests for VectorIndex."""
    
    @pytest.fixture
    def index(self):
        return VectorIndex(dimensions=128)
    
    @pytest.mark.asyncio
    async def test_add_and_search(self, index):
        """Test adding items and searching."""
        item1 = MemoryItem(
            user_id="user-1",
            agent_id="agent-1",
            content="Python programming language"
        )
        item2 = MemoryItem(
            user_id="user-1",
            agent_id="agent-1",
            content="JavaScript web development"
        )
        
        await index.add(item1)
        await index.add(item2)
        
        results = await index.search("Python", top_k=5)
        assert len(results) > 0
        assert results[0][0] == item1.id
    
    @pytest.mark.asyncio
    async def test_remove(self, index):
        """Test removing items."""
        item = MemoryItem(
            user_id="user-1",
            agent_id="agent-1",
            content="Test content"
        )
        
        await index.add(item)
        assert item.id in index.items
        
        success = await index.remove(item.id)
        assert success
        assert item.id not in index.items


class TestKeywordIndex:
    """Tests for KeywordIndex."""
    
    @pytest.fixture
    def index(self):
        return KeywordIndex()
    
    @pytest.mark.asyncio
    async def test_add_and_search(self, index):
        """Test adding items and searching."""
        item = MemoryItem(
            user_id="user-1",
            agent_id="agent-1",
            content="machine learning artificial intelligence"
        )
        
        await index.add(item)
        
        results = await index.search("machine learning", top_k=5)
        assert len(results) > 0
        assert results[0][0] == item.id
    
    @pytest.mark.asyncio
    async def test_tokenize(self, index):
        """Test text tokenization."""
        words = index._tokenize("Hello, World! This is a test.")
        assert "hello" in words
        assert "world" in words
        assert "test" in words


class TestMemoryService:
    """Tests for MemoryService."""
    
    @pytest.fixture
    async def service(self):
        service = MemoryService()
        await service.initialize()
        yield service
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, service):
        """Test storing and retrieving memories."""
        item = MemoryItem(
            user_id="user-1",
            agent_id="agent-1",
            content="Important information to remember"
        )
        
        item_id = await service.store(item)
        assert item_id is not None
        
        retrieved = await service.get(item_id)
        assert retrieved is not None
        assert retrieved.content == "Important information to remember"
    
    @pytest.mark.asyncio
    async def test_retrieve_with_query(self, service):
        """Test retrieving memories with query."""
        # Store some memories
        await service.store(MemoryItem(
            user_id="user-1",
            agent_id="agent-1",
            content="Python is a great programming language"
        ))
        await service.store(MemoryItem(
            user_id="user-1",
            agent_id="agent-1",
            content="JavaScript is used for web development"
        ))
        
        # Retrieve
        results = await service.retrieve("Python programming", top_k=5)
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_update(self, service):
        """Test updating memories."""
        item = MemoryItem(
            user_id="user-1",
            agent_id="agent-1",
            content="Original content"
        )
        
        item_id = await service.store(item)
        
        success = await service.update(item_id, {"content": "Updated content"})
        assert success
        
        updated = await service.get(item_id)
        assert updated.content == "Updated content"
    
    @pytest.mark.asyncio
    async def test_delete(self, service):
        """Test deleting memories."""
        item = MemoryItem(
            user_id="user-1",
            agent_id="agent-1",
            content="Content to delete"
        )
        
        item_id = await service.store(item)
        
        success = await service.delete(item_id)
        assert success
        
        deleted = await service.get(item_id)
        assert deleted is None
    
    @pytest.mark.asyncio
    async def test_get_stats(self, service):
        """Test getting statistics."""
        await service.store(MemoryItem(
            user_id="user-1",
            agent_id="agent-1",
            content="Test 1",
            modality=ModalityType.TEXT
        ))
        await service.store(MemoryItem(
            user_id="user-1",
            agent_id="agent-1",
            content="Test 2",
            modality=ModalityType.IMAGE
        ))
        
        stats = await service.get_stats(user_id="user-1")
        assert stats["total_items"] == 2
        assert stats["by_modality"]["text"] == 1
        assert stats["by_modality"]["image"] == 1
    
    @pytest.mark.asyncio
    async def test_clear(self, service):
        """Test clearing memories."""
        await service.store(MemoryItem(
            user_id="user-1",
            agent_id="agent-1",
            content="Test"
        ))
        
        count = await service.clear(user_id="user-1")
        assert count == 1
        
        stats = await service.get_stats(user_id="user-1")
        assert stats["total_items"] == 0
