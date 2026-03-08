"""
Unit tests for agent engine.
"""

import pytest

from ultra_claw.core.agent import UltraAgent
from ultra_claw.core.models import AgentConfig, Message, MessageRole
from ultra_claw.services.llm.mock_provider import MockProvider


class TestUltraAgent:
    """Tests for UltraAgent."""
    
    @pytest.fixture
    async def agent(self):
        config = AgentConfig(
            name="Test Agent",
            llm={"provider": "mock", "responses": ["Test response"]}
        )
        agent = UltraAgent(config=config)
        await agent.initialize()
        yield agent
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.id is not None
        assert agent.config.name == "Test Agent"
        assert agent.memory is not None
        assert agent.llm is not None
    
    @pytest.mark.asyncio
    async def test_memorize(self, agent):
        """Test memorizing content."""
        item = await agent.memorize(
            content="Important fact to remember",
            user_id="user-1",
            categories=["knowledge"],
            importance=0.8
        )
        
        assert item.id is not None
        assert item.content == "Important fact to remember"
        assert item.user_id == "user-1"
        assert item.categories == ["knowledge"]
        assert item.importance_score == 0.8
    
    @pytest.mark.asyncio
    async def test_retrieve(self, agent):
        """Test retrieving memories."""
        # Store a memory
        await agent.memorize(
            content="Python programming tips",
            user_id="user-1",
            categories=["programming"]
        )
        
        # Retrieve
        memories = await agent.retrieve(
            query="Python tips",
            user_id="user-1"
        )
        
        assert len(memories) > 0
    
    @pytest.mark.asyncio
    async def test_forget(self, agent):
        """Test forgetting a memory."""
        item = await agent.memorize(
            content="Temporary information",
            user_id="user-1"
        )
        
        success = await agent.forget(item.id)
        assert success
        
        # Verify deletion
        memories = await agent.retrieve("Temporary", user_id="user-1")
        assert len(memories) == 0
    
    @pytest.mark.asyncio
    async def test_chat(self, agent):
        """Test chatting with the agent."""
        messages = [Message(role=MessageRole.USER, content="Hello")]
        
        responses = []
        async for response in agent.chat(messages, user_id="user-1"):
            responses.append(response.content)
        
        full_response = "".join(responses)
        assert len(full_response) > 0
    
    @pytest.mark.asyncio
    async def test_get_stats(self, agent):
        """Test getting agent statistics."""
        stats = await agent.get_stats()
        
        assert "agent_id" in stats
        assert "agent_name" in stats
        assert "memory" in stats
        assert "sessions" in stats
