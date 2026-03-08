"""
Integration tests for the API.
"""

import pytest
from fastapi.testclient import TestClient

from ultra_claw.api.main import create_app


class TestAPI:
    """Tests for the REST API."""
    
    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "healthy" in data
        assert "version" in data
    
    def test_create_and_get_agent(self, client):
        """Test agent creation and retrieval."""
        # Create agent
        response = client.post("/api/v1/agents", json={
            "name": "Test Agent",
            "config": {"llm": {"provider": "mock"}}
        })
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Agent"
        assert data["status"] == "created"
        
        # Get current agent
        response = client.get("/api/v1/agents/current")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Agent"
    
    def test_chat(self, client):
        """Test chat endpoint."""
        # First create an agent
        client.post("/api/v1/agents", json={
            "name": "Test Agent",
            "config": {"llm": {"provider": "mock", "responses": ["Hello!"]}}
        })
        
        # Chat
        response = client.post("/api/v1/chat", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "user_id": "test-user"
        })
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "session_id" in data
    
    def test_memory_operations(self, client):
        """Test memory endpoints."""
        # Create agent
        client.post("/api/v1/agents", json={
            "name": "Test Agent",
            "config": {"llm": {"provider": "mock"}}
        })
        
        # Store memory
        response = client.post("/api/v1/memory", json={
            "content": "Test memory content",
            "user_id": "test-user",
            "categories": ["test"]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stored"
        memory_id = data["id"]
        
        # Retrieve memories
        response = client.get("/api/v1/memory", params={
            "query": "test memory",
            "user_id": "test-user"
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        
        # Delete memory
        response = client.delete(f"/api/v1/memory/{memory_id}")
        assert response.status_code == 200
    
    def test_session_management(self, client):
        """Test session endpoints."""
        # Create agent
        client.post("/api/v1/agents", json={
            "name": "Test Agent",
            "config": {"llm": {"provider": "mock"}}
        })
        
        # Create session
        response = client.post("/api/v1/sessions", params={
            "user_id": "test-user"
        })
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        session_id = data["id"]
        
        # Get session
        response = client.get(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test-user"
        
        # Get messages
        response = client.get(f"/api/v1/sessions/{session_id}/messages")
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
    
    def test_stats(self, client):
        """Test stats endpoint."""
        # Create agent
        client.post("/api/v1/agents", json={
            "name": "Test Agent",
            "config": {"llm": {"provider": "mock"}}
        })
        
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "agent_id" in data
        assert "memory" in data
        assert "sessions" in data
