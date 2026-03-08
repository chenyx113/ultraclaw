"""
MemU integration for Ultra-Claw.

This module provides integration with the MemU memory plugin.
"""

from typing import Any, Dict, List, Optional

import httpx

from ultra_claw.core.models import MemoryItem, ModalityType
from ultra_claw.utils.logger import get_logger

logger = get_logger(__name__)


class MemUIntegration:
    """
    Integration with MemU memory plugin.
    
    Provides a client for the MemU API, allowing Ultra-Claw to
    use MemU as a memory backend.
    """
    
    def __init__(
        self,
        api_base: str = "http://localhost:8000",
        api_key: Optional[str] = None
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """Initialize the MemU client."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.client = httpx.AsyncClient(
            base_url=self.api_base,
            headers=headers,
            timeout=30.0
        )
        
        # Test connection
        health = await self.health_check()
        if health.get("healthy"):
            logger.info(f"Connected to MemU at {self.api_base}")
        else:
            logger.warning(f"MemU health check failed: {health}")
    
    async def shutdown(self) -> None:
        """Shutdown the MemU client."""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check MemU service health."""
        if not self.client:
            return {"healthy": False, "error": "Client not initialized"}
        
        try:
            response = await self.client.get("/health")
            if response.status_code == 200:
                return {"healthy": True, **response.json()}
            else:
                return {"healthy": False, "status_code": response.status_code}
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def store_memory(self, item: MemoryItem) -> str:
        """
        Store a memory via MemU.
        
        Args:
            item: Memory item to store
            
        Returns:
            Task ID from MemU
        """
        if not self.client:
            raise RuntimeError("MemU client not initialized")
        
        payload = {
            "user_id": item.user_id,
            "agent_id": item.agent_id,
            "content": item.content,
            "modality": item.modality.value,
            "categories": item.categories,
            "tags": item.tags,
            "metadata": item.metadata,
            "importance": item.importance_score,
        }
        
        response = await self.client.post("/api/v3/memory/memorize", json=payload)
        response.raise_for_status()
        
        result = response.json()
        logger.debug(f"Stored memory via MemU: {result.get('task_id')}")
        return result.get("task_id")
    
    async def retrieve_memory(
        self,
        query: str,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryItem]:
        """
        Retrieve memories via MemU.
        
        Args:
            query: Search query
            user_id: User ID
            filters: Optional filters
            
        Returns:
            List of memory items
        """
        if not self.client:
            raise RuntimeError("MemU client not initialized")
        
        payload = {
            "query": query,
            "user_id": user_id,
            "filters": filters or {},
        }
        
        response = await self.client.post("/api/v3/memory/retrieve", json=payload)
        response.raise_for_status()
        
        result = response.json()
        items = result.get("items", [])
        
        return [
            MemoryItem(
                id=item.get("id", ""),
                user_id=item.get("user_id", ""),
                agent_id=item.get("agent_id", ""),
                content=item.get("content", ""),
                modality=ModalityType(item.get("modality", "text")),
                categories=item.get("categories", []),
                tags=item.get("tags", []),
                metadata=item.get("metadata", {}),
            )
            for item in items
        ]
    
    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory via MemU.
        
        Args:
            memory_id: Memory ID to delete
            
        Returns:
            True if deleted
        """
        if not self.client:
            raise RuntimeError("MemU client not initialized")
        
        response = await self.client.delete(f"/api/v3/memory/{memory_id}")
        return response.status_code == 200
