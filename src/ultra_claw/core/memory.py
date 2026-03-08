"""
Memory Service for Ultra-Claw.

This module provides the core memory management functionality, including
storage, retrieval, updating, and deletion of memory items with support
for vector-based semantic search.
"""

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ultra_claw.core.models import (
    MemoryFilter,
    MemoryItem,
    MemoryCategory,
    ModalityType,
    MemoryLevel,
    RetrievalResult,
    MemoryConfig,
)
from ultra_claw.utils.logger import get_logger

logger = get_logger(__name__)


class VectorIndex:
    """
    In-memory vector index for semantic search.
    
    This implementation uses TF-IDF for text embeddings and cosine similarity
    for vector search. In production, this would be replaced with a proper
    vector database like Pinecone, Weaviate, or pgvector.
    """
    
    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
        self.items: Dict[str, MemoryItem] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix: Optional[np.ndarray] = None
        self._lock = asyncio.Lock()
    
    async def add(self, item: MemoryItem) -> None:
        """Add an item to the index."""
        async with self._lock:
            self.items[item.id] = item
            if item.embedding:
                self.embeddings[item.id] = np.array(item.embedding)
            else:
                # Generate simple embedding from content hash
                embedding = self._generate_simple_embedding(item.content)
                self.embeddings[item.id] = embedding
                item.embedding = embedding.tolist()
            self._rebuild_matrix()
    
    async def remove(self, item_id: str) -> bool:
        """Remove an item from the index."""
        async with self._lock:
            if item_id in self.items:
                del self.items[item_id]
                del self.embeddings[item_id]
                self._rebuild_matrix()
                return True
            return False
    
    async def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Search for similar items using vector similarity.
        
        Returns a list of (item_id, score) tuples sorted by score.
        """
        async with self._lock:
            if not self.items:
                return []
            
            # Use provided embedding or generate from query
            if query_embedding:
                query_vec = np.array(query_embedding).reshape(1, -1)
            else:
                query_vec = self._generate_simple_embedding(query).reshape(1, -1)
            
            # Calculate similarities
            results = []
            for item_id, embedding in self.embeddings.items():
                embedding_vec = embedding.reshape(1, -1)
                
                # Ensure compatible dimensions
                if query_vec.shape[1] != embedding_vec.shape[1]:
                    # Pad or truncate to match
                    target_dim = min(query_vec.shape[1], embedding_vec.shape[1])
                    query_vec = query_vec[:, :target_dim]
                    embedding_vec = embedding_vec[:, :target_dim]
                
                similarity = cosine_similarity(query_vec, embedding_vec)[0][0]
                if similarity >= threshold:
                    results.append((item_id, float(similarity)))
            
            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def _generate_simple_embedding(self, text: str) -> np.ndarray:
        """Generate a simple embedding from text using character n-grams."""
        # Simple character n-gram based embedding
        n = 3
        text = text.lower()
        ngrams = {}
        for i in range(len(text) - n + 1):
            gram = text[i:i+n]
            ngrams[gram] = ngrams.get(gram, 0) + 1
        
        # Create fixed-size vector
        vector_size = min(self.dimensions, 768)
        vector = np.zeros(vector_size)
        
        for gram, count in ngrams.items():
            # Hash n-gram to position
            idx = int(hashlib.md5(gram.encode()).hexdigest(), 16) % vector_size
            vector[idx] += count
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _rebuild_matrix(self) -> None:
        """Rebuild the similarity matrix."""
        if not self.embeddings:
            self.matrix = None
            return
        
        embeddings_list = [self.embeddings[item_id] for item_id in sorted(self.embeddings.keys())]
        if embeddings_list:
            self.matrix = np.vstack(embeddings_list)


class KeywordIndex:
    """Inverted index for keyword-based search."""
    
    def __init__(self):
        self.index: Dict[str, set] = {}  # word -> set of item_ids
        self._lock = asyncio.Lock()
    
    async def add(self, item: MemoryItem) -> None:
        """Add an item to the index."""
        async with self._lock:
            words = self._tokenize(item.content)
            for word in words:
                if word not in self.index:
                    self.index[word] = set()
                self.index[word].add(item.id)
            
            # Also index categories and tags
            for category in item.categories:
                cat_words = self._tokenize(category)
                for word in cat_words:
                    if word not in self.index:
                        self.index[word] = set()
                    self.index[word].add(item.id)
            
            for tag in item.tags:
                tag_words = self._tokenize(tag)
                for word in tag_words:
                    if word not in self.index:
                        self.index[word] = set()
                    self.index[word].add(item.id)
    
    async def remove(self, item_id: str) -> None:
        """Remove an item from the index."""
        async with self._lock:
            for word in list(self.index.keys()):
                self.index[word].discard(item_id)
                if not self.index[word]:
                    del self.index[word]
    
    async def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for items matching the query.
        
        Returns a list of (item_id, score) tuples.
        """
        async with self._lock:
            query_words = self._tokenize(query)
            scores: Dict[str, float] = {}
            
            for word in query_words:
                if word in self.index:
                    for item_id in self.index[word]:
                        scores[item_id] = scores.get(item_id, 0) + 1
            
            # Normalize scores
            if scores:
                max_score = max(scores.values())
                if max_score > 0:
                    scores = {k: v / max_score for k, v in scores.items()}
            
            results = list(scores.items())
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization
        text = text.lower()
        words = []
        current_word = []
        
        for char in text:
            if char.isalnum():
                current_word.append(char)
            else:
                if current_word:
                    word = ''.join(current_word)
                    if len(word) > 2:  # Filter very short words
                        words.append(word)
                    current_word = []
        
        if current_word:
            word = ''.join(current_word)
            if len(word) > 2:
                words.append(word)
        
        return words


class TemporalIndex:
    """Index for time-based retrieval."""
    
    def __init__(self):
        self.items: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
    
    async def add(self, item: MemoryItem) -> None:
        """Add an item to the index."""
        async with self._lock:
            self.items[item.id] = item.timestamp
    
    async def remove(self, item_id: str) -> None:
        """Remove an item from the index."""
        async with self._lock:
            self.items.pop(item_id, None)
    
    async def search(
        self,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for items within a time range.
        
        Returns items sorted by recency.
        """
        async with self._lock:
            results = []
            
            for item_id, timestamp in self.items.items():
                if time_range:
                    start, end = time_range
                    if not (start <= timestamp <= end):
                        continue
                
                # Score based on recency (newer = higher score)
                age = (datetime.utcnow() - timestamp).total_seconds()
                score = 1.0 / (1.0 + age / 86400)  # Decay over days
                results.append((item_id, score))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]


class HybridRetriever:
    """
    Hybrid retriever combining multiple retrieval strategies.
    
    Combines vector search, keyword search, and temporal search
    for comprehensive memory retrieval.
    """
    
    def __init__(
        self,
        vector_index: VectorIndex,
        keyword_index: KeywordIndex,
        temporal_index: TemporalIndex,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.3,
        temporal_weight: float = 0.2
    ):
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.temporal_index = temporal_index
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.temporal_weight = temporal_weight
    
    async def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        filters: Optional[MemoryFilter] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval combining multiple strategies.
        """
        # Get results from each method
        vector_results = await self.vector_index.search(
            query, query_embedding, top_k * 2, threshold=0.0
        )
        keyword_results = await self.keyword_index.search(query, top_k * 2)
        temporal_results = await self.temporal_index.search(time_range, top_k * 2)
        
        # Combine scores
        combined_scores: Dict[str, float] = {}
        
        for item_id, score in vector_results:
            combined_scores[item_id] = combined_scores.get(item_id, 0) + score * self.vector_weight
        
        for item_id, score in keyword_results:
            combined_scores[item_id] = combined_scores.get(item_id, 0) + score * self.keyword_weight
        
        for item_id, score in temporal_results:
            combined_scores[item_id] = combined_scores.get(item_id, 0) + score * self.temporal_weight
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build retrieval results
        results = []
        for item_id, score in sorted_results[:top_k]:
            item = self.vector_index.items.get(item_id)
            if item:
                # Apply filters
                if filters and not filters.matches(item):
                    continue
                
                # Determine retrieval method
                if item_id in [r[0] for r in vector_results[:5]]:
                    method = "hybrid"
                elif item_id in [r[0] for r in keyword_results[:5]]:
                    method = "keyword"
                else:
                    method = "temporal"
                
                results.append(RetrievalResult(
                    item=item,
                    score=score,
                    retrieval_method=method
                ))
        
        return results


class MemoryService:
    """
    Core memory service for Ultra-Claw.
    
    Provides comprehensive memory management including storage,
    retrieval, updating, and deletion of memory items.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.vector_index = VectorIndex(dimensions=self.config.embedding_dimensions)
        self.keyword_index = KeywordIndex()
        self.temporal_index = TemporalIndex()
        self.retriever = HybridRetriever(
            self.vector_index,
            self.keyword_index,
            self.temporal_index
        )
        self._storage: Dict[str, MemoryItem] = {}
        self._categories: Dict[str, MemoryCategory] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the memory service."""
        if self._initialized:
            return
        
        logger.info("Initializing MemoryService")
        # In a real implementation, this would connect to the database
        self._initialized = True
        logger.info("MemoryService initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the memory service."""
        logger.info("Shutting down MemoryService")
        self._initialized = False
    
    async def store(self, item: MemoryItem) -> str:
        """
        Store a memory item.
        
        Args:
            item: The memory item to store
            
        Returns:
            The ID of the stored item
        """
        await self.initialize()
        
        async with self._lock:
            # Store in main storage
            self._storage[item.id] = item
            
            # Add to indices
            await self.vector_index.add(item)
            await self.keyword_index.add(item)
            await self.temporal_index.add(item)
            
            logger.debug(f"Stored memory item {item.id}", 
                        extra={"item_id": item.id, "user_id": item.user_id})
            
            return item.id
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[MemoryFilter] = None,
        top_k: int = 10,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[MemoryItem]:
        """
        Retrieve memories matching the query.
        
        Args:
            query: The search query
            filters: Optional filters to apply
            top_k: Maximum number of results
            time_range: Optional time range filter
            
        Returns:
            List of matching memory items
        """
        await self.initialize()
        
        results = await self.retriever.retrieve(
            query=query,
            filters=filters,
            time_range=time_range,
            top_k=top_k
        )
        
        # Update access statistics
        for result in results:
            result.item.touch()
        
        logger.debug(f"Retrieved {len(results)} memories for query: {query}")
        return [r.item for r in results]
    
    async def retrieve_with_scores(
        self,
        query: str,
        filters: Optional[MemoryFilter] = None,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Retrieve memories with relevance scores.
        
        Args:
            query: The search query
            filters: Optional filters to apply
            top_k: Maximum number of results
            
        Returns:
            List of retrieval results with scores
        """
        await self.initialize()
        return await self.retriever.retrieve(query, filters=filters, top_k=top_k)
    
    async def get(self, item_id: str) -> Optional[MemoryItem]:
        """
        Get a memory item by ID.
        
        Args:
            item_id: The ID of the item to retrieve
            
        Returns:
            The memory item, or None if not found
        """
        await self.initialize()
        
        item = self._storage.get(item_id)
        if item:
            item.touch()
        return item
    
    async def update(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a memory item.
        
        Args:
            item_id: The ID of the item to update
            updates: Dictionary of fields to update
            
        Returns:
            True if the item was updated, False otherwise
        """
        await self.initialize()
        
        async with self._lock:
            item = self._storage.get(item_id)
            if not item:
                return False
            
            # Update fields
            for key, value in updates.items():
                if hasattr(item, key):
                    setattr(item, key, value)
            
            # Re-index if content changed
            if "content" in updates or "categories" in updates:
                await self.vector_index.remove(item_id)
                await self.keyword_index.remove(item_id)
                await self.vector_index.add(item)
                await self.keyword_index.add(item)
            
            logger.debug(f"Updated memory item {item_id}")
            return True
    
    async def delete(self, item_id: str) -> bool:
        """
        Delete a memory item.
        
        Args:
            item_id: The ID of the item to delete
            
        Returns:
            True if the item was deleted, False otherwise
        """
        await self.initialize()
        
        async with self._lock:
            if item_id not in self._storage:
                return False
            
            del self._storage[item_id]
            await self.vector_index.remove(item_id)
            await self.keyword_index.remove(item_id)
            await self.temporal_index.remove(item_id)
            
            logger.debug(f"Deleted memory item {item_id}")
            return True
    
    async def consolidate(
        self,
        user_id: str,
        agent_id: str,
        strategy: str = "auto"
    ) -> bool:
        """
        Consolidate memories for a user/agent.
        
        This performs maintenance operations like removing expired items,
        merging duplicates, and updating importance scores.
        
        Args:
            user_id: The user ID
            agent_id: The agent ID
            strategy: Consolidation strategy (auto, aggressive, gentle)
            
        Returns:
            True if consolidation was successful
        """
        await self.initialize()
        
        async with self._lock:
            items_to_remove = []
            
            for item_id, item in self._storage.items():
                if item.user_id != user_id or item.agent_id != agent_id:
                    continue
                
                # Remove expired items
                if item.is_expired():
                    items_to_remove.append(item_id)
                    continue
                
                # Update importance based on access patterns
                if strategy == "aggressive":
                    if item.access_count == 0 and item.importance_score < 0.3:
                        items_to_remove.append(item_id)
                elif strategy == "auto":
                    if item.access_count == 0 and item.importance_score < 0.1:
                        items_to_remove.append(item_id)
            
            # Remove items
            for item_id in items_to_remove:
                await self.delete(item_id)
            
            logger.info(f"Consolidated memories: removed {len(items_to_remove)} items")
            return True
    
    async def get_by_category(
        self,
        user_id: str,
        category: str,
        agent_id: Optional[str] = None
    ) -> List[MemoryItem]:
        """
        Get all memories in a category.
        
        Args:
            user_id: The user ID
            category: The category name
            agent_id: Optional agent ID filter
            
        Returns:
            List of matching memory items
        """
        await self.initialize()
        
        results = []
        for item in self._storage.values():
            if item.user_id == user_id and category in item.categories:
                if agent_id is None or item.agent_id == agent_id:
                    results.append(item)
        
        return results
    
    async def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            Dictionary of statistics
        """
        await self.initialize()
        
        items = list(self._storage.values())
        if user_id:
            items = [i for i in items if i.user_id == user_id]
        
        total_items = len(items)
        by_modality = {}
        by_category = {}
        
        for item in items:
            modality = item.modality.value
            by_modality[modality] = by_modality.get(modality, 0) + 1
            
            for cat in item.categories:
                by_category[cat] = by_category.get(cat, 0) + 1
        
        return {
            "total_items": total_items,
            "by_modality": by_modality,
            "by_category": by_category,
            "index_size": len(self.vector_index.items),
        }
    
    async def clear(self, user_id: Optional[str] = None) -> int:
        """
        Clear all memories.
        
        Args:
            user_id: Optional user ID to clear only that user's memories
            
        Returns:
            Number of items cleared
        """
        await self.initialize()
        
        async with self._lock:
            if user_id:
                items_to_remove = [
                    item_id for item_id, item in self._storage.items()
                    if item.user_id == user_id
                ]
            else:
                items_to_remove = list(self._storage.keys())
            
            for item_id in items_to_remove:
                await self.delete(item_id)
            
            return len(items_to_remove)
