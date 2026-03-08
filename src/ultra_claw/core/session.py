"""
Session Management for Ultra-Claw.

This module provides session management capabilities, tracking
conversations between users and agents with support for persistence
and cross-platform synchronization.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ultra_claw.core.models import MemorySession, Message, MessageRole
from ultra_claw.utils.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    """
    Manager for conversation sessions.
    
    Provides:
    - Session creation and lifecycle management
    - Message tracking
    - Session persistence
    - Cross-platform synchronization support
    """
    
    def __init__(
        self,
        max_session_age: int = 86400,  # 24 hours
        max_messages_per_session: int = 1000
    ):
        self.max_session_age = max_session_age
        self.max_messages_per_session = max_messages_per_session
        self._sessions: Dict[str, MemorySession] = {}
        self._user_sessions: Dict[str, set] = {}  # user_id -> set of session_ids
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the session manager."""
        logger.info("Initializing SessionManager")
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def shutdown(self) -> None:
        """Shutdown the session manager."""
        logger.info("Shutting down SessionManager")
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def create_session(
        self,
        user_id: str,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemorySession:
        """
        Create a new session.
        
        Args:
            user_id: The user ID
            agent_id: The agent ID
            metadata: Optional session metadata
            
        Returns:
            The created session
        """
        async with self._lock:
            session = MemorySession(
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata or {}
            )
            
            self._sessions[session.id] = session
            
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = set()
            self._user_sessions[user_id].add(session.id)
            
            logger.info(f"Created session {session.id} for user {user_id}")
            return session
    
    async def get_session(self, session_id: str) -> Optional[MemorySession]:
        """
        Get a session by ID.
        
        Args:
            session_id: The session ID
            
        Returns:
            The session, or None if not found
        """
        return self._sessions.get(session_id)
    
    async def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        modality: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Message]:
        """
        Add a message to a session.
        
        Args:
            session_id: The session ID
            role: The message role
            content: The message content
            modality: The content modality
            metadata: Optional message metadata
            
        Returns:
            The created message, or None if session not found
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found")
                return None
            
            if not session.is_active:
                logger.warning(f"Session {session_id} is not active")
                return None
            
            # Check message limit
            if len(session.messages) >= self.max_messages_per_session:
                logger.warning(f"Session {session_id} has reached message limit")
                # Summarize older messages
                await self._summarize_session(session)
            
            message = Message(
                role=role,
                content=content,
                modality=modality,
                metadata=metadata or {}
            )
            
            session.add_message(message)
            logger.debug(f"Added message to session {session_id}")
            return message
    
    async def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """
        Get messages from a session.
        
        Args:
            session_id: The session ID
            limit: Maximum number of messages
            offset: Number of messages to skip
            
        Returns:
            List of messages
        """
        session = self._sessions.get(session_id)
        if not session:
            return []
        
        messages = session.messages
        if offset:
            messages = messages[offset:]
        if limit:
            messages = messages[:limit]
        
        return messages
    
    async def close_session(self, session_id: str) -> bool:
        """
        Close a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            True if the session was closed, False otherwise
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            session.close()
            logger.info(f"Closed session {session_id}")
            return True
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            True if the session was deleted, False otherwise
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            del self._sessions[session_id]
            
            if session.user_id in self._user_sessions:
                self._user_sessions[session.user_id].discard(session_id)
            
            logger.info(f"Deleted session {session_id}")
            return True
    
    async def get_user_sessions(
        self,
        user_id: str,
        active_only: bool = False
    ) -> List[MemorySession]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: The user ID
            active_only: Only return active sessions
            
        Returns:
            List of sessions
        """
        session_ids = self._user_sessions.get(user_id, set())
        sessions = []
        
        for session_id in session_ids:
            session = self._sessions.get(session_id)
            if session:
                if not active_only or session.is_active:
                    sessions.append(session)
        
        return sessions
    
    async def get_context_window(
        self,
        session_id: str,
        max_messages: int = 10
    ) -> List[Message]:
        """
        Get the recent context window for a session.
        
        Args:
            session_id: The session ID
            max_messages: Maximum number of messages to include
            
        Returns:
            List of recent messages
        """
        session = self._sessions.get(session_id)
        if not session:
            return []
        
        # Return the most recent messages
        return session.messages[-max_messages:] if session.messages else []
    
    async def update_session_summary(
        self,
        session_id: str,
        summary: str
    ) -> bool:
        """
        Update the summary for a session.
        
        Args:
            session_id: The session ID
            summary: The summary text
            
        Returns:
            True if updated, False otherwise
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            
            session.summary = summary
            return True
    
    async def _summarize_session(self, session: MemorySession) -> None:
        """
        Summarize older messages in a session.
        
        This is called when a session reaches its message limit.
        In a real implementation, this would use an LLM to generate
        a summary of the conversation.
        """
        # Keep the most recent 100 messages
        keep_count = 100
        if len(session.messages) > keep_count:
            older_messages = session.messages[:-keep_count]
            recent_messages = session.messages[-keep_count:]
            
            # Generate summary (placeholder)
            summary = f"Session contained {len(older_messages)} older messages"
            if session.summary:
                session.summary = f"{session.summary}\n{summary}"
            else:
                session.summary = summary
            
            session.messages = recent_messages
            
            logger.info(f"Summarized session {session.id}")
    
    async def _periodic_cleanup(self) -> None:
        """Periodically clean up old sessions."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
    
    async def _cleanup_old_sessions(self) -> None:
        """Remove old inactive sessions."""
        async with self._lock:
            cutoff = datetime.utcnow() - timedelta(seconds=self.max_session_age)
            to_remove = []
            
            for session_id, session in self._sessions.items():
                if not session.is_active and session.end_time and session.end_time < cutoff:
                    to_remove.append(session_id)
            
            for session_id in to_remove:
                await self.delete_session(session_id)
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old sessions")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        total_sessions = len(self._sessions)
        active_sessions = sum(1 for s in self._sessions.values() if s.is_active)
        total_messages = sum(len(s.messages) for s in self._sessions.values())
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "inactive_sessions": total_sessions - active_sessions,
            "total_messages": total_messages,
            "unique_users": len(self._user_sessions),
        }
