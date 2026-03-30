"""
Conversation Repository
Data access layer for Conversation and Message entities
"""

from typing import Optional, List
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from models.conversation import Conversation, Message, MessageRole
from .base_repository import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    """
    Conversation repository with custom query methods
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(Conversation, session)
    
    async def find_by_user_id(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 50,
        include_messages: bool = False
    ) -> List[Conversation]:
        """Find all conversations for a user"""
        query = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        if include_messages:
            query = query.options(selectinload(Conversation.messages))
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def find_by_id_with_messages(
        self,
        conversation_id: int,
        user_id: Optional[int] = None
    ) -> Optional[Conversation]:
        """Get conversation with all messages"""
        query = (
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(selectinload(Conversation.messages))
        )
        if user_id:
            query = query.where(Conversation.user_id == user_id)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def count_by_user_id(self, user_id: int) -> int:
        """Count conversations for a user"""
        query = (
            select(func.count())
            .select_from(Conversation)
            .where(Conversation.user_id == user_id)
        )
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def find_active_conversations(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 50
    ) -> List[Conversation]:
        """Find active conversations for user"""
        query = (
            select(Conversation)
            .where(
                and_(
                    Conversation.user_id == user_id,
                    Conversation.is_active == True
                )
            )
            .order_by(Conversation.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def find_recent_conversations(
        self,
        user_id: int,
        days: int = 7,
        limit: int = 10
    ) -> List[Conversation]:
        """Find conversations from last N days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = (
            select(Conversation)
            .where(
                and_(
                    Conversation.user_id == user_id,
                    Conversation.updated_at >= cutoff
                )
            )
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())


class MessageRepository(BaseRepository[Message]):
    """
    Message repository
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(Message, session)
    
    async def find_by_conversation_id(
        self,
        conversation_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Message]:
        """Get messages for a conversation"""
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def count_by_conversation_id(self, conversation_id: int) -> int:
        """Count messages in conversation"""
        query = (
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == conversation_id)
        )
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def find_faq_matches(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[Message]:
        """Find messages that were FAQ matches (for analytics)"""
        query = (
            select(Message)
            .where(Message.was_faq_match == True)
            .order_by(Message.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def count_faq_matches(self) -> int:
        """Count FAQ match messages"""
        query = (
            select(func.count())
            .select_from(Message)
            .where(Message.was_faq_match == True)
        )
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def count_rag_responses(self) -> int:
        """Count RAG (non-FAQ) responses"""
        query = (
            select(func.count())
            .select_from(Message)
            .where(
                and_(
                    Message.was_faq_match == False,
                    Message.role == MessageRole.ASSISTANT
                )
            )
        )
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def get_average_response_time(self) -> Optional[float]:
        """Get average response time in ms"""
        query = (
            select(func.avg(Message.response_time_ms))
            .where(Message.response_time_ms != None)
        )
        result = await self.session.execute(query)
        return result.scalar()
