"""
Conversation Routes
Manages user chat history and conversation contexts
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from pydantic import BaseModel

from database.connection import get_db
from models.user import User
from models.conversation import Conversation, Message, MessageRole
from dependencies.auth import get_current_user

router = APIRouter(prefix="/conversations", tags=["Conversations"])
logger = logging.getLogger(__name__)


# ============== SCHEMAS ==============

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sources: Optional[List[dict]]
    filters_applied: Optional[dict]
    was_faq_match: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class ConversationResponse(BaseModel):
    id: int
    title: Optional[str]
    summary: Optional[str]
    message_count: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ConversationDetailResponse(BaseModel):
    id: int
    title: Optional[str]
    summary: Optional[str]
    messages: List[MessageResponse]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    conversations: List[ConversationResponse]
    total: int
    page: int
    page_size: int


class CreateConversationRequest(BaseModel):
    title: Optional[str] = None


class UpdateConversationRequest(BaseModel):
    title: Optional[str] = None


# ============== ROUTES ==============

@router.post("/", response_model=ConversationResponse)
async def create_conversation(
    request: CreateConversationRequest = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new conversation"""
    conversation = Conversation(
        user_id=current_user.id,
        title=request.title if request else None
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        summary=conversation.summary,
        message_count=0,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


@router.get("/", response_model=ConversationListResponse)
async def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List user's conversations"""

    # Fetch only lightweight columns needed by sidebar (avoid loading heavy JSON fields).
    offset = (page - 1) * page_size
    query = (
        select(
            Conversation.id,
            Conversation.title,
            Conversation.summary,
            Conversation.created_at,
            Conversation.updated_at,
        )
        .where(Conversation.user_id == current_user.id)
        .order_by(desc(Conversation.updated_at))
        .offset(offset)
        .limit(page_size)
    )

    try:
        # Keep sidebar responsive; fail fast instead of hanging spinner.
        result = await asyncio.wait_for(db.execute(query), timeout=6.0)
        rows = result.all()
    except Exception as exc:
        logger.warning("list_conversations query timeout/failure user_id=%s: %s", current_user.id, exc)
        return ConversationListResponse(
            conversations=[],
            total=0,
            page=page,
            page_size=page_size,
        )

    conv_ids = [row.id for row in rows]
    count_map = {}
    if conv_ids:
        try:
            counts_result = await asyncio.wait_for(
                db.execute(
                    select(
                        Message.conversation_id,
                        func.count(Message.id)
                    )
                    .where(Message.conversation_id.in_(conv_ids))
                    .group_by(Message.conversation_id)
                ),
                timeout=4.0,
            )
            count_map = {conversation_id: count for conversation_id, count in counts_result.all()}
        except Exception as exc:
            logger.warning("list_conversations message_count timeout/failure user_id=%s: %s", current_user.id, exc)
            count_map = {}

    # Best-effort total count: don't block sidebar on this.
    total = 0
    try:
        total_query = select(func.count(Conversation.id)).where(
            Conversation.user_id == current_user.id
        )
        total = int(await asyncio.wait_for(db.scalar(total_query), timeout=3.0) or 0)
    except Exception as exc:
        logger.warning("list_conversations total_count timeout/failure user_id=%s: %s", current_user.id, exc)
        total = offset + len(rows)

    conv_responses = [
        ConversationResponse(
            id=row.id,
            title=row.title,
            summary=row.summary,
            message_count=int(count_map.get(row.id, 0) or 0),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row in rows
    ]

    return ConversationListResponse(
        conversations=conv_responses,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get conversation with all messages"""
    conversation = await db.get(Conversation, conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get messages
    query = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    result = await db.execute(query)
    messages = result.scalars().all()
    
    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        summary=conversation.summary,
        messages=[MessageResponse(
            id=m.id,
            role=m.role.value,
            content=m.content,
            sources=m.sources,
            filters_applied=m.filters_applied,
            was_faq_match=m.was_faq_match or False,
            created_at=m.created_at
        ) for m in messages],
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: int,
    request: UpdateConversationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update conversation title"""
    conversation = await db.get(Conversation, conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if request.title is not None:
        conversation.title = request.title
    
    await db.commit()
    await db.refresh(conversation)
    
    msg_count = await db.scalar(
        select(func.count(Message.id)).where(Message.conversation_id == conversation.id)
    )
    
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        summary=conversation.summary,
        message_count=msg_count or 0,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a conversation and all its messages"""
    conversation = await db.get(Conversation, conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    await db.delete(conversation)
    await db.commit()
    
    return {"success": True, "deleted_id": conversation_id}


# ============== HELPER: Save message to conversation ==============

async def save_message_to_conversation(
    db: AsyncSession,
    conversation_id: int,
    role: str,
    content: str,
    sources: Optional[List[dict]] = None,
    filters_applied: Optional[dict] = None,
    was_faq_match: bool = False,
    response_time_ms: Optional[int] = None
) -> Message:
    """Helper to save a message to a conversation"""
    message = Message(
        conversation_id=conversation_id,
        role=MessageRole(role),
        content=content,
        sources=sources,
        filters_applied=filters_applied,
        was_faq_match=was_faq_match,
        response_time_ms=response_time_ms
    )
    db.add(message)
    
    # Update conversation's updated_at
    conversation = await db.get(Conversation, conversation_id)
    if conversation:
        conversation.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(message)
    
    return message
