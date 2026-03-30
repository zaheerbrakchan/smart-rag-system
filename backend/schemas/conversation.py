"""
Conversation & Message Schemas (DTOs)
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============== REQUEST SCHEMAS ==============

class ConversationCreate(BaseModel):
    """Create new conversation"""
    title: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MessageCreate(BaseModel):
    """Create new message in conversation"""
    content: str = Field(..., min_length=1, max_length=10000)


# ============== RESPONSE SCHEMAS ==============

class MessageResponse(BaseModel):
    """Message response"""
    id: int
    conversation_id: int
    role: str
    content: str
    sources: List[Dict[str, Any]] = []
    model_used: Optional[str] = None
    filters_applied: Dict[str, Any] = {}
    response_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    was_faq_match: bool = False
    faq_confidence: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class ConversationResponse(BaseModel):
    """Conversation response"""
    id: int
    user_id: int
    title: Optional[str] = None
    summary: Optional[str] = None
    context_data: Dict[str, Any] = {}
    is_active: bool
    message_count: int
    created_at: datetime
    updated_at: datetime
    messages: Optional[List[MessageResponse]] = None
    
    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    """Paginated conversation list"""
    conversations: List[ConversationResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class ConversationSummary(BaseModel):
    """Lightweight conversation summary for list views"""
    id: int
    title: Optional[str] = None
    message_count: int
    last_message_preview: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True
