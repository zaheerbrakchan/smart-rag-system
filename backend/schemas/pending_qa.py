"""
Pending Q&A Schemas (DTOs)
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============== REQUEST SCHEMAS ==============

class PendingQACreate(BaseModel):
    """Create pending Q&A entry (auto or manual)"""
    question: str = Field(..., min_length=5)
    original_answer: str = Field(..., min_length=10)
    source_conversation_id: Optional[int] = None
    source_documents: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    detected_state: Optional[str] = None
    detected_exam: Optional[str] = None
    detected_category: Optional[str] = None
    original_confidence: Optional[float] = None


class PendingQAReview(BaseModel):
    """Admin review action on pending Q&A"""
    action: str = Field(..., pattern="^(approve|reject|modify)$")
    modified_answer: Optional[str] = None  # Required if action is 'modify'
    review_notes: Optional[str] = None
    # Optional: override detected metadata
    state: Optional[str] = None
    exam: Optional[str] = None
    category: Optional[str] = None


class BulkFAQUpload(BaseModel):
    """Bulk FAQ upload"""
    faqs: List[Dict[str, Any]] = Field(
        ...,
        description="List of {question, answer, state?, exam?, category?}"
    )
    auto_approve: bool = Field(
        default=False,
        description="If true, skip pending queue and add directly to FAQ"
    )


# ============== RESPONSE SCHEMAS ==============

class PendingQAResponse(BaseModel):
    """Pending Q&A response"""
    id: int
    question: str
    original_answer: str
    modified_answer: Optional[str] = None
    source_conversation_id: Optional[int] = None
    source_documents: List[Dict[str, Any]] = []
    detected_state: Optional[str] = None
    detected_exam: Optional[str] = None
    detected_category: Optional[str] = None
    original_confidence: Optional[float] = None
    occurrence_count: int = 1
    status: str
    reviewed_by: Optional[int] = None
    review_notes: Optional[str] = None
    faq_vector_id: Optional[str] = None
    created_at: datetime
    reviewed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class PendingQAListResponse(BaseModel):
    """Paginated pending Q&A list"""
    items: List[PendingQAResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    # Stats
    pending_count: int = 0
    approved_count: int = 0
    rejected_count: int = 0


class FAQStats(BaseModel):
    """FAQ system statistics"""
    total_faqs: int
    pending_reviews: int
    approved_today: int
    auto_learned_count: int
    faq_hit_rate: float = Field(description="Percentage of queries answered by FAQ")
