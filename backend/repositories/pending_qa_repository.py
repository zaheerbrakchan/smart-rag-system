"""
Pending Q&A Repository
Data access layer for PendingQA entity
"""

from typing import Optional, List
from datetime import datetime
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from models.pending_qa import PendingQA, QAStatus
from .base_repository import BaseRepository


class PendingQARepository(BaseRepository[PendingQA]):
    """
    Pending Q&A repository with custom query methods
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(PendingQA, session)
    
    async def find_by_status(
        self,
        status: QAStatus,
        skip: int = 0,
        limit: int = 50
    ) -> List[PendingQA]:
        """Find Q&A entries by status"""
        query = (
            select(PendingQA)
            .where(PendingQA.status == status)
            .order_by(PendingQA.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def find_pending(
        self,
        skip: int = 0,
        limit: int = 50
    ) -> List[PendingQA]:
        """Find all pending Q&A entries"""
        return await self.find_by_status(QAStatus.PENDING, skip, limit)
    
    async def count_by_status(self, status: QAStatus) -> int:
        """Count Q&A entries by status"""
        query = (
            select(func.count())
            .select_from(PendingQA)
            .where(PendingQA.status == status)
        )
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def find_by_state(
        self,
        state: str,
        status: Optional[QAStatus] = None,
        skip: int = 0,
        limit: int = 50
    ) -> List[PendingQA]:
        """Find Q&A entries by detected state"""
        query = select(PendingQA).where(PendingQA.detected_state == state)
        if status:
            query = query.where(PendingQA.status == status)
        query = query.order_by(PendingQA.created_at.desc()).offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def find_by_exam(
        self,
        exam: str,
        status: Optional[QAStatus] = None,
        skip: int = 0,
        limit: int = 50
    ) -> List[PendingQA]:
        """Find Q&A entries by detected exam"""
        query = select(PendingQA).where(PendingQA.detected_exam == exam)
        if status:
            query = query.where(PendingQA.status == status)
        query = query.order_by(PendingQA.created_at.desc()).offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def find_similar_question(
        self,
        question: str
    ) -> Optional[PendingQA]:
        """Find existing Q&A with similar question (exact match for now)"""
        # Note: For production, use vector similarity search
        query = select(PendingQA).where(PendingQA.question == question)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def increment_occurrence(self, qa_id: int) -> None:
        """Increment occurrence count for existing Q&A"""
        qa = await self.get_by_id(qa_id)
        if qa:
            qa.occurrence_count += 1
            await self.session.flush()
    
    async def approve(
        self,
        qa_id: int,
        reviewer_id: int,
        modified_answer: Optional[str] = None,
        notes: Optional[str] = None,
        faq_vector_id: Optional[str] = None
    ) -> Optional[PendingQA]:
        """Approve a pending Q&A entry"""
        qa = await self.get_by_id(qa_id)
        if qa:
            qa.status = QAStatus.MODIFIED if modified_answer else QAStatus.APPROVED
            qa.reviewed_by = reviewer_id
            qa.reviewed_at = datetime.utcnow()
            qa.review_notes = notes
            qa.faq_vector_id = faq_vector_id
            if modified_answer:
                qa.modified_answer = modified_answer
            await self.session.flush()
        return qa
    
    async def reject(
        self,
        qa_id: int,
        reviewer_id: int,
        notes: Optional[str] = None
    ) -> Optional[PendingQA]:
        """Reject a pending Q&A entry"""
        qa = await self.get_by_id(qa_id)
        if qa:
            qa.status = QAStatus.REJECTED
            qa.reviewed_by = reviewer_id
            qa.reviewed_at = datetime.utcnow()
            qa.review_notes = notes
            await self.session.flush()
        return qa
    
    async def get_stats(self) -> dict:
        """Get Q&A statistics"""
        pending = await self.count_by_status(QAStatus.PENDING)
        approved = await self.count_by_status(QAStatus.APPROVED)
        rejected = await self.count_by_status(QAStatus.REJECTED)
        modified = await self.count_by_status(QAStatus.MODIFIED)
        
        return {
            "pending": pending,
            "approved": approved + modified,
            "rejected": rejected,
            "total": pending + approved + rejected + modified
        }
    
    async def find_high_occurrence(
        self,
        min_count: int = 5,
        status: QAStatus = QAStatus.PENDING,
        limit: int = 20
    ) -> List[PendingQA]:
        """Find frequently asked questions for priority review"""
        query = (
            select(PendingQA)
            .where(
                and_(
                    PendingQA.status == status,
                    PendingQA.occurrence_count >= min_count
                )
            )
            .order_by(PendingQA.occurrence_count.desc())
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
