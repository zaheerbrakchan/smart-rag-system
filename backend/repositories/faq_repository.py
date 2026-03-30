"""
FAQ Repository
Data access layer for FAQ entity
"""

from typing import Optional, List
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.faq import FAQ
from .base_repository import BaseRepository


class FAQRepository(BaseRepository[FAQ]):
    """
    FAQ repository with custom query methods
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(FAQ, session)
    
    async def find_active(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[FAQ]:
        """Find all active FAQs"""
        query = (
            select(FAQ)
            .where(FAQ.is_active == True)
            .order_by(FAQ.view_count.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def find_by_category(
        self,
        category: str,
        skip: int = 0,
        limit: int = 50
    ) -> List[FAQ]:
        """Find FAQs by category"""
        query = (
            select(FAQ)
            .where(FAQ.is_active == True)
            .where(FAQ.category == category)
            .order_by(FAQ.view_count.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def find_by_state(
        self,
        state: Optional[str],
        skip: int = 0,
        limit: int = 50
    ) -> List[FAQ]:
        """Find FAQs applicable to a state (including global FAQs)"""
        query = (
            select(FAQ)
            .where(FAQ.is_active == True)
            .where((FAQ.state == state) | (FAQ.state == None))
            .order_by(FAQ.view_count.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def find_all_with_embeddings(self) -> List[FAQ]:
        """Find all active FAQs that have embeddings"""
        query = (
            select(FAQ)
            .where(FAQ.is_active == True)
            .where(FAQ.embedding != None)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def increment_view_count(self, faq_id: int) -> None:
        """Increment view count for a FAQ"""
        query = (
            update(FAQ)
            .where(FAQ.id == faq_id)
            .values(view_count=FAQ.view_count + 1)
        )
        await self.session.execute(query)
        await self.session.commit()
    
    async def search_by_keywords(
        self,
        search_term: str,
        limit: int = 10
    ) -> List[FAQ]:
        """Simple keyword search in question and keywords field"""
        search_pattern = f"%{search_term}%"
        query = (
            select(FAQ)
            .where(FAQ.is_active == True)
            .where(
                (FAQ.question.ilike(search_pattern)) |
                (FAQ.keywords.ilike(search_pattern))
            )
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def count_active(self) -> int:
        """Count active FAQs"""
        query = (
            select(func.count())
            .select_from(FAQ)
            .where(FAQ.is_active == True)
        )
        result = await self.session.execute(query)
        return result.scalar() or 0
