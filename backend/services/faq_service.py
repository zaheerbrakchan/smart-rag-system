"""
FAQ Service
Business logic for FAQ management and semantic search
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from models.faq import FAQ
from repositories.faq_repository import FAQRepository


# Minimum similarity score to consider a FAQ match
FAQ_MATCH_THRESHOLD = float(os.getenv("FAQ_SCORE_THRESHOLD", "0.95"))


@dataclass
class FAQMatch:
    """Result of FAQ semantic search"""
    faq: FAQ
    score: float
    is_confident: bool  # True if score is above threshold


class FAQService:
    """
    FAQ Service for managing and searching FAQs
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = FAQRepository(session)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    async def search_faqs(
        self, 
        query: str, 
        state: Optional[str] = None,
        top_k: int = 3
    ) -> List[FAQMatch]:
        """
        Search FAQs using semantic similarity.
        Returns matches sorted by score, with confidence flag.
        """
        # Generate embedding for the query
        query_embedding = self._generate_embedding(query)
        
        # Get all FAQs with embeddings (optionally filtered by state)
        all_faqs = await self.repository.find_all_with_embeddings()
        
        # Filter by state if provided (include global FAQs)
        if state:
            all_faqs = [
                faq for faq in all_faqs 
                if faq.state is None or faq.state == state
            ]
        
        # Calculate similarity scores
        matches = []
        for faq in all_faqs:
            if faq.embedding:
                score = self._cosine_similarity(query_embedding, faq.embedding)
                matches.append(FAQMatch(
                    faq=faq,
                    score=score,
                    is_confident=score >= FAQ_MATCH_THRESHOLD
                ))
        
        # Sort by score descending
        matches.sort(key=lambda x: x.score, reverse=True)
        
        return matches[:top_k]
    
    async def find_best_match(
        self, 
        query: str, 
        state: Optional[str] = None
    ) -> Optional[FAQMatch]:
        """
        Find the best FAQ match for a query.
        Returns None if no confident match found.
        """
        matches = await self.search_faqs(query, state, top_k=1)
        
        if matches and matches[0].is_confident:
            # Increment view count for the matched FAQ
            await self.repository.increment_view_count(matches[0].faq.id)
            return matches[0]
        
        return None
    
    async def create_faq(
        self,
        question: str,
        answer: str,
        category: Optional[str] = None,
        keywords: Optional[str] = None,
        state: Optional[str] = None
    ) -> FAQ:
        """
        Create a new FAQ with auto-generated embedding
        """
        # Generate embedding for the question
        embedding = self._generate_embedding(question)
        
        faq = FAQ(
            question=question,
            answer=answer,
            category=category,
            keywords=keywords,
            state=state,
            embedding=embedding,
            is_active=True
        )
        
        created = await self.repository.create(faq)
        return created
    
    async def update_faq(
        self,
        faq_id: int,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        category: Optional[str] = None,
        keywords: Optional[str] = None,
        state: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> Optional[FAQ]:
        """
        Update an existing FAQ. Re-generates embedding if question changes.
        """
        faq = await self.repository.find_by_id(faq_id)
        if not faq:
            return None
        
        # Update fields
        if question is not None:
            faq.question = question
            # Re-generate embedding since question changed
            faq.embedding = self._generate_embedding(question)
        
        if answer is not None:
            faq.answer = answer
        if category is not None:
            faq.category = category
        if keywords is not None:
            faq.keywords = keywords
        if state is not None:
            faq.state = state
        if is_active is not None:
            faq.is_active = is_active
        
        updated = await self.repository.update(faq)
        return updated
    
    async def delete_faq(self, faq_id: int) -> bool:
        """Delete a FAQ"""
        return await self.repository.delete(faq_id)
    
    async def get_all_faqs(
        self, 
        skip: int = 0, 
        limit: int = 100,
        active_only: bool = True
    ) -> List[FAQ]:
        """Get all FAQs"""
        if active_only:
            return await self.repository.find_active(skip, limit)
        return await self.repository.find_all(skip, limit)
    
    async def get_faq_by_id(self, faq_id: int) -> Optional[FAQ]:
        """Get FAQ by ID"""
        return await self.repository.find_by_id(faq_id)
    
    async def regenerate_all_embeddings(self) -> int:
        """
        Regenerate embeddings for all FAQs.
        Useful when switching embedding models.
        """
        all_faqs = await self.repository.find_all()
        count = 0
        
        for faq in all_faqs:
            faq.embedding = self._generate_embedding(faq.question)
            await self.repository.update(faq)
            count += 1
        
        return count
    
    async def get_stats(self) -> dict:
        """Get FAQ statistics"""
        active_count = await self.repository.count_active()
        all_faqs = await self.repository.find_all()
        
        total_views = sum(faq.view_count for faq in all_faqs)
        
        return {
            "total_faqs": len(all_faqs),
            "active_faqs": active_count,
            "total_views": total_views
        }
