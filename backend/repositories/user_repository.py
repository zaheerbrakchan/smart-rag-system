"""
User Repository
Data access layer for User entity
"""

from typing import Optional, List
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User, UserRole
from .base_repository import BaseRepository


class UserRepository(BaseRepository[User]):
    """
    User repository with custom query methods.
    Similar to interface UserRepository extends JpaRepository<User, Long> in Spring
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(User, session)
    
    async def find_by_normalized_phone(self, phone: str) -> Optional[User]:
        """Find user by phone, comparing E.164-normalized values."""
        from services.otp_service import OTPService

        target = OTPService._format_phone(phone)
        query = select(User).where(User.phone.isnot(None))
        result = await self.session.execute(query)
        for user in result.scalars().all():
            try:
                if OTPService._format_phone(user.phone) == target:
                    return user
            except Exception:
                continue
        return None
    
    async def find_by_role(
        self, 
        role: UserRole, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[User]:
        """Find all users by role"""
        query = (
            select(User)
            .where(User.role == role)
            .order_by(User.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def find_active_users(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """Find all active users"""
        query = (
            select(User)
            .where(User.is_active == True)
            .order_by(User.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def count_by_role(self, role: UserRole) -> int:
        """Count users by role"""
        query = select(func.count()).select_from(User).where(User.role == role)
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def update_last_login(self, user_id: int) -> None:
        """Update last login timestamp"""
        user = await self.get_by_id(user_id)
        if user:
            user.last_login_at = datetime.utcnow()
            await self.session.flush()
    
    async def search_users(
        self,
        query_str: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """Search users by name or phone."""
        search = f"%{query_str.lower()}%"
        query = (
            select(User)
            .where(
                (User.full_name.ilike(search)) | (User.phone.ilike(search))
            )
            .order_by(User.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def find_by_target_exam(
        self,
        exam: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """Find users targeting specific exam (JSONB query)"""
        query = (
            select(User)
            .where(User.target_exams.contains([exam]))
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
