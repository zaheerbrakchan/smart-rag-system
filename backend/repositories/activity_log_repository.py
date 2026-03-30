"""
Activity Log Repository
Data access layer for ActivityLog entity
"""

from typing import Optional, List
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from models.activity_log import ActivityLog, ActionType
from .base_repository import BaseRepository


class ActivityLogRepository(BaseRepository[ActivityLog]):
    """
    Activity Log repository with analytics queries
    """
    
    def __init__(self, session: AsyncSession):
        super().__init__(ActivityLog, session)
    
    async def log_action(
        self,
        action_type: ActionType,
        description: str,
        user_id: Optional[int] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_data: Optional[dict] = None,
        response_data: Optional[dict] = None,
        success: bool = True,
        error_details: Optional[dict] = None
    ) -> ActivityLog:
        """Create a new activity log entry"""
        log = ActivityLog(
            user_id=user_id,
            action_type=action_type,
            description=description,
            target_type=target_type,
            target_id=target_id,
            ip_address=ip_address,
            user_agent=user_agent,
            request_data=request_data or {},
            response_data=response_data or {},
            success=success,
            error_details=error_details or {}
        )
        return await self.create(log)
    
    async def find_by_user_id(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[ActivityLog]:
        """Find logs for a specific user"""
        query = (
            select(ActivityLog)
            .where(ActivityLog.user_id == user_id)
            .order_by(ActivityLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def find_by_action_type(
        self,
        action_type: ActionType,
        skip: int = 0,
        limit: int = 100
    ) -> List[ActivityLog]:
        """Find logs by action type"""
        query = (
            select(ActivityLog)
            .where(ActivityLog.action_type == action_type)
            .order_by(ActivityLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def find_recent(
        self,
        hours: int = 24,
        skip: int = 0,
        limit: int = 100
    ) -> List[ActivityLog]:
        """Find logs from last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        query = (
            select(ActivityLog)
            .where(ActivityLog.created_at >= cutoff)
            .order_by(ActivityLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def find_failed_actions(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[ActivityLog]:
        """Find failed actions for debugging"""
        query = (
            select(ActivityLog)
            .where(ActivityLog.success == False)
            .order_by(ActivityLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def count_by_action_type(
        self,
        action_type: ActionType,
        since: Optional[datetime] = None
    ) -> int:
        """Count actions of specific type"""
        query = (
            select(func.count())
            .select_from(ActivityLog)
            .where(ActivityLog.action_type == action_type)
        )
        if since:
            query = query.where(ActivityLog.created_at >= since)
        
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def get_daily_stats(self, days: int = 7) -> List[dict]:
        """Get action counts per day for last N days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = (
            select(
                func.date(ActivityLog.created_at).label('date'),
                func.count().label('count')
            )
            .where(ActivityLog.created_at >= cutoff)
            .group_by(func.date(ActivityLog.created_at))
            .order_by(func.date(ActivityLog.created_at))
        )
        result = await self.session.execute(query)
        return [{"date": str(row.date), "count": row.count} for row in result]
    
    async def get_action_breakdown(
        self,
        since: Optional[datetime] = None
    ) -> dict:
        """Get count breakdown by action type"""
        query = (
            select(
                ActivityLog.action_type,
                func.count().label('count')
            )
            .group_by(ActivityLog.action_type)
        )
        if since:
            query = query.where(ActivityLog.created_at >= since)
        
        result = await self.session.execute(query)
        return {row.action_type.value: row.count for row in result}
    
    async def find_by_target(
        self,
        target_type: str,
        target_id: str,
        skip: int = 0,
        limit: int = 50
    ) -> List[ActivityLog]:
        """Find logs for a specific target entity"""
        query = (
            select(ActivityLog)
            .where(
                and_(
                    ActivityLog.target_type == target_type,
                    ActivityLog.target_id == target_id
                )
            )
            .order_by(ActivityLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
