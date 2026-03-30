"""
Base Repository with common CRUD operations
Similar to JpaRepository<T, ID> in Spring Data JPA
"""

from typing import TypeVar, Generic, Optional, List, Type, Any
from sqlalchemy import select, func, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import Base

# Generic type for entity
T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T]):
    """
    Base repository with common CRUD operations.
    
    Usage:
        class UserRepository(BaseRepository[User]):
            def __init__(self, session: AsyncSession):
                super().__init__(User, session)
    """
    
    def __init__(self, model: Type[T], session: AsyncSession):
        self.model = model
        self.session = session
    
    async def create(self, entity: T) -> T:
        """Create new entity (like save() in JPA)"""
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity
    
    async def get_by_id(self, id: int) -> Optional[T]:
        """Find by ID (like findById() in JPA)"""
        return await self.session.get(self.model, id)
    
    async def get_all(
        self, 
        skip: int = 0, 
        limit: int = 100,
        order_by: Any = None
    ) -> List[T]:
        """Get all with pagination (like findAll(Pageable) in JPA)"""
        query = select(self.model)
        if order_by is not None:
            query = query.order_by(order_by)
        query = query.offset(skip).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def count(self) -> int:
        """Count all entities"""
        query = select(func.count()).select_from(self.model)
        result = await self.session.execute(query)
        return result.scalar() or 0
    
    async def update(self, entity: T) -> T:
        """Update entity (like save() on existing entity in JPA)"""
        await self.session.flush()
        await self.session.refresh(entity)
        return entity
    
    async def delete(self, entity: T) -> None:
        """Delete entity"""
        await self.session.delete(entity)
        await self.session.flush()
    
    async def delete_by_id(self, id: int) -> bool:
        """Delete by ID"""
        entity = await self.get_by_id(id)
        if entity:
            await self.delete(entity)
            return True
        return False
    
    async def exists_by_id(self, id: int) -> bool:
        """Check if entity exists"""
        entity = await self.get_by_id(id)
        return entity is not None
    
    async def bulk_create(self, entities: List[T]) -> List[T]:
        """Bulk insert entities"""
        self.session.add_all(entities)
        await self.session.flush()
        for entity in entities:
            await self.session.refresh(entity)
        return entities
