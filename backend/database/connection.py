"""
Database Connection Configuration
Production-grade async PostgreSQL connection with SQLAlchemy 2.0
"""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import AsyncAdaptedQueuePool
from dotenv import load_dotenv

load_dotenv()

# Database URL - supports both sync and async
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://postgres:postgres@localhost:5432/neet_assistant"
)

# For Alembic (sync operations)
SYNC_DATABASE_URL = DATABASE_URL.replace("+asyncpg", "")

# Create async engine with connection pooling for better performance
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("DEBUG", "false").lower() == "true",  # SQL logging in debug mode
    poolclass=AsyncAdaptedQueuePool,  # Connection pool for better performance
    pool_size=5,  # Number of connections to keep open
    max_overflow=10,  # Additional connections when pool is exhausted
    pool_timeout=30,  # Wait time for available connection
    pool_recycle=300,  # Recycle connections after 5 minutes
    pool_pre_ping=True,  # Verify connection before using
    future=True,
    connect_args={
        "timeout": 30,  # Connection timeout in seconds
        "command_timeout": 30  # Query timeout
    }
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


# Base class for all models
class Base(DeclarativeBase):
    pass


# Dependency for FastAPI
async def get_db() -> AsyncSession:
    """
    Dependency that provides a database session.
    Usage in FastAPI:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections"""
    await engine.dispose()
