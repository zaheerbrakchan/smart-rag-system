"""
Database Connection Configuration
Production-grade async PostgreSQL connection with SQLAlchemy 2.0
"""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import AsyncAdaptedQueuePool
from dotenv import load_dotenv

from services.db_url import normalize_async_database_url

load_dotenv()

_raw_db_url = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/neet_assistant",
)
# Database URL — normalize to asyncpg and remove unsupported query args.
DATABASE_URL = normalize_async_database_url(_raw_db_url)


def _asyncpg_connect_args() -> dict:
    args: dict = {"timeout": 30, "command_timeout": 30}
    args["server_settings"] = {"hnsw.ef_search": "40"}
    # Keep SSL mode explicit via env; for VPS installs this is often "disable".
    ssl_mode = os.getenv("DB_SSL_MODE", "disable").strip().lower()
    if ssl_mode in {"disable", "false", "0", "off"}:
        pass
    elif ssl_mode in {"prefer", "allow", "require", "verify-ca", "verify-full"}:
        args["ssl"] = ssl_mode
    else:
        args["ssl"] = "disable"
    return args

# For Alembic (sync operations)
SYNC_DATABASE_URL = DATABASE_URL.replace("+asyncpg", "")

# Create async engine with connection pooling for better performance
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("DEBUG", "false").lower() == "true",  # SQL logging in debug mode
    poolclass=AsyncAdaptedQueuePool,  # Connection pool for better performance
    pool_size=10,  # Number of connections to keep open
    max_overflow=20,  # Additional connections when pool is exhausted
    pool_timeout=30,  # Wait time for available connection
    pool_recycle=300,  # Recycle connections after 5 minutes
    pool_pre_ping=True,  # Verify connection before using
    future=True,
    connect_args=_asyncpg_connect_args(),
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
            # Must commit after yield: flushed INSERT/UPDATE are no longer in
            # session.new, so a "only if new/dirty" check can skip commit and
            # roll back writes (e.g. OTP rows never persisted -> "Invalid OTP").
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database tables (import package so every model registers on Base.metadata)."""
    import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections"""
    await engine.dispose()
