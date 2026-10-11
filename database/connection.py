from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from src.core.config import (
    DATABASE_URL,
    DB_CONNECT_TIMEOUT_SECONDS,
    DB_ECHO,
    DB_MAX_OVERFLOW,
    DB_POOL_SIZE,
)

async_engine = create_async_engine(
    DATABASE_URL,
    echo=DB_ECHO,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    connect_args={"timeout": DB_CONNECT_TIMEOUT_SECONDS},
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False, # Crucial for async sqlalchemy
    autocommit=False,
    autoflush=False,
)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
