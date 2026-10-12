"""Global minimum spacing between sync starts, shared by all API replicas."""

import logging

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.core.config import SYNC_RATE_LIMIT_SECONDS
from src.database.connection import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def enforce_sync_rate_limit() -> None:
    # The conditional upsert serializes competing admissions, including the
    # first requests when no row exists. Use the database clock on all hosts.
    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                admitted = await session.scalar(text("""
                    INSERT INTO sync_rate_limits (scope, next_allowed_at)
                    VALUES ('extract_sync', clock_timestamp() +
                        make_interval(secs => CAST(:seconds AS double precision)))
                    ON CONFLICT (scope) DO UPDATE
                    SET next_allowed_at = clock_timestamp() +
                        make_interval(secs => CAST(:seconds AS double precision))
                    WHERE sync_rate_limits.next_allowed_at <= clock_timestamp()
                    RETURNING next_allowed_at
                """), {"seconds": SYNC_RATE_LIMIT_SECONDS})
                retry_after = None
                if admitted is None:
                    retry_after = await session.scalar(text("""
                        SELECT GREATEST(1, CEIL(EXTRACT(EPOCH FROM
                            (next_allowed_at - clock_timestamp()))))::integer
                        FROM sync_rate_limits WHERE scope = 'extract_sync'
                    """))
        # Finish the transaction before accepting or rejecting the request.
        if admitted is None:
            raise HTTPException(
                status_code=429,
                detail="Sync extraction rate limit reached. Retry after the indicated delay.",
                headers={"Retry-After": str(retry_after or SYNC_RATE_LIMIT_SECONDS)},
            )
    except (SQLAlchemyError, OSError) as error:
        logger.exception("Sync rate limiter unavailable")
        # Never bypass the limit when its shared state is unavailable.
        raise HTTPException(
            status_code=503,
            detail="Sync extraction temporarily unavailable.",
            headers={"Retry-After": "5"},
        ) from error
