"""Database ownership, retry scheduling and recovery (no OCR imports)."""

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError

from src.core.config import JOB_LEASE_SECONDS, MAX_JOB_RETRIES
from src.database.connection import AsyncSessionLocal
from src.database.models import ExtractionRequest, ExtractionStatus

logger = logging.getLogger(__name__)
PROCESSING = (ExtractionStatus.OCR_PROCESSING, ExtractionStatus.LM_PROCESSING)


def utc_now():
    return datetime.now(timezone.utc)


def retry_delay(retry_count):
    return min(10 * 2 ** min(max(0, retry_count - 1), 5), 300)


def assign_lease(job):
    job.lease_token = str(uuid4())
    job.lease_expires_at = utc_now() + timedelta(seconds=JOB_LEASE_SECONDS)
    job.next_retry_at = None
    return job.lease_token


def clear_lease(job):
    job.lease_token = None
    job.lease_expires_at = None


def owns_lease(job, token, status):
    return (
        job.status == status
        and job.lease_token == token
        and job.lease_expires_at is not None
        and job.lease_expires_at > utc_now()
    )


def record_failure(job, error_msg, fallback_status):
    now = utc_now()
    job.retry_count = min(job.retry_count + 1, MAX_JOB_RETRIES)
    job.last_error_message = error_msg
    job.updated_at = now
    clear_lease(job)
    if not job.queue_managed or job.retry_count >= MAX_JOB_RETRIES:
        job.status = ExtractionStatus.FAILED
        job.next_retry_at = None
    else:
        job.status = fallback_status
        job.next_retry_at = now + timedelta(seconds=retry_delay(job.retry_count))


async def renew_lease(job_id, token):
    now = utc_now()
    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                update(ExtractionRequest)
                .where(
                    ExtractionRequest.id == job_id,
                    ExtractionRequest.lease_token == token,
                    ExtractionRequest.status.in_(PROCESSING),
                    ExtractionRequest.lease_expires_at > now,
                )
                # Heartbeats must not change queue ordering or updated_at.
                .values(
                    lease_expires_at=now + timedelta(seconds=JOB_LEASE_SECONDS),
                    updated_at=ExtractionRequest.updated_at,
                )
            )
            return result.rowcount == 1


async def heartbeat(job_id, token):
    while True:
        await asyncio.sleep(JOB_LEASE_SECONDS / 3)
        try:
            if not await renew_lease(job_id, token):
                return
        except (SQLAlchemyError, OSError):
            logger.exception("Could not renew lease for job %s", job_id)


async def stop_heartbeat(task):
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def mark_job_failed(job_id, error_msg, fallback_status, token):
    expected = (ExtractionStatus.OCR_PROCESSING
                if fallback_status == ExtractionStatus.UPLOADED
                else ExtractionStatus.LM_PROCESSING)
    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                job = (await session.execute(
                    select(ExtractionRequest)
                    .where(ExtractionRequest.id == job_id)
                    .with_for_update()
                )).scalar_one_or_none()
                if job is not None and owns_lease(job, token, expected):
                    record_failure(job, error_msg, fallback_status)
    except (SQLAlchemyError, OSError):
        # Recovery will retry after the outstanding lease expires.
        logger.exception("Could not record failure for job %s", job_id)


async def recover_expired_jobs(limit=100):
    """Safe across schedulers: lock rows and invalidate old ownership tokens."""
    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                jobs = (await session.execute(
                    select(ExtractionRequest)
                    .where(
                        ExtractionRequest.status.in_(PROCESSING),
                        ExtractionRequest.lease_expires_at <= utc_now(),
                    )
                    .order_by(ExtractionRequest.lease_expires_at, ExtractionRequest.id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )).scalars().all()
                for job in jobs:
                    fallback = (ExtractionStatus.UPLOADED
                                if job.status == ExtractionStatus.OCR_PROCESSING
                                else ExtractionStatus.OCR_COMPLETED)
                    record_failure(job, "Processing lease expired; worker interrupted", fallback)
                return len(jobs)
    except (SQLAlchemyError, OSError):
        logger.exception("Could not recover expired jobs")
        return 0
