import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api import router
from src.database.connection import get_db_session
from src.service import sync_rate_limit as limiter


@pytest.mark.asyncio
async def test_concurrent_admission_is_global_across_sessions(db, monkeypatch):
    monkeypatch.setattr(limiter, 'SYNC_RATE_LIMIT_SECONDS', 180)
    results = await asyncio.gather(*[
        limiter.enforce_sync_rate_limit() for _ in range(10)
    ], return_exceptions=True)
    assert sum(r is None for r in results) == 1
    rejected = [r for r in results if isinstance(r, HTTPException)]
    assert len(rejected) == 9
    assert all(r.status_code == 429 and 1 <= int(r.headers['Retry-After']) <= 180 for r in rejected)
    # A new session factory (as in another replica/restart) shares the budget.
    monkeypatch.setattr(limiter, 'AsyncSessionLocal', async_sessionmaker(db.kw['bind']))
    with pytest.raises(HTTPException) as error:
        await limiter.enforce_sync_rate_limit()
    assert error.value.status_code == 429


@pytest.mark.asyncio
async def test_cooldown_is_not_extended_by_rejections_and_expires(db, monkeypatch):
    monkeypatch.setattr(limiter, 'SYNC_RATE_LIMIT_SECONDS', 7)
    await limiter.enforce_sync_rate_limit()
    async with db() as s:
        before = await s.scalar(text('SELECT next_allowed_at FROM sync_rate_limits'))
    with pytest.raises(HTTPException) as error:
        await limiter.enforce_sync_rate_limit()
    assert 1 <= int(error.value.headers['Retry-After']) <= 7
    async with db() as s:
        assert await s.scalar(text('SELECT next_allowed_at FROM sync_rate_limits')) == before
        await s.execute(text("UPDATE sync_rate_limits SET next_allowed_at = clock_timestamp() - interval '1 second'"))
        await s.commit()
    await limiter.enforce_sync_rate_limit()
    async with db() as s:
        remaining = await s.scalar(text('SELECT EXTRACT(EPOCH FROM (next_allowed_at - clock_timestamp())) FROM sync_rate_limits'))
        assert 0 < remaining <= 7


@pytest.mark.asyncio
async def test_missing_limiter_table_fails_closed(db):
    async with db() as s:
        await s.execute(text('DROP TABLE sync_rate_limits'))
        await s.commit()
    with pytest.raises(HTTPException) as error:
        await limiter.enforce_sync_rate_limit()
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_http_rate_limit_precedes_storage_and_compute_but_not_async(db, monkeypatch):
    app = FastAPI()
    app.include_router(router.public_router)
    async def session_dependency():
        async with db() as session:
            yield session
    app.dependency_overrides[get_db_session] = session_dependency
    storage = AsyncMock(return_value=('test', 'test.png'))
    pipeline = AsyncMock(return_value={'result_json': {'items': []}, 'ocr_result': 'text'})
    monkeypatch.setattr(router.image_storage, 'store_upload', storage)
    monkeypatch.setattr(router, 'run_extraction_pipeline', pipeline)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        invalid = await client.post('/extract/sync', files={'file': ('bad.exe', b'bad')})
        assert invalid.status_code == 400
        first = await client.post('/extract/sync', files={'file': ('test.png', b'fake image')})
        assert first.status_code == 200
        second = await client.post('/extract/sync', files={'file': ('test.png', b'fake image')})
        assert second.status_code == 429
        assert int(second.headers['Retry-After']) > 0
        assert storage.await_count == pipeline.await_count == 1
        queued = await client.post('/extract/async', files={'file': ('test.png', b'fake image')})
        assert queued.status_code == 202
        assert storage.await_count == 2 and pipeline.await_count == 1


@pytest.mark.asyncio
async def test_failed_processing_still_consumes_admission(db, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(router.image_storage, 'store_upload', AsyncMock(side_effect=OSError('storage failed')))
    async with db() as session:
        with pytest.raises(OSError):
            await router.extract_image_sync(file=SimpleNamespace(filename='test.png'),
                                            use_llm=False, session=session)
    with pytest.raises(HTTPException) as error:
        await limiter.enforce_sync_rate_limit()
    assert error.value.status_code == 429


@pytest.mark.asyncio
async def test_rate_limit_migration(db):
    from pathlib import Path
    async with db.kw['bind'].connect() as conn:
        raw = await conn.get_raw_connection()
        await raw.driver_connection.execute('DROP TABLE sync_rate_limits')
        sql = (Path(__file__).resolve().parents[1] / 'database/migrations/002_sync_rate_limit.sql').read_text()
        await raw.driver_connection.execute(sql)
    await limiter.enforce_sync_rate_limit()
    with pytest.raises(HTTPException) as error:
        await limiter.enforce_sync_rate_limit()
    assert error.value.status_code == 429
