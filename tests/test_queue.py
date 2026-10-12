import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from src.database.models import ExtractionRequest as Request, ExtractionResponse as Response, ExtractionStatus as S
from src.service import job_leases as leases
from src.workers import tasks
from src.api import router


async def add_job(db, **values):
    async with db() as session:
        job = Request(image_id='test', minio_path='test.png', use_lm=False,
                      status=S.UPLOADED, **values)
        session.add(job)
        await session.commit()
        return job.id


async def get_job(db, job_id):
    async with db() as session:
        return await session.get(Request, job_id)


def claim_stmt():
    return (select(Request).where(Request.status == S.UPLOADED)
            .order_by(Request.id).limit(1).with_for_update(skip_locked=True))


def ocr_result():
    return dict(items=[], stats={}, type={'name': 'lab'}, ocr_result='text', raw_llm_output=None)


@pytest.mark.asyncio
async def test_concurrent_claims_have_distinct_owners(db):
    ids = {await add_job(db), await add_job(db)}
    claims = await asyncio.gather(*[
        tasks._claim_job(claim_stmt(), S.OCR_PROCESSING, 'test') for _ in range(4)
    ])
    claimed = [c for c in claims if c is not None]
    assert {c[0] for c in claimed} == ids
    assert len(claimed) == 2
    assert len({c[1] for c in claimed}) == 2


@pytest.mark.asyncio
async def test_locked_first_row_is_skipped(db):
    first, second = await add_job(db), await add_job(db)
    async with db() as locker:
        async with locker.begin():
            await locker.execute(select(Request).where(Request.id == first).with_for_update())
            claim = await asyncio.wait_for(tasks._claim_job(claim_stmt(), S.OCR_PROCESSING, 'test'), 2)
            assert claim[0] == second


@pytest.mark.asyncio
async def test_recovery_once_backoff_and_stale_failure(db):
    job_id = await add_job(db)
    _, old_token = await tasks._claim_job(claim_stmt(), S.OCR_PROCESSING, 'test')
    async with db() as s:
        job = await s.get(Request, job_id)
        job.lease_expires_at = leases.utc_now() - timedelta(seconds=1)
        await s.commit()
    assert sum(await asyncio.gather(leases.recover_expired_jobs(), leases.recover_expired_jobs())) == 1
    job = await get_job(db, job_id)
    assert job.status == S.UPLOADED and job.retry_count == 1
    assert job.next_retry_at > job.updated_at
    assert job.lease_token is None
    assert await tasks.task_process_ocr() == 'empty'
    _, new_token = await tasks._claim_job(claim_stmt(), S.OCR_PROCESSING, 'test')
    await leases.mark_job_failed(job_id, 'stale', S.UPLOADED, old_token)
    job = await get_job(db, job_id)
    assert job.lease_token == new_token and job.retry_count == 1
    assert not await leases.renew_lease(job_id, old_token)


@pytest.mark.asyncio
async def test_heartbeat_preserves_updated_at_and_cannot_revive_expired(db):
    job_id = await add_job(db)
    _, token = await tasks._claim_job(claim_stmt(), S.OCR_PROCESSING, 'test')
    before = await get_job(db, job_id)
    assert await leases.renew_lease(job_id, token)
    after = await get_job(db, job_id)
    assert after.updated_at == before.updated_at
    assert after.lease_expires_at >= before.lease_expires_at
    assert await leases.recover_expired_jobs() == 0
    async with db() as s:
        job = await s.get(Request, job_id)
        job.lease_expires_at = leases.utc_now() - timedelta(seconds=1)
        await s.commit()
    assert not await leases.renew_lease(job_id, token)


@pytest.mark.asyncio
async def test_retry_budget_above_three_and_terminal_failure(db, monkeypatch):
    monkeypatch.setattr(leases, 'MAX_JOB_RETRIES', 5)
    job_id = await add_job(db, retry_count=3)
    _, token = await tasks._claim_job(claim_stmt(), S.OCR_PROCESSING, 'test')
    await leases.mark_job_failed(job_id, 'failed', S.UPLOADED, token)
    assert (await get_job(db, job_id)).retry_count == 4
    _, token = await tasks._claim_job(claim_stmt(), S.OCR_PROCESSING, 'test')
    await leases.mark_job_failed(job_id, 'failed again', S.UPLOADED, token)
    job = await get_job(db, job_id)
    assert job.status == S.FAILED and job.next_retry_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['ocr', 'llm'])
async def test_stale_result_cannot_overwrite_reclaimed_job(db, monkeypatch, stage):
    job_id = await add_job(db)
    if stage == 'llm':
        async with db() as s:
            job = await s.get(Request, job_id)
            job.status, job.use_lm, job.ocr_raw_text = S.OCR_COMPLETED, True, 'text'
            await s.commit()
    expected = S.OCR_PROCESSING if stage == 'ocr' else S.LM_PROCESSING
    async def replace_owner(*args, **kwargs):
        async with db() as s:
            job = await s.get(Request, job_id)
            job.lease_token = 'new-owner'
            await s.commit()
        return ocr_result() if stage == 'ocr' else {'data': {'items': []}}
    async def compute(fn, *args, **kwargs):
        if fn == tasks.image_storage.download:
            return b'image'
        return await replace_owner()
    monkeypatch.setattr(asyncio, 'to_thread', compute)
    run = tasks.task_process_ocr if stage == 'ocr' else tasks.task_process_llm
    assert await run() == 'empty'
    job = await get_job(db, job_id)
    assert job.status == expected and job.lease_token == 'new-owner'
    async with db() as s:
        assert (await s.execute(select(Response))).scalars().all() == []


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['ocr', 'llm'])
async def test_cancellation_leaves_lease_for_recovery(db, monkeypatch, stage):
    job_id = await add_job(db)
    if stage == 'llm':
        async with db() as s:
            job = await s.get(Request, job_id)
            job.status, job.use_lm, job.ocr_raw_text = S.OCR_COMPLETED, True, 'text'
            await s.commit()
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError
    monkeypatch.setattr(asyncio, 'to_thread', cancelled)
    run = tasks.task_process_ocr if stage == 'ocr' else tasks.task_process_llm
    with pytest.raises(asyncio.CancelledError):
        await run()
    job = await get_job(db, job_id)
    assert job.lease_token is not None and job.retry_count == 0
    assert not any(t.get_coro().__name__ == 'heartbeat' for t in asyncio.all_tasks())
    async with db() as s:
        row = await s.get(Request, job_id)
        row.lease_expires_at = leases.utc_now() - timedelta(seconds=1)
        await s.commit()
    await leases.recover_expired_jobs()
    assert (await get_job(db, job_id)).status == (S.UPLOADED if stage == 'ocr' else S.OCR_COMPLETED)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [False, True])
async def test_sync_never_queued_and_error_keeps_request_id(db, monkeypatch, failure):
    monkeypatch.setattr(router.image_storage, 'store_upload', AsyncMock(return_value=('image', 'test.png')))
    async def pipeline(**kwargs):
        async with db() as s:
            job = (await s.execute(select(Request))).scalar_one()
            assert job.status == S.OCR_PROCESSING and not job.queue_managed
        assert await tasks.task_process_ocr() == 'empty'
        if failure:
            raise ValueError('original error')
        return {'result_json': {'items': []}, 'ocr_result': 'raw text'}
    monkeypatch.setattr(router, 'run_extraction_pipeline', pipeline)
    async with db() as s:
        call = router.extract_image_sync(file=SimpleNamespace(filename='test.png'), use_llm=False,
                                        llm_model='test', llm_prompt='test', schema_name='test', session=s)
        if failure:
            with pytest.raises(HTTPException) as error:
                await call
            assert error.value.status_code == 500
        else:
            assert (await call)['status'] == 'completed'
    async with db() as s:
        job = (await s.execute(select(Request))).scalar_one()
        assert job.status == (S.FAILED if failure else S.COMPLETED)
        assert job.lease_token is None
        if failure:
            assert job.last_error_message == 'original error'
        else:
            assert job.ocr_raw_text == 'raw text'


@pytest.mark.asyncio
async def test_expired_sync_is_failed_not_requeued(db):
    job_id = await add_job(db, queue_managed=False)
    async with db() as s:
        job = await s.get(Request, job_id)
        job.status = S.OCR_PROCESSING
        leases.assign_lease(job)
        job.lease_expires_at = leases.utc_now() - timedelta(seconds=1)
        await s.commit()
    await leases.recover_expired_jobs()
    job = await get_job(db, job_id)
    assert job.status == S.FAILED and job.next_retry_at is None


@pytest.mark.asyncio
async def test_both_drains_are_bounded(monkeypatch):
    from src import main
    monkeypatch.setattr(main, 'MAX_JOBS_PER_DRAIN', 3)
    monkeypatch.setattr(main, 'recover_expired_jobs', AsyncMock())
    monkeypatch.setattr(main, 'wake_queue', lambda stage: None)
    for name, drain in [('task_process_ocr', main._drain_ocr_queue), ('task_process_llm', main._drain_llm_queue)]:
        worker = AsyncMock(return_value='done')
        monkeypatch.setattr(main, name, worker)
        await drain()
        assert worker.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['ocr', 'llm'])
async def test_success_clears_ownership_and_saves_response(db, monkeypatch, stage):
    job_id = await add_job(db)
    if stage == 'llm':
        async with db() as s:
            job = await s.get(Request, job_id)
            job.status, job.use_lm, job.ocr_raw_text = S.OCR_COMPLETED, True, 'text'
            await s.commit()
    async def compute(fn, *args, **kwargs):
        if fn == tasks.image_storage.download:
            return b'image'
        return ocr_result() if stage == 'ocr' else {'data': {'items': []}}
    monkeypatch.setattr(asyncio, 'to_thread', compute)
    run = tasks.task_process_ocr if stage == 'ocr' else tasks.task_process_llm
    assert await run() == 'done'
    job = await get_job(db, job_id)
    assert job.status == S.COMPLETED and job.lease_token is None
    assert job.lease_expires_at is None and job.next_retry_at is None
    async with db() as s:
        assert (await s.execute(select(Response))).scalar_one().result_json['items'] == []


@pytest.mark.asyncio
async def test_sync_stale_completion_is_rejected(db, monkeypatch):
    monkeypatch.setattr(router.image_storage, 'store_upload', AsyncMock(return_value=('image', 'test.png')))
    async def pipeline(**kwargs):
        async with db() as s:
            job = (await s.execute(select(Request))).scalar_one()
            job.lease_expires_at = leases.utc_now() - timedelta(seconds=1)
            await s.commit()
        await leases.recover_expired_jobs()
        return {'result_json': {'items': []}}
    monkeypatch.setattr(router, 'run_extraction_pipeline', pipeline)
    async with db() as s:
        with pytest.raises(HTTPException) as error:
            await router.extract_image_sync(file=SimpleNamespace(filename='test.png'), use_llm=False,
                                            llm_model='test', llm_prompt='test', schema_name='test', session=s)
        assert error.value.status_code == 409
    async with db() as s:
        assert (await s.execute(select(Request))).scalar_one().status == S.FAILED
        assert (await s.execute(select(Response))).scalars().all() == []


def test_backoff_is_bounded_for_large_retry_budget():
    assert [leases.retry_delay(i) for i in (1, 2, 3, 6, 10000)] == [10, 20, 40, 300, 300]


@pytest.mark.asyncio
async def test_migration_preserves_delays_and_recovers_old_processing(db):
    from pathlib import Path
    engine = db.kw['bind']
    async with engine.connect() as conn:
        raw = await conn.get_raw_connection()
        # Reconstruct the old columns/constraint in this isolated test schema.
        await raw.driver_connection.execute('''
            ALTER TABLE lab_report_extraction_requests
                DROP COLUMN next_retry_at, DROP COLUMN lease_token,
                DROP COLUMN lease_expires_at, DROP COLUMN queue_managed,
                DROP CONSTRAINT chk_retry_count;
            ALTER TABLE lab_report_extraction_requests ADD CONSTRAINT chk_retry_count
                CHECK (retry_count >= 0 AND retry_count <= 3);
            INSERT INTO lab_report_extraction_requests
                (image_id, minio_path, use_lm, status, retry_count, updated_at)
            VALUES ('retry', 'a.png', false, 'uploaded', 1, now() + interval '1 minute'),
                   ('crashed', 'b.png', false, 'ocr_processing', 0, now());
        ''')
        sql = (Path(__file__).resolve().parents[1] / 'database/migrations/001_job_leases.sql').read_text()
        await raw.driver_connection.execute(sql)
    async with db() as s:
        rows = (await s.execute(select(Request).order_by(Request.id))).scalars().all()
        assert rows[0].next_retry_at > rows[0].updated_at
        assert rows[1].lease_expires_at <= leases.utc_now()
        rows[0].retry_count = 7
        await s.commit()
    assert await leases.recover_expired_jobs() == 1
