"""Real PostgreSQL transactions with only external compute/storage stubbed."""
import os
import sys
import types
from pathlib import Path
from uuid import uuid4
from unittest.mock import Mock

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

# The deployed application imports this repository as src.
src = types.ModuleType('src')
src.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules['src'] = src
os.environ['APP_ENV'] = 'test'
os.environ['LOG_FILE'] = os.devnull

# No model loading, storage initialization or HTTP calls during collection.
for name in ('cv2', 'numpy', 'si24', 'si24.rulebased_labrows_parser',
             'si24.ocr_row_detector', 'src.service.minio_storage',
             'src.service.ocr_report_classifier', 'src.service.lm_caller_client'):
    module = types.ModuleType(name)
    sys.modules[name] = module
sys.modules['si24'].__path__ = []
for name in ('rulebased_labrows_parser', 'ocr_row_detector'):
    setattr(sys.modules['si24'], name, sys.modules['si24.' + name])
sys.modules['si24.ocr_row_detector'].PaddleOcrExtractor = Mock()
sys.modules['si24.ocr_row_detector'].OCRTableExtractor4 = Mock()
sys.modules['src.service.ocr_report_classifier'].OCRReportClassifier = Mock()
sys.modules['src.service.minio_storage'].image_storage = Mock()
sys.modules['src.service.lm_caller_client'].LmCallerClient = Mock()

from src.database.models import Base
from src.service import job_leases
from src.service import sync_rate_limit
from src.workers import tasks


@pytest_asyncio.fixture
async def db(monkeypatch):
    url = os.getenv('TEST_DATABASE_URL')
    if not url:
        pytest.skip('Set TEST_DATABASE_URL to a disposable PostgreSQL database')
    schema = 'queue_test_' + uuid4().hex
    admin = create_async_engine(url)
    async with admin.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA {schema}'))
    engine = create_async_engine(url, connect_args={'server_settings': {'search_path': schema}})
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        monkeypatch.setattr(job_leases, 'AsyncSessionLocal', sessions)
        monkeypatch.setattr(tasks, 'AsyncSessionLocal', sessions)
        monkeypatch.setattr(sync_rate_limit, 'AsyncSessionLocal', sessions)
        yield sessions
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA {schema} CASCADE'))
        await admin.dispose()
