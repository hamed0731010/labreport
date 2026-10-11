"""Pydantic response schemas for the HTTP API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class HealthLiveResponse(BaseModel):
    status: str


class HealthReadyResponse(BaseModel):
    status: str
    database: str
    storage: str


class AsyncAcceptedResponse(BaseModel):
    request_id: int
    status: str
    message: str


class SyncExtractionResponse(BaseModel):
    request_id: int
    status: str
    result: dict[str, Any]
    ocr_result: str | None = None
    raw_llm_output: str | None = None


class ExtractionStatusResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: int
    status: str
    retry_count: int
    created_at: datetime
    updated_at: datetime
    llm_queue_position: int | None = None
    message: str | None = None
