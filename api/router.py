import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, tuple_
from src.core.config import (
    ALLOWED_EXTENSIONS,
    ASYNC_LLM_PROMPT,
    ASYNC_SCHEMA_NAME,
    LLM_MODEL,
    SYNC_LLM_PROMPT,
    SYNC_SCHEMA_NAME,
)
from src.database.connection import get_db_session
from src.database.models import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractionSchema,
    ExtractionStatus,
    Prompt,
)

from src.service.extraction_pipeline import run_extraction_pipeline
from src.service.job_leases import (
    assign_lease, clear_lease, heartbeat, mark_job_failed, owns_lease, stop_heartbeat,
)
from src.service.minio_storage import image_storage
from src.api.schema import (
    AsyncAcceptedResponse,
    ExtractionStatusResponse,
    HealthLiveResponse,
    HealthReadyResponse,
    SyncExtractionResponse,
)

logger = logging.getLogger(__name__)

public_router = APIRouter()


def validate_file_extension(filename: str | None) -> bool:
    """Return whether the uploaded filename uses an allowed extension."""
    return bool(filename) and Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


async def resolve_prompt_and_schema(
    session: AsyncSession, llm_prompt: str, schema_name: str
) -> tuple[Prompt, ExtractionSchema]:
    """Load the active LLM prompt and schema selected by the request."""
    prompt_result = await session.execute(
        select(Prompt).where(Prompt.name == llm_prompt, Prompt.is_active.is_(True))
    )
    prompt = prompt_result.scalar_one_or_none()
    if prompt is None:
        raise HTTPException(
            status_code=400,
            detail=f"Active prompt '{llm_prompt}' not found in database.",
        )

    schema_result = await session.execute(
        select(ExtractionSchema).where(
            ExtractionSchema.name == schema_name,
            ExtractionSchema.is_active.is_(True),
        )
    )
    schema = schema_result.scalar_one_or_none()
    if schema is None:
        raise HTTPException(
            status_code=400,
            detail=f"Active schema '{schema_name}' not found in database.",
        )

    return prompt, schema


# ==========================================
# Health Checks (Per ADR)
# ==========================================
@public_router.get("/health/live", summary="Liveness Probe", response_model=HealthLiveResponse)
def health_live():
    return {"status": "alive"}


@public_router.get("/health/ready", summary="Readiness Probe", response_model=HealthReadyResponse)
async def health_ready(session: AsyncSession = Depends(get_db_session)):
    try:
        await session.execute(select(1))
        await asyncio.to_thread(image_storage.ensure_bucket)
        return {"status": "ready", "database": "connected", "storage": "connected"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Database or image storage unavailable")


# ==========================================
# Ingestion Endpoint (Async, queued)
# ==========================================
@public_router.post("/extract/async", summary="Upload a single image for async processing", status_code=status.HTTP_202_ACCEPTED, response_model=AsyncAcceptedResponse)
async def upload_image_async(
    file: UploadFile = File(...),
    use_llm: bool = Form(False, description="Set to True to parse rows using LLM"),
    llm_model: str = Form(LLM_MODEL, description="Model name if use_llm is True"),
    llm_prompt: str = Form(ASYNC_LLM_PROMPT, description="Prompt name in DB"),
    schema_name: str = Form(ASYNC_SCHEMA_NAME, description="Schema name in DB"),
    session: AsyncSession = Depends(get_db_session)
):
    logger.info(f"Upload request received: {file.filename}, use_llm={use_llm}")

    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    prompt_id = None
    schema_id = None

    if use_llm:
        prompt_obj, schema_obj = await resolve_prompt_and_schema(session, llm_prompt, schema_name)
        prompt_id = prompt_obj.id
        schema_id = schema_obj.id

    image_id, object_name = await image_storage.store_upload(file)

    # 2. Register Job in PostgreSQL Queue
    new_request = ExtractionRequest(
        image_id=image_id,
        minio_path=object_name,
        use_lm=use_llm,
        prompt_id=prompt_id,
        schema_id=schema_id,
        lm_model=llm_model if use_llm else None,
        status=ExtractionStatus.UPLOADED,
    )

    session.add(new_request)
    await session.commit()
    await session.refresh(new_request)
    request_id = new_request.id

    from src.service.queue_scheduler import wake_queue

    wake_queue("ocr")

    logger.info(f"Job queued successfully. Request ID: {new_request.id}")

    # 3. Return 202 Accepted immediately
    return {
        "request_id": new_request.id,
        "status": "uploaded",
        "message": "Image queued for background processing. Poll /status/{request_id} for results."
    }


# ==========================================
# Ingestion Endpoint (Sync, immediate answer)
# ==========================================
@public_router.post("/extract/sync", summary="Upload a single image and get results immediately (no polling)", response_model=SyncExtractionResponse)
async def extract_image_sync(
    file: UploadFile = File(...),
    use_llm: bool = Form(False, description="Set to True to parse rows using LLM"),
    llm_model: str = Form(LLM_MODEL, description="Model name if use_llm is True"),
    llm_prompt: str = Form(SYNC_LLM_PROMPT, description="Prompt name in DB"),
    schema_name: str = Form(SYNC_SCHEMA_NAME, description="Schema name in DB"),
    session: AsyncSession = Depends(get_db_session),
):
    logger.info(f"Sync extraction request received: {file.filename}, use_llm={use_llm}")

    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    prompt_obj = None
    schema_obj = None

    if use_llm:
        prompt_obj, schema_obj = await resolve_prompt_and_schema(session, llm_prompt, schema_name)

    image_id, object_name = await image_storage.store_upload(file)

    # A sync request must never appear in the background queue.
    new_request = ExtractionRequest(
        image_id=image_id,
        minio_path=object_name,
        use_lm=use_llm,
        prompt_id=prompt_obj.id if prompt_obj else None,
        schema_id=schema_obj.id if schema_obj else None,
        lm_model=llm_model if use_llm else None,
        status=ExtractionStatus.OCR_PROCESSING,
        queue_managed=False,
    )
    token = assign_lease(new_request)
    session.add(new_request)
    await session.flush()
    request_id = new_request.id
    await session.commit()

    lease_task = asyncio.create_task(heartbeat(request_id, token))
    try:
        pipeline_output = await run_extraction_pipeline(
            object_name=object_name,
            use_llm=use_llm,
            llm_model=llm_model,
            prompt_obj=prompt_obj,
            schema_obj=schema_obj,
        )

        # Refresh under a lock: recovery may have invalidated our ownership.
        new_request = (await session.execute(
            select(ExtractionRequest)
            .where(ExtractionRequest.id == request_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )).scalar_one()
        if not owns_lease(new_request, token, ExtractionStatus.OCR_PROCESSING):
            raise HTTPException(status_code=409, detail="Processing lease expired; check request status")

        new_response = ExtractionResponse(
            request_id=request_id,
            result_json=pipeline_output["result_json"],
            raw_llm_output=pipeline_output.get("raw_llm_output"),
        )
        session.add(new_response)
        new_request.ocr_raw_text = pipeline_output.get("ocr_result")
        new_request.status = ExtractionStatus.COMPLETED
        new_request.updated_at = datetime.now(timezone.utc)
        new_request.last_error_message = None
        clear_lease(new_request)
        await session.commit()

        logger.info("Sync extraction completed. Request ID: %s", request_id)
        return {
            "request_id": request_id,
            "status": "completed",
            "result": pipeline_output["result_json"],
            "ocr_result": pipeline_output.get("ocr_result"),
            "raw_llm_output": pipeline_output.get("raw_llm_output"),
        }
    except HTTPException:
        await session.rollback()
        raise
    except asyncio.CancelledError:
        # to_thread cannot stop the underlying call. Leave ownership to expire;
        # recovery marks sync requests failed instead of enqueueing them.
        await session.rollback()
        raise
    except Exception as error:
        logger.exception("Sync extraction failed for request %s", request_id)
        await session.rollback()
        await mark_job_failed(request_id, str(error), ExtractionStatus.UPLOADED, token)
        raise HTTPException(status_code=500, detail="Extraction failed") from error
    finally:
        await stop_heartbeat(lease_task)


# ==========================================
# Polling Endpoint (for the async path)
# ==========================================
@public_router.get("/status/{request_id}", summary="Check status and get results", response_model=ExtractionStatusResponse)
async def get_extraction_status(request_id: int, session: AsyncSession = Depends(get_db_session)):
    req_res = await session.execute(select(ExtractionRequest).where(ExtractionRequest.id == request_id))
    req = req_res.scalar_one_or_none()

    if not req:
        raise HTTPException(status_code=404, detail="Request ID not found")

    response_data = {
        "request_id": req.id,
        "status": req.status.value,
        "retry_count": req.retry_count,
        "created_at": req.created_at,
        "updated_at": req.updated_at,
        "next_retry_at": req.next_retry_at,
    }

    if req.status == ExtractionStatus.OCR_COMPLETED and req.use_lm:
        ahead = await session.scalar(
            select(func.count()).select_from(ExtractionRequest).where(
                ExtractionRequest.status == ExtractionStatus.OCR_COMPLETED,
                ExtractionRequest.use_lm.is_(True),
                tuple_(ExtractionRequest.updated_at, ExtractionRequest.id)
                < tuple_(req.updated_at, req.id),
            )
        )
        response_data["llm_queue_position"] = ahead + 1
        response_data["message"] = "OCR finished; waiting for the LLM worker."
    elif req.status == ExtractionStatus.LM_PROCESSING:
        response_data["message"] = "The LLM worker has claimed this request; the result is not saved yet."

    if req.status == ExtractionStatus.FAILED:
        response_data["error"] = req.last_error_message

    # OCR text is persisted as soon as the OCR stage finishes, so expose it
    # while an LLM job is still queued or processing as well as after completion.
    if req.ocr_raw_text is not None:
        response_data["ocr_result"] = req.ocr_raw_text

    if req.status == ExtractionStatus.COMPLETED:
        resp_res = await session.execute(select(ExtractionResponse).where(ExtractionResponse.request_id == request_id))
        result = resp_res.scalar_one_or_none()

        if result:
            response_data["result"] = result.result_json
            response_data["raw_llm_output"] = result.raw_llm_output

    return response_data
