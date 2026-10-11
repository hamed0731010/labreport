import asyncio
import logging

from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np

import si24.rulebased_labrows_parser
import si24.ocr_row_detector

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from src.core.config import LLM_BASE_URL, MAX_JOB_RETRIES
from src.database.connection import AsyncSessionLocal
from src.database.models import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractionSchema,
    ExtractionStatus,
    Prompt,
)
from src.service.lm_caller_client import LmCallerClient
from src.service.ocr_report_classifier import OCRReportClassifier
from src.service.minio_storage import image_storage


logger = logging.getLogger(__name__)

MAX_JOBS_PER_DRAIN = 20
RETRY_BASE_SECONDS = 10
RETRY_MAX_SECONDS = 300


def utc_now():
    return datetime.now(timezone.utc)


def retry_delay(retry_count: int) -> int:
    return min(
        RETRY_BASE_SECONDS * (2 ** max(0, retry_count - 1)),
        RETRY_MAX_SECONDS,
    )


def stringify_ocr_rows(rows):
    return "\n".join(
        str([item.get("text", "") for item in row])
        for row in rows
    )


# Loaded once per process.
paddle_extractor = (
    si24.ocr_row_detector.PaddleOcrExtractor(
        lang="fa",
        use_textline_orientation=True,
    )
)

table_extractor = (
    si24.ocr_row_detector.OCRTableExtractor4()
)

classifier = OCRReportClassifier()


def run_ocr_and_parse_sync(
    image_data: bytes,
    filename: str,
    use_lm: bool,
    llm_kwargs: dict | None = None,
    parse_rule_based: bool = True,
) -> dict:
    """Run blocking OCR and optionally parse the extracted rows."""

    encoded_image = np.frombuffer(
        image_data,
        dtype=np.uint8,
    )

    img = cv2.imdecode(
        encoded_image,
        cv2.IMREAD_COLOR,
    )

    if img is None:
        raise ValueError(
            f"Could not decode uploaded image: {filename}"
        )

    boxes = paddle_extractor.extract(img)

    logger.info(
        "OCR extraction completed: %s boxes",
        len(boxes),
    )

    rows = table_extractor.parse_text_table(boxes)

    logger.info(
        "Table extraction completed: %s rows",
        len(rows),
    )

    rows_text = stringify_ocr_rows(rows)

    all_texts = [
        item.get("text", "")
        for row in rows
        for item in row
    ]

    doc_type = classifier.classify(str(all_texts))

    parsed = {
        "items": None,
        "stats": None,
        "raw_llm_output": None,
        "ocr_result": rows_text,
    }

    if doc_type.get("name") == "lab":

        if use_lm:
            if not llm_kwargs:
                raise ValueError(
                    "llm_kwargs is required when use_lm=True"
                )

            client = LmCallerClient(
                base_url=LLM_BASE_URL
            )

            llm_response = client.extract_from_text(
                text=rows_text,
                prompt_text=llm_kwargs["prompt_text"],
                model_name=llm_kwargs["model_name"],
                custom_schema=llm_kwargs["custom_schema"],
                filename=llm_kwargs["filename"],
                debug=True,
            )

            llm_data = llm_response.get("data", {})

            if not isinstance(llm_data, dict):
                raise ValueError(
                    "LLM response data must be a JSON object"
                )

            parsed["items"] = llm_data.get(
                "items",
                llm_data,
            )
            parsed["stats"] = llm_data.get("stats", {})
            parsed["raw_llm_output"] = str(llm_data)

        elif parse_rule_based:

            parser = (
                si24.rulebased_labrows_parser
                .RulebasedLabRowsParser()
            )

            rule_result = parser.parse(rows)

            parsed["items"] = rule_result.get("items")
            parsed["stats"] = rule_result.get("stats")

    return {
        "items": parsed["items"],
        "stats": parsed["stats"],
        "ocr_result": rows_text,
        "rows": rows,
        "type": doc_type,
        "raw_llm_output": parsed["raw_llm_output"],
    }

async def _claim_job(
    stmt,
    processing_status: ExtractionStatus,
    stage: str,
):
    """Atomically claim one eligible database job."""

    try:
        async with AsyncSessionLocal() as session:

            async with session.begin():

                job = (
                    await session.execute(stmt)
                ).scalar_one_or_none()

                if job is None:
                    return None

                job.status = processing_status
                job.updated_at = utc_now()

                job_id = job.id

            return job_id

    except (SQLAlchemyError, OSError):
        logger.exception(
            "[%s] Failed to claim queue job",
            stage,
        )
        return "retry"
# ==========================================
# Task 1: Process one queued OCR job (Uploaded -> OCR Completed / Completed)
# ==========================================
async def task_process_ocr():
    """Process one eligible OCR job."""

    now = utc_now()

    stmt = (
        select(ExtractionRequest)
        .where(
            ExtractionRequest.status
            == ExtractionStatus.UPLOADED
        )
        .where(ExtractionRequest.updated_at <= now)
        .order_by(
            ExtractionRequest.created_at.asc(),
            ExtractionRequest.id.asc(),
        )
        .limit(1)
        .with_for_update(skip_locked=True)
    )

    job_id = await _claim_job(
        stmt,
        ExtractionStatus.OCR_PROCESSING,
        "OCR",
    )

    if job_id == "retry":
        return "retry"

    if job_id is None:
        return "empty"

    try:
        logger.info(
            "[OCR] Starting job %s",
            job_id,
        )

        # Read job data, then close the session.
        async with AsyncSessionLocal() as session:
            job = await session.get(
                ExtractionRequest,
                job_id,
            )

            if job is None:
                raise ValueError(
                    f"OCR job {job_id} not found"
                )

            minio_path = job.minio_path
            use_lm = job.use_lm

        # No database session is held here.
        image_data = await asyncio.to_thread(
            image_storage.download,
            minio_path,
        )

        result = await asyncio.to_thread(
            run_ocr_and_parse_sync,
            image_data=image_data,
            filename=Path(minio_path).name,
            use_lm=False,
            llm_kwargs=None,
            parse_rule_based=not use_lm,
        )

        # Save results in a fresh transaction.
        async with AsyncSessionLocal() as session:

            async with session.begin():

                job = (
                    await session.execute(
                        select(ExtractionRequest)
                        .where(
                            ExtractionRequest.id == job_id
                        )
                        .with_for_update()
                    )
                ).scalar_one()

                if job.status != ExtractionStatus.OCR_PROCESSING:
                    logger.warning(
                        "[OCR] Job %s status changed; "
                        "skipping result",
                        job_id,
                    )
                    return "empty"

                job.ocr_raw_text = result["ocr_result"]

                if use_lm:
                    job.status = ExtractionStatus.OCR_COMPLETED

                    # Give LLM its own retry budget.
                    job.retry_count = 0

                    logger.info(
                        "[OCR] Job %s queued for LLM",
                        job_id,
                    )

                else:
                    result_json = {
                        "items": result["items"],
                        "stats": result["stats"],
                        "type": result["type"],
                        "ocr_result": result["ocr_result"],
                    }

                    response = (
                        await session.execute(
                            select(ExtractionResponse)
                            .where(
                                ExtractionResponse.request_id
                                == job_id
                            )
                        )
                    ).scalar_one_or_none()

                    if response is None:
                        response = ExtractionResponse(
                            request_id=job_id
                        )
                        session.add(response)

                    response.result_json = result_json
                    response.raw_llm_output = (
                        result["raw_llm_output"]
                    )

                    job.status = ExtractionStatus.COMPLETED
                    job.retry_count = 0

                    logger.info(
                        "[OCR] Job %s completed",
                        job_id,
                    )

                job.updated_at = utc_now()
                job.last_error_message = None

        return "done"

    except asyncio.CancelledError:
        logger.warning(
            "[OCR] Job %s cancelled",
            job_id,
        )
        raise

    except Exception as error:
        logger.exception(
            "[OCR] Job %s failed",
            job_id,
        )

        await mark_job_failed(
            job_id,
            str(error),
            ExtractionStatus.UPLOADED,
        )

        return "retry"
# ==========================================
# Task 2: Process one queued LLM job (OCR Completed -> Completed)
# ==========================================
async def task_process_llm(eligible_before=None):
    """Process one eligible LLM job."""

    cutoff = eligible_before or utc_now()

    stmt = (
        select(ExtractionRequest)
        .where(
            ExtractionRequest.status
            == ExtractionStatus.OCR_COMPLETED
        )
        .where(ExtractionRequest.use_lm.is_(True))
        .where(ExtractionRequest.updated_at <= cutoff)
        .order_by(
            ExtractionRequest.updated_at.asc(),
            ExtractionRequest.id.asc(),
        )
        .limit(1)
        .with_for_update(skip_locked=True)
    )

    job_id = await _claim_job(
        stmt,
        ExtractionStatus.LM_PROCESSING,
        "LLM",
    )

    if job_id == "retry":
        return "retry"

    if job_id is None:
        return "empty"

    try:
        logger.info("[LLM] Starting job %s", job_id)

        # Load all required data.
        async with AsyncSessionLocal() as session:

            job = await session.get(
                ExtractionRequest,
                job_id,
            )

            if job is None:
                raise ValueError(
                    f"LLM job {job_id} not found"
                )

            ocr_text = job.ocr_raw_text or ""
            model_name = job.lm_model
            minio_path = job.minio_path

            prompt_text = "Extract the following fields."
            custom_schema = {"title": "LabReport"}

            if job.prompt_id:
                prompt = await session.get(
                    Prompt,
                    job.prompt_id,
                )

                if prompt:
                    prompt_text = prompt.template_text

            if job.schema_id:
                schema = await session.get(
                    ExtractionSchema,
                    job.schema_id,
                )

                if schema:
                    custom_schema = dict(
                        schema.schema_definition
                    )
                    custom_schema.setdefault(
                        "title",
                        schema.name,
                    )

        # Session is closed before LLM invocation.
        if not ocr_text.strip():
            raise ValueError(
                f"Job {job_id} has no OCR text"
            )

        client = LmCallerClient(
            base_url=LLM_BASE_URL
        )

        started_at = asyncio.get_running_loop().time()

        llm_response = await asyncio.to_thread(
            client.extract_from_text,
            text=ocr_text,
            prompt_text=prompt_text,
            model_name=model_name,
            custom_schema=custom_schema,
            filename=(
                f"request-{job_id}-"
                f"{Path(minio_path).name}"
            ),
            debug=True,
        )

        elapsed = (
            asyncio.get_running_loop().time()
            - started_at
        )

        logger.info(
            "[LLM] Job %s completed in %.1fs",
            job_id,
            elapsed,
        )

        llm_data = llm_response.get("data", {})

        if not isinstance(llm_data, dict):
            raise ValueError(
                "LLM response data must be a JSON object"
            )

        result_json = {
            "items": llm_data.get("items", llm_data),
            "stats": llm_data.get("stats", {}),
            "ocr_result": ocr_text,
        }

        # Save results using a new transaction.
        async with AsyncSessionLocal() as session:

            async with session.begin():

                job = (
                    await session.execute(
                        select(ExtractionRequest)
                        .where(
                            ExtractionRequest.id == job_id
                        )
                        .with_for_update()
                    )
                ).scalar_one()

                if job.status != ExtractionStatus.LM_PROCESSING:
                    logger.warning(
                        "[LLM] Job %s status changed; "
                        "skipping result",
                        job_id,
                    )
                    return "empty"

                response = (
                    await session.execute(
                        select(ExtractionResponse)
                        .where(
                            ExtractionResponse.request_id
                            == job_id
                        )
                    )
                ).scalar_one_or_none()

                if response is None:
                    response = ExtractionResponse(
                        request_id=job_id
                    )
                    session.add(response)

                response.result_json = result_json
                response.raw_llm_output = str(llm_data)

                job.status = ExtractionStatus.COMPLETED
                job.updated_at = utc_now()
                job.retry_count = 0
                job.last_error_message = None

                if job.ocr_metadata:
                    metadata = dict(job.ocr_metadata)

                    metadata.pop("llm_job_id", None)
                    metadata.pop("llm_runner_status", None)

                    job.ocr_metadata = metadata or None

        return "done"

    except asyncio.CancelledError:
        logger.warning(
            "[LLM] Job %s cancelled",
            job_id,
        )
        raise

    except Exception as error:
        logger.exception(
            "[LLM] Job %s failed",
            job_id,
        )

        await mark_job_failed(
            job_id,
            str(error),
            ExtractionStatus.OCR_COMPLETED,
        )

        return "retry"

    
async def mark_job_failed(
    job_id: int,
    error_msg: str,
    fallback_status: ExtractionStatus,
):
    """Record a failure and schedule a delayed retry."""

    try:
        async with AsyncSessionLocal() as session:

            async with session.begin():

                job = (
                    await session.execute(
                        select(ExtractionRequest)
                        .where(ExtractionRequest.id == job_id)
                        .with_for_update()
                    )
                ).scalar_one_or_none()

                if job is None:
                    return

                if job.status not in (
                    ExtractionStatus.OCR_PROCESSING,
                    ExtractionStatus.LM_PROCESSING,
                ):
                    return

                # Only update the expected processing stage.
                expected_status = (
                    ExtractionStatus.OCR_PROCESSING
                    if fallback_status == ExtractionStatus.UPLOADED
                    else ExtractionStatus.LM_PROCESSING
                )

                if job.status != expected_status:
                    return

                job.retry_count = min(
                    job.retry_count + 1,
                    MAX_JOB_RETRIES,
                )

                job.last_error_message = error_msg

                if job.retry_count >= MAX_JOB_RETRIES:

                    job.status = ExtractionStatus.FAILED
                    job.updated_at = utc_now()

                    logger.error(
                        "[Queue] Job %s permanently failed",
                        job_id,
                    )

                else:
                    delay = retry_delay(job.retry_count)

                    job.status = fallback_status

                    # Temporarily reuse updated_at
                    # as the next eligible processing time.
                    job.updated_at = (
                        utc_now()
                        + timedelta(seconds=delay)
                    )

                    logger.warning(
                        "[Queue] Job %s will retry in %ss",
                        job_id,
                        delay,
                    )

    except (SQLAlchemyError, OSError):
        logger.exception(
            "[Queue] Failed to record error for job %s",
            job_id,
        )