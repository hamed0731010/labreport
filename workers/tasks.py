import asyncio
import logging

from pathlib import Path

import cv2
import numpy as np

import si24.rulebased_labrows_parser
import si24.ocr_row_detector

from sqlalchemy import select, or_
from sqlalchemy.exc import SQLAlchemyError

from src.core.config import LLM_BASE_URL
from src.service.job_leases import (
    assign_lease, clear_lease, heartbeat, mark_job_failed,
    owns_lease, stop_heartbeat, utc_now,
)
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
                token = assign_lease(job)

                job_id = job.id

            return job_id, token

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
        .where(ExtractionRequest.queue_managed.is_(True))
        .where(or_(ExtractionRequest.next_retry_at.is_(None),
                   ExtractionRequest.next_retry_at <= now))
        .order_by(
            ExtractionRequest.created_at.asc(),
            ExtractionRequest.id.asc(),
        )
        .limit(1)
        .with_for_update(skip_locked=True)
    )

    claim = await _claim_job(
        stmt,
        ExtractionStatus.OCR_PROCESSING,
        "OCR",
    )

    if claim == "retry":
        return "retry"

    if claim is None:
        return "empty"

    job_id, token = claim
    lease_task = asyncio.create_task(heartbeat(job_id, token))
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

                if not owns_lease(job, token, ExtractionStatus.OCR_PROCESSING):
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

                clear_lease(job)
                job.next_retry_at = None
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
            token,
        )

        return "retry"
    finally:
        await stop_heartbeat(lease_task)

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
        .where(ExtractionRequest.queue_managed.is_(True))
        .where(ExtractionRequest.updated_at <= cutoff)
        .where(or_(ExtractionRequest.next_retry_at.is_(None),
                   ExtractionRequest.next_retry_at <= cutoff))
        .order_by(
            ExtractionRequest.updated_at.asc(),
            ExtractionRequest.id.asc(),
        )
        .limit(1)
        .with_for_update(skip_locked=True)
    )

    claim = await _claim_job(
        stmt,
        ExtractionStatus.LM_PROCESSING,
        "LLM",
    )

    if claim == "retry":
        return "retry"

    if claim is None:
        return "empty"

    job_id, token = claim
    lease_task = asyncio.create_task(heartbeat(job_id, token))
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

                if not owns_lease(job, token, ExtractionStatus.LM_PROCESSING):
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

                clear_lease(job)
                job.next_retry_at = None
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
            token,
        )

        return "retry"
    finally:
        await stop_heartbeat(lease_task)
