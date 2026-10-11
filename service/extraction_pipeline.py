"""Coordinates image retrieval and OCR/optional LLM parsing for sync requests."""

import asyncio
from pathlib import Path
from typing import Any

from src.service.minio_storage import image_storage
from src.workers.tasks import run_ocr_and_parse_sync


async def run_extraction_pipeline(
    object_name: str,
    use_llm: bool,
    llm_model: str,
    prompt_obj: Any,
    schema_obj: Any,
) -> dict[str, Any]:
    """Run the shared OCR service and optionally pass extracted rows to the LLM."""
    llm_kwargs = None
    if use_llm:
        custom_schema = schema_obj.schema_definition if schema_obj else {}
        if isinstance(custom_schema, dict) and "title" not in custom_schema:
            custom_schema = {**custom_schema, "title": schema_obj.name if schema_obj else "ExtractionSchema"}

        llm_kwargs = {
            "model_name": llm_model,
            "prompt_text": prompt_obj.template_text if prompt_obj else "Extract the following fields.",
            "custom_schema": custom_schema,
            "filename": Path(object_name).name,
        }

    image_data = await asyncio.to_thread(image_storage.download, object_name)
    result = await asyncio.to_thread(
        run_ocr_and_parse_sync,
        image_data=image_data,
        filename=Path(object_name).name,
        use_lm=use_llm,
        llm_kwargs=llm_kwargs,
    )

    return {
        "result_json": {
            "items": result.get("items"),
            "stats": result.get("stats"),
            "type": result.get("type"),
        },
        "ocr_result": result.get("ocr_result"),
        "raw_llm_output": result.get("raw_llm_output"),
    }
