import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.router import public_router
from src.core.config import (
    APP_NAME,
    APP_VERSION,
    CORS_ALLOW_ORIGINS,
    LLM_POLL_INTERVAL_SECONDS,
    LOG_FILE,
    LOG_LEVEL,
    OCR_POLL_INTERVAL_SECONDS,
)
from src.workers.tasks import task_process_llm, task_process_ocr
from src.service.queue_scheduler import (
    configure_queue_handlers,
    start_queue_polling,
    scheduler,
    wake_queue,
)


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger(__name__)
async def _drain_ocr_queue() -> None:
    while True:
        result = await task_process_ocr()
        if result == "empty":
            return
        if result == "retry":
            return
        if result == "done":
            wake_queue("llm")


async def _drain_llm_queue() -> None:
    # A fixed cutoff visits each eligible row at most once per drain.
    cutoff = datetime.now(timezone.utc)
    for _ in range(25):
        result = await task_process_llm(eligible_before=cutoff)
        if result != "done":
            return


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_queue_handlers(_drain_ocr_queue, _drain_llm_queue)
    
    start_queue_polling(OCR_POLL_INTERVAL_SECONDS, LLM_POLL_INTERVAL_SECONDS)
    scheduler.start()
    logger.info("Event-driven job dispatcher started")
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


app = FastAPI(
    lifespan=lifespan,
    title=APP_NAME,
    description="API for queuing Lab Report extractions (async) or processing them immediately (sync)",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials="*" not in CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(public_router)


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=LOG_LEVEL.lower())
