"""Queue scheduler shared by the application and API."""

from datetime import datetime, timezone
from collections.abc import Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler


QueueHandler = Callable[[], Awaitable[None]]

scheduler = AsyncIOScheduler(
    job_defaults={
        "max_instances": 1,
        "coalesce": True,
    }
)

_queue_handlers: dict[str, QueueHandler] = {}


def configure_queue_handlers(
    ocr_handler: QueueHandler,
    llm_handler: QueueHandler,
) -> None:
    _queue_handlers["ocr"] = ocr_handler
    _queue_handlers["llm"] = llm_handler


def wake_queue(stage: str) -> None:
    if stage not in ("ocr", "llm"):
        raise ValueError(f"Unknown queue stage: {stage}")

    job_id = f"{stage}_queue_drain"

    if scheduler.get_job(job_id):
        scheduler.modify_job(
            job_id,
            next_run_time=datetime.now(timezone.utc),
        )


def start_queue_polling(
    ocr_interval_seconds: int,
    llm_interval_seconds: int,
) -> None:

    for stage, interval in (
        ("ocr", ocr_interval_seconds),
        ("llm", llm_interval_seconds),
    ):
        if interval <= 0:
            raise ValueError("Queue interval must be positive")

        handler = _queue_handlers.get(stage)

        if handler is None:
            raise RuntimeError(
                f"No handler configured for {stage}"
            )

        scheduler.add_job(
            handler,
            trigger="interval",
            seconds=interval,
            id=f"{stage}_queue_drain",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            next_run_time=datetime.now(timezone.utc),
        )