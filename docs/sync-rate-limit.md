# Sync request rate limit

`/extract/sync` admits at most one new request every `SYNC_RATE_LIMIT_SECONDS`
seconds (default **180**). This is one service-wide budget shared by all clients,
models, and API processes/replicas using the same PostgreSQL database/schema.
It applies with or without LLM. `/extract/async` is not rate-limited by this change.

The default uses the reported mean of 30 seconds OCR plus 150 seconds LLM as a
conservative starting point; tune it with measured service capacity. Set a positive
integer in the environment, for example `SYNC_RATE_LIMIT_SECONDS=180`, consistently
on every replica. Restart replicas after changing it. An existing cooldown remains
in force; the new interval takes effect on the next accepted request.

Excess requests receive HTTP **429** with a `Retry-After` header containing whole
seconds. Clients should wait at least that long before retrying. A retry is subject
to competition from other clients, so admission is not guaranteed at that time.
There is no in-process waiting queue. The limiter runs after filename/prompt/schema
validation, but before MinIO upload, extraction-request insertion, or OCR/LLM work.
FastAPI may already have received and parsed the multipart upload, so this does not
limit incoming upload bandwidth. Accepted attempts consume their slot even if
storage or processing later fails; rejected attempts do not extend the cooldown.

The PostgreSQL conditional upsert serializes concurrent admissions and uses the
database clock. State survives API restarts and no connection/lock is held during
processing. Database/limiter errors fail closed with **503**, never unlimited access.

This limits starts, not total concurrent LLM calls. A Sync job taking longer than
the interval can overlap the next Sync job, and background Async jobs have their
own scheduling. A strict combined concurrency cap would require shared admission
across both paths or a limiter in LLM Runner itself.

## Deployment

For existing databases, apply `database/migrations/002_sync_rate_limit.sql` once,
after migration 001 if it has not been applied. Deploy the new code to all replicas;
old replicas do not enforce this limit. For a fresh database, updated
`Base.metadata.create_all()` creates the table; do not apply the CREATE migration
again. Never clear the limiter row to reset capacity during normal operation.
