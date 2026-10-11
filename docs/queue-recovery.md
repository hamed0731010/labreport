# Queue ownership and deployment

The queue now uses a random ownership token and a renewable lease. Every success
and failure write checks that token, the processing stage and lease expiration
while holding the request row lock. A stale worker cannot overwrite a newer
attempt. OCR and LLM drains recover expired rows using `FOR UPDATE SKIP LOCKED`.
Recovery consumes the stage's retry budget and applies exponential backoff.
Heartbeat updates do not change `updated_at`; retry eligibility uses
`next_retry_at`. Both drains process at most `MAX_JOBS_PER_DRAIN` rows per run.

Sync requests are inserted directly as `ocr_processing`, with
`queue_managed=false`. They use the same fencing and heartbeat but an interrupted
sync request becomes failed; it is never silently retried by a background worker.
Successful sync OCR text is also available from the status endpoint.

## Existing database deployment

1. Back up the database and stop **all** API/worker processes, including other
   replicas. This migration is not compatible with running old workers.
2. Review old processing rows. Historical sync/async requests are indistinguishable
   in the old schema. The migration treats them as queued requests by default;
   mark known abandoned sync requests failed before restarting, if applicable.
3. Apply `database/migrations/001_job_leases.sql` once with `psql` and
   `ON_ERROR_STOP=1`, against the application's database/schema.
4. Start the new application. The first queue polls recover expired rows.

`Base.metadata.create_all()` only creates missing tables; it does not migrate
existing tables. For a brand-new database, initialize using the updated models
and do not apply the ALTER migration. To roll back code, stop all processes and
restore the backup; do not run the old workers against active leased jobs.

## Settings and operational limits

- `JOB_LEASE_SECONDS=120`: renewed every one-third of this duration. Keep clocks
  synchronized across hosts and choose a duration larger than normal event-loop
  stalls or short database interruptions.
- `MAX_JOBS_PER_DRAIN=25`: applies to OCR and LLM. Remaining work is picked up on
  subsequent polls.
- `MAX_JOB_RETRIES=3`: existing behavior counts failures, including expired
  attempts. The database now permits any nonnegative count, so configurations
  above three work. Successful OCR starts LLM with a fresh budget.

Cancellation stops the heartbeat and propagates; it does not immediately requeue
the request. `asyncio.to_thread` cannot terminate the underlying OCR/HTTP call.
Expired work may therefore execute externally more than once, but stale results
cannot commit. This is at-least-once execution, not exactly-once external effects.
Recovery runs at the start of each drain; a busy drain can delay recovery until
the next run. Leases detect lost workers, not a permanently hung computation whose
event loop still renews its lease. OCR/LLM service timeouts remain necessary.

## Regression tests

Install `tests/requirements.txt` in an isolated environment and run
`python -m pytest -q`. The fixture uses a unique temporary PostgreSQL schema and
stubbed OCR, image storage and LLM calls. Set `TEST_DATABASE_URL` to a disposable
`postgresql+asyncpg://...` database with schema creation permissions. Without it,
database tests are skipped. No production database or external model is used.
The tests map this repository root to the existing `src` package namespace,
matching the application's imports without changing deployment layout.
