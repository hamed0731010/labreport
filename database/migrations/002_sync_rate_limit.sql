-- Apply once before starting API replicas with the sync rate limiter enabled.
BEGIN;
CREATE TABLE sync_rate_limits (
    scope varchar(32) PRIMARY KEY,
    next_allowed_at timestamptz NOT NULL
);
COMMIT;
