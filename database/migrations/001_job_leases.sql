-- Stop ALL old API/worker processes before running this migration.
-- Apply once using psql -v ON_ERROR_STOP=1 -f <this file>.
BEGIN;
ALTER TABLE lab_report_extraction_requests
    ADD COLUMN next_retry_at timestamptz,
    ADD COLUMN lease_token varchar(36),
    ADD COLUMN lease_expires_at timestamptz,
    ADD COLUMN queue_managed boolean NOT NULL DEFAULT true;

-- Preserve outstanding retry delays from the old overloaded timestamp.
UPDATE lab_report_extraction_requests
SET next_retry_at = updated_at, updated_at = CURRENT_TIMESTAMP
WHERE status IN ('uploaded', 'ocr_completed') AND updated_at > CURRENT_TIMESTAMP;

-- Old processing rows have no owner token. With old workers stopped they
-- are safe to recover on the next poll (including consumption of retry budget).
UPDATE lab_report_extraction_requests
SET lease_expires_at = CURRENT_TIMESTAMP
WHERE status IN ('ocr_processing', 'lm_processing');

ALTER TABLE lab_report_extraction_requests DROP CONSTRAINT chk_retry_count;
ALTER TABLE lab_report_extraction_requests ADD CONSTRAINT chk_retry_count
    CHECK (retry_count >= 0);
DROP INDEX IF EXISTS idx_requests_polling;
CREATE INDEX idx_requests_lease_expiry
    ON lab_report_extraction_requests (lease_expires_at);
COMMIT;
