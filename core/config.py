"""Application settings loaded from the selected environment file."""

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
ENV = (
    os.getenv("APP_ENV")
    or os.getenv("ENV")
    or os.getenv("ENVIRONMENT")
    or "development"
).strip().lower()
ENV_FILES = {
    "development": ".env.development",
    "staging": ".env.staging",
    "beta": ".env.beta",
    "test": ".env.test",
}
ENVIRONMENT = ENV  # Keep existing settings that reference ENVIRONMENT compatible.

if ENV not in ENV_FILES:
    supported = ", ".join(sorted(ENV_FILES))
    raise ValueError(f"Unsupported APP_ENV {ENV!r}; choose one of: {supported}")

# Runtime environment variables take precedence over values in the selected file.
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ENV_FILES[ENV])                      

def _int_setting(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _bool_setting(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise ValueError(f"{name} must be a boolean value")
    return value in {"1", "true", "yes", "on"}


DB_USER = os.getenv("DB_USER", "LabExtractorUser")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "LabExtractor")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)
DB_ECHO = _bool_setting("DB_ECHO")
DB_POOL_SIZE = _int_setting("DB_POOL_SIZE", 10)
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_CONNECT_TIMEOUT_SECONDS = _int_setting("DB_CONNECT_TIMEOUT_SECONDS", 10)
if DB_MAX_OVERFLOW < 0:
    raise ValueError("DB_MAX_OVERFLOW cannot be negative")

_minio_endpoint_setting = os.getenv("MINIO_ENDPOINT", "localhost:9000").rstrip("/")
MINIO_SECURE = _bool_setting(
    "MINIO_SECURE",
    _minio_endpoint_setting.startswith("https://") or ENVIRONMENT == "production",
)
MINIO_CERTIFICATE = _bool_setting(
    "MINIO_CERTIFICATE", True
)
MINIO_ENDPOINT = _minio_endpoint_setting.removeprefix("https://").removeprefix("http://")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "lab-report-extraction")
MINIO_OBJECT_PREFIX = os.getenv("MINIO_OBJECT_PREFIX", "lab-reports").strip("/")
MINIO_AUTO_CREATE_BUCKET = _bool_setting(
    "MINIO_AUTO_CREATE_BUCKET", ENVIRONMENT != "production"
)
if ENVIRONMENT == "production":
    required_minio_values = (
        os.getenv("MINIO_ENDPOINT"),
        os.getenv("MINIO_ACCESS_KEY"),
        os.getenv("MINIO_SECRET_KEY"),
    )
    if any(not value or value.startswith("replace-with-") for value in required_minio_values):
        raise ValueError("Set a real MinIO endpoint and credentials for production")
ALLOWED_EXTENSIONS = frozenset(
    extension.strip().lower()
    for extension in os.getenv(
        "ALLOWED_EXTENSIONS", ".jpg,.jpeg,.png,.gif,.bmp,.webp"
    ).split(",")
    if extension.strip()
)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dev-lmrunner.si24.ir").rstrip("/")
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "500"))
if LLM_REQUEST_TIMEOUT <= 0:
    raise ValueError("LLM_REQUEST_TIMEOUT must be greater than zero")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-vl-4b-instruct-q4_k_m:latest")
ASYNC_LLM_PROMPT = os.getenv("ASYNC_LLM_PROMPT", "rows2json_v2_fa")
ASYNC_SCHEMA_NAME = os.getenv("ASYNC_SCHEMA_NAME", "LabReportDetailed")
SYNC_LLM_PROMPT = os.getenv("SYNC_LLM_PROMPT", "rows2json_v2_fa")
SYNC_SCHEMA_NAME = os.getenv("SYNC_SCHEMA_NAME", "LabReportDetailed")
SYNC_RATE_LIMIT_SECONDS = _int_setting("SYNC_RATE_LIMIT_SECONDS", 180)
OCR_POLL_INTERVAL_SECONDS = _int_setting("OCR_POLL_INTERVAL_SECONDS", 5)
LLM_POLL_INTERVAL_SECONDS = _int_setting("LLM_POLL_INTERVAL_SECONDS", 5)
MAX_JOBS_PER_DRAIN = _int_setting("MAX_JOBS_PER_DRAIN", 25)
JOB_LEASE_SECONDS = _int_setting("JOB_LEASE_SECONDS", 120)
MAX_JOB_RETRIES = _int_setting("MAX_JOB_RETRIES", 3)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "app_api.log")
APP_NAME = os.getenv("APP_NAME", "Lab Report OCR Extractor API")
APP_VERSION = os.getenv("APP_VERSION", "2.2.0")
CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]
