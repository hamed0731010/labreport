import asyncio
import logging
import mimetypes
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import UUID

from minio import Minio
from minio.error import S3Error
from fastapi import HTTPException, UploadFile
from urllib3 import PoolManager

from src.core.config import (
    MINIO_ACCESS_KEY,
    MINIO_AUTO_CREATE_BUCKET,
    MINIO_BUCKET,
    MINIO_CERTIFICATE,
    MINIO_ENDPOINT,
    MINIO_OBJECT_PREFIX,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
)

logger = logging.getLogger(__name__)


class MinioImageStorage:
    def __init__(self) -> None:
        self.client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
            http_client=PoolManager(
                cert_reqs="CERT_REQUIRED" if MINIO_CERTIFICATE else "CERT_NONE"
            ),
        )

    def ensure_bucket(self) -> None:
        if self.client.bucket_exists(MINIO_BUCKET):
            return
        if not MINIO_AUTO_CREATE_BUCKET:
            raise RuntimeError(f"Configured MinIO bucket '{MINIO_BUCKET}' does not exist")
        try:
            self.client.make_bucket(MINIO_BUCKET)
        except S3Error as error:
            # Another process may have created the bucket at the same time.
            if error.code not in {"BucketAlreadyExists", "BucketAlreadyOwnedByYou"}:
                raise

    @staticmethod
    def object_name(image_id: UUID, filename: str) -> str:
        extension = Path(filename).suffix.lower()
        month_path = datetime.now(timezone.utc).strftime("%Y/%m")
        parts = [part for part in (MINIO_OBJECT_PREFIX, month_path, f"{image_id}{extension}") if part]
        return "/".join(parts)

    def upload(self, object_name: str, content: bytes, content_type: str) -> None:
        self.ensure_bucket()
        self.client.put_object(
            MINIO_BUCKET,
            object_name,
            BytesIO(content),
            length=len(content),
            content_type=content_type,
        )

    async def store_upload(self, file: UploadFile) -> tuple[str, str]:
        """Store an uploaded file and return its image ID and object key."""
        image_uuid = uuid.uuid4()
        image_id = str(image_uuid)
        filename = file.filename or "upload"
        object_name = self.object_name(image_uuid, filename)

        try:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            content_type = file.content_type or mimetypes.guess_type(filename)[0]
            await asyncio.to_thread(
                self.upload,
                object_name,
                content,
                content_type or "application/octet-stream",
            )
        except HTTPException:
            raise
        except Exception as error:
            logger.exception("Failed to upload image to MinIO")
            raise HTTPException(
                status_code=503, detail="Image storage unavailable"
            ) from error
        finally:
            await file.close()

        return image_id, object_name

    def download(self, object_name: str) -> bytes:
        response = self.client.get_object(MINIO_BUCKET, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()


image_storage = MinioImageStorage()
