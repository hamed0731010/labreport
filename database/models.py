from datetime import datetime
from typing import Any, Dict, Optional
import enum

from sqlalchemy import *

from sqlalchemy.dialects.postgresql import JSONB, ENUM
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class ExtractionStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    OCR_PROCESSING = "ocr_processing"
    OCR_COMPLETED = "ocr_completed"
    LM_PROCESSING = "lm_processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Prompt(Base):
    __tablename__ = "prompts"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_prompts_name_version"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    template_text: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    requests: Mapped[list["ExtractionRequest"]] = relationship(back_populates="prompt")

class ExtractionSchema(Base):
    __tablename__ = "schemas"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_schemas_name_version"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    schema_definition: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    requests: Mapped[list["ExtractionRequest"]] = relationship(back_populates="schema")

class ExtractionRequest(Base):
    __tablename__ = "lab_report_extraction_requests"
    __table_args__ = (
        CheckConstraint("retry_count >= 0 AND retry_count <= 3", name="chk_retry_count"),
        Index("idx_requests_polling", "created_at", postgresql_where=(func.lower("status") == "uploaded")),
        Index("idx_requests_status_created_at", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    image_id: Mapped[str] = mapped_column(String(255), nullable=False)
    minio_path: Mapped[str] = mapped_column(String(512), nullable=False)
    use_lm: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    prompt_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("prompts.id", ondelete="SET NULL"))
    schema_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("schemas.id", ondelete="SET NULL"))
    
    lm_model: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[ExtractionStatus] = mapped_column(
        ENUM(
            ExtractionStatus,
            name="extraction_status",
            create_type=True,
            values_callable=lambda obj: [e.value for e in obj] # <-- Add this line
        ),
        default=ExtractionStatus.UPLOADED,
        nullable=False
    )
    
    ocr_raw_text: Mapped[Optional[str]] = mapped_column(Text)
    ocr_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    prompt: Mapped[Optional[Prompt]] = relationship(back_populates="requests")
    schema: Mapped[Optional[ExtractionSchema]] = relationship(back_populates="requests")
    response: Mapped[Optional["ExtractionResponse"]] = relationship(back_populates="request")

class ExtractionResponse(Base):
    __tablename__ = "lab_report_extraction_responses"
    
    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    request_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("lab_report_extraction_requests.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    result_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    raw_llm_output: Mapped[Optional[str]] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    request: Mapped[ExtractionRequest] = relationship(back_populates="response")
