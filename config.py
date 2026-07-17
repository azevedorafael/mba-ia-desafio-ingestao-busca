from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, validator
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    google_api_key: str | None = Field(None, env="GOOGLE_API_KEY")
    google_embedding_model: str = Field("models/embedding-004", env="GOOGLE_EMBEDDING_MODEL")
    google_llm_model: str = Field("gemini-2.5-flash", env="GOOGLE_LLM_MODEL")
    openai_api_key: str | None = Field(None, env="OPENAI_API_KEY")
    openai_embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_llm_model: str = Field("gpt-4o-mini", env="OPENAI_LLM_MODEL")
    database_url: str | None = Field(None, env="DATABASE_URL")
    pgvector_url: str = Field(..., env="PGVECTOR_URL")
    pgvector_collection: str = Field(..., env="PGVECTOR_COLLECTION")
    pdf_path: Path = Field(..., env="PDF_PATH")
    top_k: int = Field(10, env="TOP_K")

    class Config:
        env_file = ".env"
        case_sensitive = False
        validate_assignment = True

    @validator("pdf_path", pre=True)
    def normalize_pdf_path(cls, value: str | Path) -> Path:
        if not value:
            raise ValueError("PDF_PATH cannot be empty")
        return Path(value).expanduser().resolve()

    def embedding_provider(self) -> str:
        if self.google_api_key:
            return "google"
        if self.openai_api_key:
            return "openai"
        raise ValueError("At least one embedding provider must be configured: GOOGLE_API_KEY or OPENAI_API_KEY")

    def llm_provider(self) -> str | None:
        if self.google_api_key:
            return "google"
        if self.openai_api_key:
            return "openai"
        return None
