from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    localrag_api_key: str = ""

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "localrag_documents"

    ollama_base_url: str = "http://ollama:11434"

    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"
    ollama_num_ctx: int = Field(default=6144, ge=2048, le=32768)
    embedding_model: str = "nomic-embed-text"

    database_url: str | None = None
    database_host: str = "postgres"
    database_port: int = 5432
    database_name: str = "localrag"
    database_user: str = "localrag"
    database_password: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
