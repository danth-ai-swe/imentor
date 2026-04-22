from functools import lru_cache
from pathlib import Path
from typing import Optional
import openai


from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential,
)

env_path = Path(__file__).parent.parent.parent / ".env"


class AppConfig(BaseSettings):
    GIT_COMMIT_ID: str = "unknown"
    DEFINITION_NAME: str = "unknown"
    APP_BUILD_NUMBER: str = "unknown"
    PROJECT_NAME: str = "OKR AI Agent"
    PROFILE_NAME: str = "default"
    DEBUG: bool = False

    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_API_VERSION: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    GPT_TEMPERATURE: float = 0
    GPT_TOP_P: float = 1
    GPT_MAX_TOKENS: int = 3500
    GPT_TIMEOUT: int = 30
    GPT_MAX_RETRIES: int = 3
    TIME_OUT_API: int = 30

    CHAT_HISTORY_API_BASE: str = "http://10.98.36.75:8081"
    APP_DOMAIN: str = "https://api.fpt-apps.com/imt-ai-brain"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_APIKEY: str
    # ───────────────── RabbitMQ ─────────────────
    RABBITMQ_USER: str
    RABBITMQ_PASS: str

    # ───────────────── Redis ─────────────────
    REDIS_PASSWORD: str

    # ───────────────── MinIO ─────────────────
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_ACCESS_KEY: str

    # ───────────────── MongoDB ─────────────────
    MONGO_INITDB_ROOT_USERNAME: str
    MONGO_INITDB_ROOT_PASSWORD: str

    # ───────────────── Postgres ─────────────────
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    # ───────────────── PgAdmin ─────────────────
    PGADMIN_DEFAULT_EMAIL: str
    PGADMIN_DEFAULT_PASSWORD: str
    PGADMIN_PORT: int = 5050

    # ───────────────── Grafana ─────────────────
    GF_ADMIN_USER: str
    GF_ADMIN_PASSWORD: str

    # ───────────────── Custom Auth ─────────────────
    ME_USERNAME: str
    ME_PASSWORD: str

    # ───────────────── ClickHouse ─────────────────
    CLICKHOUSE_USER: str
    CLICKHOUSE_PASSWORD: str

    # ───────────────── Langfuse ─────────────────
    LANGFUSE_SALT: str
    LANGFUSE_ENCRYPTION_KEY: str
    LANGFUSE_NEXTAUTH_SECRET: str
    NEXTAUTH_URL: str

    LANGFUSE_INIT_USER_EMAIL: str
    LANGFUSE_INIT_USER_NAME: str
    LANGFUSE_INIT_USER_PASSWORD: str

    LANGFUSE_INIT_ORG_ID: str
    LANGFUSE_INIT_ORG_NAME: str

    LANGFUSE_INIT_PROJECT_ID: str
    LANGFUSE_INIT_PROJECT_NAME: str
    LANGFUSE_INIT_PROJECT_PUBLIC_KEY: str
    LANGFUSE_INIT_PROJECT_SECRET_KEY: str

    LANGFUSE_SECRET_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_BASE_URL: str

    # ───────────────── Optional ─────────────────
    TELEMETRY_ENABLED: bool = True
    LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: bool = False

    EMAIL_FROM_ADDRESS: Optional[str] = None
    SMTP_CONNECTION_URL: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache()
def get_app_config() -> AppConfig:
    return AppConfig()


def retry_policy():
    return retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True,
    )
