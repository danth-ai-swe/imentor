from functools import lru_cache
from pathlib import Path
from typing import Optional

import openai
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type, wait_exponential,
)

env_path = Path(__file__).parent.parent.parent / '.env'


class AppConfig(BaseSettings):
    GIT_COMMIT_ID: str = "unknown"
    DEFINITION_NAME: str = "unknown"
    APP_BUILD_NUMBER: str = "unknown"

    PROJECT_NAME: str = "OKR AI Agent"
    PROFILE_NAME: str = "default"
    DEBUG: bool = False

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = 20
    REDIS_DEFAULT_TTL: int = 3600

    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASS: str = "guest"

    MQ_ORDERS_EXCHANGE: str = "publish.exchange"
    MQ_ORDERS_QUEUE: str = "publish.queue"
    MQ_ORDERS_KEY: str = "publish.routing_key"

    MQ_NOTIF_EXCHANGE: str = "notifications.exchange"
    MQ_NOTIF_QUEUE: str = "notifications.queue"
    MQ_NOTIF_KEY: str = "notif.send"

    MQ_DLX_NAME: str = "dlx"
    MQ_DLQ_NAME: str = "dlq"

    MQ_MESSAGE_TTL_MS: int = 3_600_000
    MQ_MAX_QUEUE_LENGTH: int = 10_000

    MQ_PUBLISH_CONN_NAME: str = "app.publisher"
    MQ_CONSUME_CONN_NAME: str = "app.consumer"

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
    IP_ADDRESS: str = "10.98.36.83"

    CHAT_HISTORY_API_BASE: str = "http://10.98.36.75:8081"
    APP_IP: str = "127.0.0.1"
    APP_PORT: int = 8080
    APP_DOMAIN: str = "https://api.fpt-apps.com/imt-ai-brain"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_APIKEY: str
    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def RABBITMQ_URL(self) -> str:
        return (
            f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASS}"
            f"@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/"
        )


@lru_cache()
def get_app_config() -> AppConfig:
    return AppConfig()


def _retry_policy():
    return retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True,
    )