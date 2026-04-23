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
    SEARX_HOST: str
    CF_ACCESS_CLIENT_ID: str
    CF_ACCESS_CLIENT_SECRET: str
    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
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
