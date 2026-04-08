from typing import List, Dict, Any

import numpy as np
import openai
from openai import AsyncAzureOpenAI

from src.config.app_config import get_app_config, _retry_policy
from src.utils.logger_utils import alog_method_call, log_method_call

config = get_app_config()

_sync_client: openai.AzureOpenAI | None = None
_async_client: AsyncAzureOpenAI | None = None


def get_sync_client() -> openai.AzureOpenAI:
    global _sync_client
    if _sync_client is None:
        _sync_client = openai.AzureOpenAI(
            azure_endpoint=config.OPENAI_API_BASE,
            api_key=config.OPENAI_API_KEY,
            api_version=config.OPENAI_API_VERSION,
            timeout=config.GPT_TIMEOUT,
            max_retries=config.GPT_MAX_RETRIES,
        )
    return _sync_client


def _get_async_client() -> AsyncAzureOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncAzureOpenAI(
            azure_endpoint=config.OPENAI_API_BASE,
            api_key=config.OPENAI_API_KEY,
            api_version=config.OPENAI_API_VERSION,
            timeout=config.GPT_TIMEOUT,
            max_retries=config.GPT_MAX_RETRIES,
        )
    return _async_client


class AzureEmbeddingClient:
    @log_method_call
    @_retry_policy()
    def embed_query(self, text: str) -> List[float]:
        response = get_sync_client().embeddings.create(
            input=text,
            model=config.OPENAI_EMBEDDING_MODEL,
            encoding_format="float",
        )
        return response.data[0].embedding

    @log_method_call
    @_retry_policy()
    def embed_query_full(self, text: str) -> Dict[str, Any]:
        response = get_sync_client().embeddings.create(
            input=[text],
            model=config.OPENAI_EMBEDDING_MODEL,
            encoding_format="float",
        )
        return response.model_dump()

    @staticmethod
    def embed_documents(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = get_sync_client().embeddings.create(
            input=texts,
            model=config.OPENAI_EMBEDDING_MODEL,
            encoding_format="float",
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]

    @alog_method_call
    @_retry_policy()
    async def aembed_query(self, text: str) -> List[float]:
        response = await _get_async_client().embeddings.create(
            input=text,
            model=config.OPENAI_EMBEDDING_MODEL,
            encoding_format="float",
        )
        return response.data[0].embedding

    @alog_method_call
    @_retry_policy()
    async def aembed_query_full(self, text: str) -> Dict[str, Any]:
        response = await _get_async_client().embeddings.create(
            input=[text],
            model=config.OPENAI_EMBEDDING_MODEL,
            encoding_format="float",
        )
        return response.model_dump()

    @alog_method_call
    @_retry_policy()
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = await _get_async_client().embeddings.create(
            input=texts,
            model=config.OPENAI_EMBEDDING_MODEL,
            encoding_format="float",
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]

    @alog_method_call
    @_retry_policy()
    async def aembed_query_byte(self, text: str) -> List[int]:
        floats = await self.aembed_query(text)
        vec = np.array(floats, dtype=np.float32)

        scale = np.quantile(np.abs(vec), 0.99)
        if scale == 0:
            scale = 1.0

        quantized = np.clip(np.round(vec / scale * 127), -128, 127).astype(np.int8)
        return quantized.tolist()


_embedding_client_instance: AzureEmbeddingClient | None = None


def get_openai_embedding_client() -> AzureEmbeddingClient:
    global _embedding_client_instance
    if _embedding_client_instance is None:
        _embedding_client_instance = AzureEmbeddingClient()
    return _embedding_client_instance
