import asyncio
import base64
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from typing import List, Dict, Any

import numpy as np
import openai
from openai import AsyncAzureOpenAI

from src.config.app_config import retry_policy, get_app_config
from src.utils.logger_utils import alog_method_call

config = get_app_config()

# ── Thread-safe singletons ────────────────────────────────────────────────────
_sync_client: openai.AzureOpenAI | None = None
_async_client: AsyncAzureOpenAI | None = None

_sync_lock = threading.Lock()
_async_lock = threading.Lock()
_embed_lock = threading.Lock()


def _client_kwargs() -> dict:
    return dict(
        azure_endpoint=config.OPENAI_API_BASE,
        api_key=config.OPENAI_API_KEY,
        api_version=config.OPENAI_API_VERSION,
        timeout=config.GPT_TIMEOUT,
        max_retries=config.GPT_MAX_RETRIES,
    )


def get_sync_client() -> openai.AzureOpenAI:
    global _sync_client
    if _sync_client is None:
        with _sync_lock:
            if _sync_client is None:
                _sync_client = openai.AzureOpenAI(**_client_kwargs())
    return _sync_client


def get_async_client() -> AsyncAzureOpenAI:
    global _async_client
    if _async_client is None:
        with _async_lock:
            if _async_client is None:
                _async_client = AsyncAzureOpenAI(**_client_kwargs())
    return _async_client


# ── Normalize ─────────────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    return text.strip()


# ── Base64 decode ─────────────────────────────────────────────────────────────
def _decode_base64(b64: str) -> List[float]:
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
    return arr.tolist()


# ── Cache ─────────────────────────────────────────────────────────────────────
_CACHE_SIZE = 10_000


@lru_cache(maxsize=_CACHE_SIZE)
def _cached_embed_single(text: str) -> tuple[float, ...]:
    text = _normalize(text)

    raw = get_sync_client().embeddings.create(
        input=[text],
        model=config.OPENAI_EMBEDDING_MODEL,
        encoding_format="base64",
    )
    return tuple(_decode_base64(raw.data[0].embedding))


# ── Async LRU cache ───────────────────────────────────────────────────────────
_async_cache: "OrderedDict[str, List[float]]" = OrderedDict()
_async_cache_lock = asyncio.Lock()


async def _aget_from_cache(text: str):
    async with _async_cache_lock:
        value = _async_cache.get(text)
        if value is not None:
            _async_cache.move_to_end(text)
        return value


async def _aset_cache(text: str, value: List[float]):
    async with _async_cache_lock:
        _async_cache[text] = value
        _async_cache.move_to_end(text)
        while len(_async_cache) > _CACHE_SIZE:
            _async_cache.popitem(last=False)


# ── Batch config ──────────────────────────────────────────────────────────────
_BATCH_SIZE: int = getattr(config, "OPENAI_EMBEDDING_BATCH_SIZE", 512)
_CONCURRENCY_LIMIT: int = getattr(config, "OPENAI_EMBEDDING_CONCURRENCY", 5)

_semaphore = asyncio.Semaphore(_CONCURRENCY_LIMIT)

# ── Metrics ───────────────────────────────────────────────────────────────────
_metrics = {
    "cache_hits": 0,
    "cache_misses": 0,
    "total_calls": 0,
    "total_latency": 0.0,
    "batch_sizes": [],
}


def get_metrics():
    return {
        **_metrics,
        "avg_latency": (
            _metrics["total_latency"] / _metrics["total_calls"]
            if _metrics["total_calls"]
            else 0
        ),
    }


# ── Client ────────────────────────────────────────────────────────────────────
class AzureEmbeddingClient:

    @staticmethod
    def embed_documents(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        return [list(_cached_embed_single(_normalize(t))) for t in texts]

    @alog_method_call
    async def aembed_query(self, text: str) -> List[float]:
        results = await self.aembed_documents([text])
        return results[0]

    @retry_policy()
    @alog_method_call
    async def aembed_documents(self, texts: List[str]) -> list[Any] | list[None]:
        if not texts:
            return []

        start_time = time.perf_counter()
        _metrics["total_calls"] += 1

        texts = [_normalize(t) for t in texts]

        # ── check async cache ────────────────────────────────────────────────
        cached_results = []
        uncached_texts = []
        index_map = {}

        for i, t in enumerate(texts):
            cached = await _aget_from_cache(t)
            if cached is not None:
                _metrics["cache_hits"] += 1
                cached_results.append((i, cached))
            else:
                _metrics["cache_misses"] += 1
                index_map[len(uncached_texts)] = i
                uncached_texts.append(t)

        # ── batching ────────────────────────────────────────────────────────
        batches = [
            uncached_texts[i: i + _BATCH_SIZE]
            for i in range(0, len(uncached_texts), _BATCH_SIZE)
        ]

        _metrics["batch_sizes"].extend([len(b) for b in batches])

        async def _fetch(batch: List[str]) -> List[List[float]]:
            async with _semaphore:
                resp = await get_async_client().embeddings.create(
                    input=batch,
                    model=config.OPENAI_EMBEDDING_MODEL,
                    encoding_format="base64",
                )
                ordered = sorted(resp.data, key=lambda x: x.index)
                return [_decode_base64(d.embedding) for d in ordered]

        # ── fetch uncached ──────────────────────────────────────────────────
        fetched = []
        if batches:
            batch_results = await asyncio.gather(*[_fetch(b) for b in batches])
            fetched = [emb for batch in batch_results for emb in batch]

        # ── merge results ────────────────────────────────────────────────────
        results = [None] * len(texts)

        # cached
        for i, val in cached_results:
            results[i] = val

        # fetched
        for j, val in enumerate(fetched):
            original_idx = index_map[j]
            results[original_idx] = val
            await _aset_cache(texts[original_idx], val)

        # ── metrics ─────────────────────────────────────────────────────────
        latency = time.perf_counter() - start_time
        _metrics["total_latency"] += latency

        return results


_embedding_client_instance: AzureEmbeddingClient | None = None


def get_openai_embedding_client() -> AzureEmbeddingClient:
    global _embedding_client_instance
    if _embedding_client_instance is None:
        with _embed_lock:
            if _embedding_client_instance is None:
                _embedding_client_instance = AzureEmbeddingClient()
    return _embedding_client_instance
