import asyncio
import hashlib
import threading
import time
from collections import OrderedDict
from typing import AsyncGenerator, Dict, List, Optional

from src.config.app_config import retry_policy, get_app_config
from src.rag.llm.embedding_llm import get_async_client, get_sync_client
from src.utils.logger_utils import alog_method_call, log_method_call

config = get_app_config()

# ── Thread-safe singletons ────────────────────────────────────────────────────
_sync_lock = threading.Lock()
_async_lock = threading.Lock()
_chat_lock = threading.Lock()

# ── Prompt-response TTL LRU cache ─────────────────────────────────────────────
# Safe because GPT_TEMPERATURE = 0 → deterministic output.
_PROMPT_CACHE_SIZE = 256
_PROMPT_CACHE_TTL = 1_800  # 30 min

_prompt_cache: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
_prompt_cache_lock = asyncio.Lock()


def _prompt_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


async def _prompt_cache_get(prompt: str) -> Optional[str]:
    key = _prompt_key(prompt)
    now = time.monotonic()
    async with _prompt_cache_lock:
        entry = _prompt_cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if now - ts > _PROMPT_CACHE_TTL:
            _prompt_cache.pop(key, None)
            return None
        _prompt_cache.move_to_end(key)
        return value


async def _prompt_cache_put(prompt: str, value: str) -> None:
    if not value:
        return
    key = _prompt_key(prompt)
    now = time.monotonic()
    async with _prompt_cache_lock:
        _prompt_cache[key] = (now, value)
        _prompt_cache.move_to_end(key)
        while len(_prompt_cache) > _PROMPT_CACHE_SIZE:
            _prompt_cache.popitem(last=False)


def _client_kwargs() -> dict:
    return dict(
        azure_endpoint=config.OPENAI_API_BASE,
        api_key=config.OPENAI_API_KEY,
        api_version=config.OPENAI_API_VERSION,
        timeout=config.GPT_TIMEOUT,
        max_retries=config.GPT_MAX_RETRIES,
    )


# ── AzureChatClient ───────────────────────────────────────────────────────────
class AzureChatClient:

    # ── Shared helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _build_messages(
            system_prompt: str,
            messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            *[
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ],
        ]

    @staticmethod
    def _chat_params(
            messages: List[Dict],
            max_tokens: int | None = None,
            temperature: float | None = None,
            top_p: float | None = None,
    ) -> dict:
        return dict(
            model=config.OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=config.GPT_TEMPERATURE if temperature is None else temperature,
            top_p=config.GPT_TOP_P if top_p is None else top_p,
            max_tokens=max_tokens or config.GPT_MAX_TOKENS,
        )

    # ── Sync ──────────────────────────────────────────────────────────────────
    @log_method_call
    @retry_policy()
    def invoke(self, prompt: str) -> str:
        """Single-turn text completion."""
        response = get_sync_client().chat.completions.create(
            **self._chat_params([{"role": "user", "content": prompt}])
        )
        return response.choices[0].message.content or ""

    @log_method_call
    @retry_policy()
    def invoke_with_image(
            self,
            prompt: str,
            image_base64: str,
            media_type: str = "image/jpeg",
    ) -> str:
        """Vision API — sync."""
        response = get_sync_client().chat.completions.create(
            **self._chat_params([{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                    {"type": "text", "text": prompt},
                ],
            }])
        )
        return response.choices[0].message.content

    # ── Async core ────────────────────────────────────────────────────────────
    async def achat(self, messages: List[Dict], max_tokens: int | None = None) -> str:
        response = await get_async_client().chat.completions.create(
            **self._chat_params(messages, max_tokens=max_tokens)
        )
        return response.choices[0].message.content

    def chat(self, messages: List[Dict], max_tokens: int | None = None) -> str:
        response = get_sync_client().chat.completions.create(
            **self._chat_params(messages, max_tokens=max_tokens)
        )
        return response.choices[0].message.content

    # ── Async public ──────────────────────────────────────────────────────────
    @alog_method_call
    @retry_policy()
    async def ainvoke(self, prompt: str) -> str:
        """Async single-turn text completion (TTL-cached, temp=0)."""
        cached = await _prompt_cache_get(prompt)
        if cached is not None:
            return cached
        result = await self.achat([{"role": "user", "content": prompt}])
        await _prompt_cache_put(prompt, result)
        return result

    @alog_method_call
    @retry_policy()
    async def ainvoke_creative(
            self,
            prompt: str,
            temperature: float = 0.7,
            top_p: float = 0.95,
            max_tokens: int | None = None,
    ) -> str:
        """Async single-turn completion with a higher temperature for more
        creative / varied prose (e.g. the markdown final-answer step).
        Bypasses the prompt cache so repeated queries don't collapse back to
        the first sampled response.
        """
        response = await get_async_client().chat.completions.create(
            **self._chat_params(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )
        return response.choices[0].message.content or ""

    async def astream_creative(
            self,
            prompt: str,
            temperature: float = 0.7,
            top_p: float = 0.95,
            max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming variant of ainvoke_creative. Yields text deltas as Azure
        emits them — lets callers ship tokens to the client for faster
        perceived latency. Not cached (streams are inherently single-use).
        """
        response = await get_async_client().chat.completions.create(
            **self._chat_params(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
            stream=True,
        )
        async for event in response:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece


_chat_client_instance: AzureChatClient | None = None


# ── Singleton factory ─────────────────────────────────────────────────────────
def get_openai_chat_client() -> AzureChatClient:
    global _chat_client_instance
    if _chat_client_instance is None:
        with _chat_lock:
            if _chat_client_instance is None:
                _chat_client_instance = AzureChatClient()
    return _chat_client_instance