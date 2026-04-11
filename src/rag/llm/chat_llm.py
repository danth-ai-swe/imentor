from __future__ import annotations

import threading
from typing import Dict, List

from src.config.app_config import _retry_policy, get_app_config
from src.rag.llm.embedding_llm import get_async_client, get_sync_client
from src.utils.logger_utils import alog_method_call, log_method_call

config = get_app_config()

# ── Thread-safe singletons ────────────────────────────────────────────────────
_chat_client_instance: AzureChatClient | None = None

_sync_lock = threading.Lock()
_async_lock = threading.Lock()
_chat_lock = threading.Lock()


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
    def _chat_params(messages: List[Dict], max_tokens: int | None = None) -> dict:
        return dict(
            model=config.OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=config.GPT_TEMPERATURE,
            top_p=config.GPT_TOP_P,
            max_tokens=max_tokens or config.GPT_MAX_TOKENS,
        )

    # ── Sync ──────────────────────────────────────────────────────────────────
    @log_method_call
    @_retry_policy()
    def invoke(self, prompt: str) -> str:
        """Single-turn text completion."""
        response = get_sync_client().chat.completions.create(
            **self._chat_params([{"role": "user", "content": prompt}])
        )
        return response.choices[0].message.content or ""

    @log_method_call
    @_retry_policy()
    def create_agentic_chunker_message(
            self,
            system_prompt: str,
            messages: List[Dict[str, str]],
            max_tokens: int | None = None,
    ) -> str:
        response = get_sync_client().chat.completions.create(
            **self._chat_params(
                self._build_messages(system_prompt, messages),
                max_tokens=max_tokens
            )
        )
        return response.choices[0].message.content or ""

    @log_method_call
    @_retry_policy()
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
    async def _achat(self, messages: List[Dict], max_tokens: int | None = None) -> str:
        response = await get_async_client().chat.completions.create(
            **self._chat_params(messages, max_tokens=max_tokens)
        )
        return response.choices[0].message.content

    # ── Async public ──────────────────────────────────────────────────────────
    @alog_method_call
    @_retry_policy()
    async def ainvoke(self, prompt: str) -> str:
        """Async single-turn text completion."""
        return await self._achat([{"role": "user", "content": prompt}])

    @alog_method_call
    @_retry_policy()
    async def acreate_agentic_chunker_message(
            self,
            system_prompt: str,
            messages: List[Dict[str, str]],
            max_tokens: int | None = None,
    ) -> str:
        return await self._achat(
            self._build_messages(system_prompt, messages),
            max_tokens=max_tokens
        )


# ── Singleton factory ─────────────────────────────────────────────────────────
def get_openai_chat_client() -> AzureChatClient:
    global _chat_client_instance
    if _chat_client_instance is None:
        with _chat_lock:
            if _chat_client_instance is None:
                _chat_client_instance = AzureChatClient()
    return _chat_client_instance
