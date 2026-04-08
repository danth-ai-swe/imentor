from typing import List, Dict, Any

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
        )
    return _async_client


class AzureChatClient:
    @log_method_call
    @_retry_policy()
    def invoke(self, prompt: str) -> str:
        """Simple single-turn text completion."""
        response = get_sync_client().chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.GPT_TEMPERATURE,
            top_p=config.GPT_TOP_P,
            max_tokens=config.GPT_MAX_TOKENS,
        )
        return response.choices[0].message.content

    @log_method_call
    @_retry_policy()
    def invoke_full(self, prompt: str) -> Dict[str, Any]:
        """Single-turn completion — returns full response dict."""
        response = get_sync_client().chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.GPT_TEMPERATURE,
            top_p=config.GPT_TOP_P,
            max_tokens=config.GPT_MAX_TOKENS,
        )
        return response.model_dump()

    @log_method_call
    @_retry_policy()
    def create_agentic_chunker_message(
            self,
            system_prompt: str,
            messages: List[Dict[str, str]],
    ) -> str:
        """Multi-turn conversation with an explicit system prompt."""
        chat_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            chat_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
        response = get_sync_client().chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=chat_messages,
            temperature=config.GPT_TEMPERATURE,
            top_p=config.GPT_TOP_P,
            max_tokens=config.GPT_MAX_TOKENS,
        )
        return response.choices[0].message.content

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
            model=config.OPENAI_CHAT_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=config.GPT_TEMPERATURE,
            top_p=config.GPT_TOP_P,
            max_tokens=config.GPT_MAX_TOKENS,
        )
        return response.choices[0].message.content

    # ── Async ────────────────────────────────────────────────────────────────

    @alog_method_call
    @_retry_policy()
    async def acreate_agentic_chunker_message(
            self,
            system_prompt: str,
            messages: List[Dict[str, str]],
    ) -> str:
        """Async multi-turn conversation with an explicit system prompt."""
        chat_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            chat_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
        response = await _get_async_client().chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=chat_messages,
            temperature=config.GPT_TEMPERATURE,
            top_p=config.GPT_TOP_P,
            max_tokens=config.GPT_MAX_TOKENS,
        )
        return response.choices[0].message.content

    @alog_method_call
    @_retry_policy()
    async def acreate_json_message(
            self,
            system_prompt: str,
            messages: List[Dict[str, str]],
            max_tokens: int | None = None,
    ) -> str:
        """
        Async JSON mode — ``response_format=json_object``.

        Đảm bảo output là valid JSON, không cần strip markdown fences.
        Hỗ trợ tuỳ chỉnh ``max_tokens`` cho từng use-case (vd: quiz batch).
        """
        chat_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            chat_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
        response = await _get_async_client().chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=chat_messages,
            temperature=config.GPT_TEMPERATURE,
            top_p=config.GPT_TOP_P,
            max_tokens=max_tokens or config.GPT_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    @alog_method_call
    @_retry_policy()
    async def ainvoke(self, prompt: str) -> str:
        """Async single-turn text completion."""
        response = await _get_async_client().chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.GPT_TEMPERATURE,
            top_p=config.GPT_TOP_P,
            max_tokens=config.GPT_MAX_TOKENS,
        )
        return response.choices[0].message.content

    @alog_method_call
    @_retry_policy()
    async def ainvoke_full(self, prompt: str) -> Dict[str, Any]:
        """Async single-turn completion — returns full response dict."""
        response = await _get_async_client().chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.GPT_TEMPERATURE,
            top_p=config.GPT_TOP_P,
            max_tokens=config.GPT_MAX_TOKENS,
        )
        return response.model_dump()

    @alog_method_call
    @_retry_policy()
    async def ainvoke_with_image(
            self,
            prompt: str,
            image_base64: str,
            media_type: str = "image/jpeg",
    ) -> str:
        """Vision API — async."""
        response = await _get_async_client().chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=config.GPT_TEMPERATURE,
            top_p=config.GPT_TOP_P,
            max_tokens=config.GPT_MAX_TOKENS,
        )
        return response.choices[0].message.content


_chat_client_instance: AzureChatClient | None = None


def get_openai_chat_client() -> AzureChatClient:
    global _chat_client_instance
    if _chat_client_instance is None:
        _chat_client_instance = AzureChatClient()
    return _chat_client_instance
