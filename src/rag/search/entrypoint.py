from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import httpx

from src.config.app_config import get_app_config
from src.constants.app_constant import (
    CHARS_PER_TOKEN,
    CHAT_HISTORY_TOKEN_BUDGET,
    CHAT_HISTORY_TIMEOUT,
    MAX_ASSISTANT_RESPONSE_CHARS,
    TRUNCATION_SUFFIX,
)
from src.rag.search.prompt import SYSTEM_PROMPT_TEMPLATE
from src.utils.language_utils import detect_language as detect_lang_code, language_name
from src.utils.logger_utils import alog_function_call


@dataclass
class SearchResult:
    metadata: Dict[str, Any]
    text: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def truncate_assistant_content(
        text: str, max_chars: int = MAX_ASSISTANT_RESPONSE_CHARS
) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATION_SUFFIX


@alog_function_call
async def afetch_chat_history(
        conversation_id: str | None = None,
) -> List[Dict[str, Any]]:
    if not conversation_id or not conversation_id.strip():
        return []

    config = get_app_config()
    url = f"{config.CHAT_HISTORY_API_BASE}/api/chats/history-for-ai"
    headers = {"X-Api-Key": "9gzILl3s3DxYGbqGC6xPed2wqa2uWUFRASqkmpoSuv0="}

    try:
        async with httpx.AsyncClient(timeout=CHAT_HISTORY_TIMEOUT) as client:
            resp = await client.get(
                url, params={"conversation_id": conversation_id}, headers=headers
            )
            resp.raise_for_status()
            body = resp.json()
    except httpx.HTTPError:
        return []

    if not body.get("success"):
        return []

    messages = body.get("data", {}).get("messages", [])
    history: List[Dict[str, Any]] = []
    i = 0

    while i < len(messages) - 1:
        user_msg = messages[i]
        ai_msg = messages[i + 1]

        if (
                user_msg.get("sender") == "user"
                and ai_msg.get("sender") == "ai"
                and ai_msg.get("intent") == "core_knowledge"
        ):
            history.append({
                "role": "user",
                "content": user_msg.get("content", {}).get("text", ""),
            })
            assistant_text = truncate_assistant_content(
                ai_msg.get("content", {}).get("text", "")
            )
            history.append({"role": "assistant", "content": assistant_text})
            i += 2
        else:
            i += 1

    return history


def get_detected_language(text: str) -> str:
    return language_name(detect_lang_code(text))


def trim_chat_history_by_tokens(
        chat_history: List[Dict[str, Any]],
        token_budget: int = CHAT_HISTORY_TOKEN_BUDGET,
        chars_per_token: int = CHARS_PER_TOKEN,
) -> List[Dict[str, Any]]:
    char_budget = token_budget * chars_per_token
    selected: List[Dict[str, Any]] = []
    used = 0

    for entry in reversed(chat_history):
        content = entry.get("content", "")
        if used + len(content) > char_budget:
            break
        selected.insert(0, entry)
        used += len(content)

    return selected


def _build_system_prompt(detected_language: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(detected_language=detected_language)


def build_final_prompt(
        user_input: str,
        detected_language: str,
        relevant_chunks: List[Dict[str, Any]],
        node_data_list: List[Dict[str, Any]],
        chat_history: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, str]]]:
    system_prompt = _build_system_prompt(detected_language)

    context_blocks = [
        f"[{chunk.get('metadata', {}).get('file_name', 'unknown')}]:\n{chunk['text']}"
        for chunk in relevant_chunks
    ]
    context_str = "\n\n---\n\n".join(context_blocks)

    node_ref = ""
    if node_data_list:
        lines = [
            f"- **{nd.get('node_name', 'N/A')}**: {nd.get('definition', 'N/A')}  \n"
            f"  Category: {nd.get('category', 'N/A')} | Tags: {nd.get('domain_tags', 'N/A')}"
            for nd in node_data_list
        ]
        node_ref = "\n\n## Knowledge Node Reference\n" + "\n".join(lines)

    trimmed_history = trim_chat_history_by_tokens(chat_history)
    messages: List[Dict[str, str]] = [
        {"role": entry["role"], "content": entry["content"]}
        for entry in trimmed_history
        if entry.get("role") in ("user", "assistant")
    ]

    messages.append({
        "role": "user",
        "content": (
            f"## Retrieved Context\n{context_str}\n\n"
            f"{node_ref}\n\n"
            f"## User Question\n{user_input}"
        ),
    })

    return system_prompt, messages
