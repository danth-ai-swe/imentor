from typing import Any, Dict, List, Tuple

from src.constants.app_constant import (
    MAX_RECENT_HISTORY_ENTRIES,
)
from src.external.fetch_history import fetch_raw_chat_history
from src.rag.llm.chat_llm import get_openai_chat_client
from src.rag.search.pipeline import ChunkDict
from src.rag.search.prompt import SYSTEM_PROMPT_TEMPLATE, SUMMARIZE_PROMPT_TEMPLATE
from src.utils.logger_utils import alog_function_call


def _filter_core_knowledge_pairs(
        messages: list[dict],
) -> list[dict]:
    history: list[dict] = []
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
            history.append({"role": "assistant", "content": ai_msg.get("content", {}).get("text", "")})
            i += 2
        else:
            i += 1

    return history


def _format_history(history: list[dict]) -> str:
    lines: list[str] = []
    for entry in history[-MAX_RECENT_HISTORY_ENTRIES:]:
        role = entry.get("role", "user").capitalize()
        if entry.get("parts"):
            text = " ".join(p["text"] for p in entry["parts"])
        else:
            text = entry.get("content", "")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


async def _summarize_history(history_string: str) -> str:
    """Gọi Azure OpenAI để tóm tắt history string, trả về summary string."""
    prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
        conversation_history=history_string
    )

    raw = await get_openai_chat_client().ainvoke(prompt)

    start = raw.find("<result>")
    end = raw.find("</result>")
    if start != -1 and end != -1:
        return raw[start + len("<result>"):end].strip()

    return raw.strip()


@alog_function_call
async def afetch_chat_history(
        conversation_id: str | None = None,
) -> str:
    if not conversation_id or not conversation_id.strip():
        return ""

    # Bước 1: Call API
    messages = await fetch_raw_chat_history(conversation_id)
    if not messages:
        return ""

    # Bước 2: Lọc intent
    filtered = _filter_core_knowledge_pairs(messages)
    if not filtered:
        return ""

    # Bước 3: Lấy 6 phần tử cuối → build string
    history_string = _format_history(filtered)

    # Bước 4: Tóm tắt bằng Claude → summary string
    summary = await _summarize_history(history_string)

    return summary


def build_final_prompt(
        user_input: str,
        detected_language: str,
        relevant_chunks:  List[ChunkDict],
        node_data_list: List[Dict[str, Any]],
        chat_history_summary: str,  # <-- đổi sang string
) -> Tuple[str, List[Dict[str, str]]]:
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        detected_language=detected_language
    )

    # Context từ retrieval
    context_blocks = [
        f"[{chunk.get('metadata', {}).get('file_name', 'unknown')}]:\n{chunk['text']}"
        for chunk in relevant_chunks
    ]
    context_str = "\n\n---\n\n".join(context_blocks)

    # Node reference
    node_ref = ""
    if node_data_list:
        lines = [
            f"- **{nd.get('Node Name', 'N/A')}**: {nd.get('Definition', 'N/A')}  \n"
            f"  Category: {nd.get('Category', 'N/A')} | Tags: {nd.get('Domain Tags', 'N/A')}\n"
            f"  Overall Summary: {nd.get('Summary', 'N/A')}"
            for nd in node_data_list
        ]
        node_ref = "\n\n## Knowledge Node Reference\n" + "\n".join(lines)

    # 👉 Messages chỉ còn system + 1 user message (có history summary)
    messages: List[Dict[str, str]] = []

    # Inject history dạng summary
    history_block = ""
    if chat_history_summary:
        history_block = f"## Conversation Summary\n{chat_history_summary}\n\n"

    messages.append({
        "role": "user",
        "content": (
            f"{history_block}"
            f"## Retrieved Context\n{context_str}\n\n"
            f"{node_ref}\n\n"
            f"## User Question\n{user_input}"
        ),
    })

    return system_prompt, messages
