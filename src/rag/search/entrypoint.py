from src.constants.app_constant import (
    MAX_RECENT_HISTORY_ENTRIES,
)
from src.external.fetch_history import fetch_raw_chat_history
from src.rag.llm.chat_llm import AzureChatClient
from src.rag.search.prompt import SYSTEM_PROMPT_TEMPLATE, SUMMARIZE_PROMPT_TEMPLATE
from src.utils.app_utils import parse_json_response
from src.utils.logger_utils import alog_function_call, logger


def _filter_core_knowledge_pairs(
        messages: list[dict],
) -> list[dict]:
    history: list[dict] = []
    i = 0

    while i < len(messages) - 1:
        user_msg = messages[i]
        ai_msg = messages[i + 1]

        if (
                user_msg.get("role") == "user"
                and ai_msg.get("role") == "assistant"
                and ai_msg.get("intent") == "core_knowledge"
        ):
            history.append({
                "role": "user",
                "content": user_msg.get("content", ""),
            })
            history.append({
                "role": "assistant",
                "content": ai_msg.get("content", ""),
            })
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


async def _summarize_history(llm: AzureChatClient, history_string: str, query: str) -> str:
    """Recomp-style query-focused summary. Returns plain string (possibly empty)."""
    prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
        conversation_history=history_string,
        query=query,
    )
    raw = (await llm.ainvoke(prompt)).strip()
    try:
        result = parse_json_response(raw).get("summary", "")
        return result if isinstance(result, str) else ""
    except Exception:
        logger.warning("Failed to parse summary response, returning empty. Raw: %s", raw)
        return ""


@alog_function_call
async def afetch_chat_history(
    llm: AzureChatClient,
    conversation_id: str | None = None,
    query: str = "",
) -> str:
    if not conversation_id or not conversation_id.strip():
        return ""
    if not query or not query.strip():
        return ""

    messages = await fetch_raw_chat_history(conversation_id)
    if not messages:
        return ""

    filtered = _filter_core_knowledge_pairs(messages)
    if not filtered:
        return ""

    history_string = _format_history(filtered)
    summary = await _summarize_history(llm, history_string, query)
    return summary


def build_final_prompt(
        user_input: str,
        detected_language: str,
        relevant_chunks: list,
) -> str:
    context_blocks = []
    for chunk in relevant_chunks:
        metadata = chunk.get("metadata", {}) or {}

        file_name = metadata.get("file_name", "unknown")
        course = metadata.get("course", "")
        module_num = metadata.get("module_number", "")
        lesson_num = metadata.get("lesson_number", "")
        page_number = metadata.get("page_number", "")

        header_parts = [f"Source: {file_name}"]
        if course:
            header_parts.append(f"Course: {course}")
        if module_num != "":
            header_parts.append(f"Module: {module_num}")
        if lesson_num != "":
            header_parts.append(f"Lesson: {lesson_num}")
        if page_number != "":
            header_parts.append(f"Page: {page_number}")

        header = " | ".join(header_parts)
        text = chunk.get("text", "")

        context_blocks.append(f"[{header}]\n{text}")

    context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context)"

    return SYSTEM_PROMPT_TEMPLATE.format(
        detected_language=detected_language,
        context_str=context_str,
        node_ref="",
        user_input=user_input,
    )
