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


async def _summarize_history(llm: AzureChatClient, history_string: str) -> str:
    """Gọi Azure OpenAI để tóm tắt history string, trả về summary string dạng JSON."""
    prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
        conversation_history=history_string
    )

    raw = (await llm.ainvoke(prompt)).strip()

    try:
        return parse_json_response(raw).get("summary", "")
    except Exception:
        logger.warning("Failed to parse summary response as JSON, returning raw text. Response: %s", raw)
    return raw


@alog_function_call
async def afetch_chat_history(llm: AzureChatClient,
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
    summary = await _summarize_history(llm, history_string)

    return summary


def _build_chunk_filter_payload(
        index: int,
        chunk,
        max_text_chars: int = 400,
) -> dict:
    """
    Build một dict mô tả chunk gồm metadata + text + node summary,
    dùng để đưa vào prompt filter.
    """
    metadata = chunk.get("metadata", {}) or {}

    # ── Source header ──────────────────────────────────────────────────
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

    # ── Node reference ─────────────────────────────────────────────────
    node_id = metadata.get("node_id")
    node_name = metadata.get("node_name", "")
    category = metadata.get("category", "")
    summary = metadata.get("summary", "")

    node_label = None
    if node_id is not None and summary:
        label_parts = [f"Node {node_id}"]
        if node_name:
            label_parts.append(node_name)
        if category:
            label_parts.append(f"({category})")
        node_label = (
                " - ".join(label_parts[:2])
                + (f" {label_parts[2]}" if len(label_parts) > 2 else "")
        )

    payload: dict = {
        "index": index,
        "header": " | ".join(header_parts),
        "text": chunk.get("text", "")[:max_text_chars],
    }
    if node_label:
        payload["node"] = node_label
        payload["node_summary"] = summary[:200]

    return payload


def build_final_prompt(
        user_input: str,
        detected_language: str,
        relevant_chunks: list,
) -> str:
    # ===== 1. Build context blocks từ các chunk =====
    context_blocks = []
    for chunk in relevant_chunks:
        payload = chunk.get("payload", {}) or {}

        file_name = payload.get("file_name", "unknown")
        course = payload.get("course", "")
        module_num = payload.get("module_number", "")
        lesson_num = payload.get("lesson_number", "")
        page_number = payload.get("page_number", "")

        # Header mô tả nguồn gốc của chunk
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

    # ===== 2. Build Knowledge Node Reference (dedupe theo node_id) =====
    seen_node_ids = set()
    node_lines = []
    for chunk in relevant_chunks:
        payload = chunk.get("payload", {}) or {}
        node_id = payload.get("node_id")
        summary = payload.get("summary")

        if node_id is None or not summary:
            continue
        if node_id in seen_node_ids:
            continue
        seen_node_ids.add(node_id)

        node_name = payload.get("node_name", "")
        category = payload.get("category", "")

        label_parts = [f"Node {node_id}"]
        if node_name:
            label_parts.append(node_name)
        if category:
            label_parts.append(f"({category})")
        label = " - ".join(label_parts[:2]) + (f" {label_parts[2]}" if len(label_parts) > 2 else "")

        node_lines.append(f"- {label}\n  Summary: {summary}")

    node_ref = ""
    if node_lines:
        node_ref = "## Knowledge Node Reference\n" + "\n".join(node_lines)

    # ===== 3. Ghép prompt cuối =====
    final_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        detected_language=detected_language,
        context_str=context_str,
        node_ref=node_ref,
        user_input=user_input,
    )

    return final_prompt
