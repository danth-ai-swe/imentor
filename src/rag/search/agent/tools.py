"""LangChain @tool wrappers used by the agent dispatcher.

Each tool wraps an existing helper from src.rag.search.pipeline /
searxng_search and writes results back into AgentState via the Command
return type so reducers in StateGraph pick them up.
"""

from typing import Annotated, Any, Dict, List, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.config.app_config import get_app_config
from src.constants.app_constant import (
    COLLECTION_NAME,
    CORE_VECTOR_TOP_K,
    OVERALL_COLLECTION_NAME,
    VECTOR_SEARCH_TOP_K,
)
from src.rag.db_vector import get_qdrant_client
from src.rag.llm.chat_llm import get_openai_chat_client
from src.rag.llm.embedding_llm import get_openai_embedding_client
from src.rag.search.agent.state import AgentState
from src.rag.search.model import ChunkDict
from src.rag.search.pipeline import _aembed_text, _ahyde_generate, _avector_search, extract_sources
from src.rag.search.searxng_search import web_rag_answer
from src.utils.logger_utils import logger


def _chunks_preview(chunks: List[ChunkDict], n: int = 3, chars: int = 300) -> str:
    parts = []
    for c in chunks[:n]:
        text = (c.get("text") or "")[:chars].replace("\n", " ")
        parts.append(text)
    return " ||| ".join(parts) if parts else "(no chunks)"


@tool
async def search_core_collection(
    query: str,
    reasoning: str,
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Search the LOMA281/LOMA291 textbook chunks (core knowledge collection).
    Use for substantive insurance questions: concepts, definitions, formulas,
    contract terms, or any textbook content.

    Args:
        query: English search query (already rewritten as a standalone question).
        reasoning: One short sentence explaining why this tool was chosen (logged).
    """
    logger.info("[tool:search_core] query=%r reasoning=%r", query, reasoning)

    llm = get_openai_chat_client()
    embedder = get_openai_embedding_client()
    qdrant = get_qdrant_client(COLLECTION_NAME)

    try:
        hyde = await _ahyde_generate(llm, query)
        dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, hyde)
        chunks = await _avector_search(
            qdrant, dense_vec, colbert_vec, query, top_k=CORE_VECTOR_TOP_K,
        )
    except Exception as e:
        logger.exception("[tool:search_core] failed")
        return Command(update={
            "messages": [ToolMessage(
                content=f"search_core_collection error: {e}. Try a different tool.",
                tool_call_id=tool_call_id,
            )],
            "tool_call_count": state.get("tool_call_count", 0) + 1,
        })

    found = bool(chunks)
    preview = _chunks_preview(chunks)
    payload = {"found": found, "chunk_count": len(chunks), "preview": preview}

    return Command(update={
        "chunks": (state.get("chunks") or []) + chunks,
        "selected_collection": "core",
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "messages": [ToolMessage(content=str(payload), tool_call_id=tool_call_id)],
    })


@tool
async def search_overall_collection(
    query: str,
    reasoning: str,
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Search course-level metadata (syllabus, module structure, lesson counts,
    course descriptions). Use ONLY for questions like 'how many modules in
    LOMA 281?', 'what does LOMA 291 cover?', 'list lessons in module 3'.

    Args:
        query: English search query.
        reasoning: One short sentence explaining the choice (logged).
    """
    logger.info("[tool:search_overall] query=%r reasoning=%r", query, reasoning)

    embedder = get_openai_embedding_client()
    qdrant = get_qdrant_client(OVERALL_COLLECTION_NAME)

    try:
        dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, query)
        chunks = await _avector_search(
            qdrant, dense_vec, colbert_vec, query, top_k=VECTOR_SEARCH_TOP_K,
        )
    except Exception as e:
        logger.exception("[tool:search_overall] failed")
        return Command(update={
            "messages": [ToolMessage(
                content=f"search_overall_collection error: {e}. Try a different tool.",
                tool_call_id=tool_call_id,
            )],
            "tool_call_count": state.get("tool_call_count", 0) + 1,
        })

    found = bool(chunks)
    preview = _chunks_preview(chunks)
    payload = {"found": found, "chunk_count": len(chunks), "preview": preview}

    return Command(update={
        "chunks": (state.get("chunks") or []) + chunks,
        "selected_collection": "overall",
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "messages": [ToolMessage(content=str(payload), tool_call_id=tool_call_id)],
    })


@tool
async def search_web(
    query: str,
    reasoning: str,
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Search the public web (SearXNG). Use only when:
    (1) you already searched the textbook collection and `found` was false, OR
    (2) the question is clearly time-sensitive (recent events, dates, news)
        and outside the LOMA textbook scope.

    Args:
        query: English search query.
        reasoning: One short sentence explaining the choice (logged).
    """
    logger.info("[tool:search_web] query=%r reasoning=%r", query, reasoning)

    llm = get_openai_chat_client()
    embedder = get_openai_embedding_client()
    detected_language = state.get("detected_language") or "English"

    try:
        rs = await web_rag_answer(llm, embedder, query, detected_language)
    except Exception as e:
        logger.exception("[tool:search_web] failed")
        return Command(update={
            "messages": [ToolMessage(
                content=f"search_web error: {e}",
                tool_call_id=tool_call_id,
            )],
            "tool_call_count": state.get("tool_call_count", 0) + 1,
        })

    answer = rs.get("answer") or ""
    sources = rs.get("sources") or []
    # web_rag_answer returns a localized "no result" string with empty sources
    # when SearXNG / crawl / chunking yielded nothing — treat that as not-found
    # so the agent can decide its next step instead of seeing a stale web_answer.
    found = bool(answer) and bool(sources)
    payload = {"found": found, "answer_preview": answer[:200]}

    update = {
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "messages": [ToolMessage(content=str(payload), tool_call_id=tool_call_id)],
    }
    if found:
        update["web_answer"] = answer
        update["sources"] = sources
        update["selected_collection"] = "web"
        update["web_search_used"] = True

    return Command(update=update)


@tool
async def ask_clarification(
    reason: Literal["off_topic", "vague"],
    message: str,
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Terminate the loop with a message to the user. Use for off-topic
    questions or for questions too vague to search effectively.

    Args:
        reason: "off_topic" (not insurance-related) or "vague" (unclear).
        message: Text to show the user (must be in their language).
    """
    logger.info("[tool:ask_clarification] reason=%s", reason)

    return Command(update={
        "clarification": {"type": reason, "response": message},
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "messages": [ToolMessage(
            content=f"clarification_sent: {reason}",
            tool_call_id=tool_call_id,
        )],
    })


ALL_TOOLS = [
    search_core_collection,
    search_overall_collection,
    search_web,
    ask_clarification,
]
