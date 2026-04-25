"""LangChain @tool wrappers used by the agent dispatcher.

Each tool wraps an existing helper from src.rag.search.pipeline /
searxng_search and writes results back into AgentState via the Command
return type so reducers in StateGraph pick them up.
"""

from typing import Annotated, Any, Dict, List, Literal

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

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
from src.config.app_config import get_app_config
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
            "messages": [{
                "role": "tool",
                "content": f"search_core_collection error: {e}. Try a different tool.",
            }],
        })

    found = bool(chunks)
    preview = _chunks_preview(chunks)
    payload = {"found": found, "chunk_count": len(chunks), "preview": preview}

    return Command(update={
        "chunks": (state.get("chunks") or []) + chunks,
        "selected_collection": "core",
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "messages": [{"role": "tool", "content": str(payload)}],
    })
