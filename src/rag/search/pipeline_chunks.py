"""Chunks-only dispatcher for the Java backend.

Mirrors `async_pipeline_dispatch` from pipeline.py but skips the final
`_agenerate_answer` call. Returns either:
  - mode="static": a pre-built reply (off-topic, quiz, clarity, web fallback,
                   no-result, input-too-long); Java streams it as-is.
  - mode="chunks": enriched chunks + standalone_query + detected_language;
                   Java builds the prompt and streams from OpenAI.

Reuses the helpers in pipeline.py (validate, intent routing, clarity, HyDE,
embed, vector search, rerank, neighbor enrich) — does NOT duplicate them.
"""
from typing import Any, Dict, List, Optional

from src.apis.app_model import ChatSourceModel
from src.config.app_config import AppConfig, get_app_config
from src.constants.app_constant import (
    COLLECTION_NAME,
    OVERALL_COLLECTION_NAME,
    INTENT_CORE_KNOWLEDGE,
    INTENT_OFF_TOPIC,
    INTENT_QUIZ,
    INTENT_OVERALL_COURSE_KNOWLEDGE,
    OFF_TOPIC_RESPONSE_MAP,
    NO_RESULT_RESPONSE_MAP,
    CORE_VECTOR_TOP_K,
    CORE_RERANK_TOP_K,
    VECTOR_SEARCH_TOP_K,
)
from src.rag.db_vector import get_qdrant_client
from src.rag.llm.chat_llm import AzureChatClient, get_openai_chat_client
from src.rag.llm.embedding_llm import AzureEmbeddingClient, get_openai_embedding_client
from src.rag.search.pipeline import (
    _acheck_input_clarity,
    _aembed_text,
    _afetch_neighbor_chunks,
    _ahyde_generate,
    _aroute_intent,
    _avalidate_and_prepare,
    _avector_search,
    _merge_chunks,
    extract_sources,
)
from src.rag.search.reranker import arerank_chunks
from src.rag.search.searxng_search import web_rag_answer
from src.utils.app_utils import is_quiz_intent
from src.utils.logger_utils import logger, StepTimer


def _serialize_sources(sources: List[Any]) -> List[Dict[str, Any]]:
    return [
        s.model_dump(by_alias=False) if isinstance(s, ChatSourceModel) else dict(s)
        for s in (sources or [])
    ]


def _serialize_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in chunks or []:
        out.append({
            "text": c.get("text", ""),
            "metadata": dict(c.get("metadata", {})),
        })
    return out


def _static(
    *, intent: str, response: Optional[str], detected_language: Optional[str],
    sources: Optional[List[Any]] = None, web_search_used: bool = False,
    answer_satisfied: bool = True, standalone_query: str = "",
) -> Dict[str, Any]:
    return {
        "mode": "static",
        "intent": intent,
        "detectedLanguage": detected_language,
        "standaloneQuery": standalone_query,
        "answerSatisfied": answer_satisfied,
        "webSearchUsed": web_search_used,
        "sources": _serialize_sources(sources or []),
        "chunks": [],
        "response": response or "",
    }


def _chunks_payload(
    *, intent: str, detected_language: str, standalone_query: str,
    sources: List[Any], chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "mode": "chunks",
        "intent": intent,
        "detectedLanguage": detected_language,
        "standaloneQuery": standalone_query,
        "answerSatisfied": True,
        "webSearchUsed": False,
        "sources": _serialize_sources(sources),
        "chunks": _serialize_chunks(chunks),
        "response": None,
    }


async def _arun_core_search_chunks(
    llm: AzureChatClient,
    embedder: AzureEmbeddingClient,
    standalone_query: str,
    detected_language: str,
) -> Dict[str, Any]:
    timer = StepTimer(f"core_search_chunks:{COLLECTION_NAME}")
    config: AppConfig = get_app_config()

    try:
        qdrant = get_qdrant_client(COLLECTION_NAME)

        async with timer.astep("clarity_check"):
            clarity = await _acheck_input_clarity(llm, standalone_query, detected_language)
        if not clarity.get("clear", True):
            response_parts = clarity.get("response", [])
            response_str = (
                response_parts if isinstance(response_parts, str)
                else "\n".join(r.strip() for r in response_parts if r) if isinstance(response_parts, list)
                else str(response_parts)
            )
            return _static(
                intent=INTENT_CORE_KNOWLEDGE, response=response_str,
                detected_language=detected_language, answer_satisfied=False,
                standalone_query=standalone_query,
            )

        async with timer.astep("hyde"):
            hyde = await _ahyde_generate(llm, standalone_query)
        async with timer.astep("hyde_embedding"):
            dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, hyde)
        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, dense_vec, colbert_vec, standalone_query, top_k=CORE_VECTOR_TOP_K,
            )

        if not sorted_chunks:
            async with timer.astep("web_search_fallback"):
                rs = await web_rag_answer(llm, embedder, standalone_query, detected_language)
            return _static(
                intent=INTENT_CORE_KNOWLEDGE, response=rs["answer"],
                detected_language=detected_language, sources=rs["sources"],
                web_search_used=True, standalone_query=standalone_query,
            )

        async with timer.astep("rerank"):
            reranked = await arerank_chunks(standalone_query, sorted_chunks, CORE_RERANK_TOP_K)
        async with timer.astep("enrich_chunks"):
            neighbor_chunks = await _afetch_neighbor_chunks(qdrant, reranked)
            enriched_chunks = _merge_chunks(reranked, neighbor_chunks)

        sources = extract_sources(reranked, config.APP_DOMAIN)
        return _chunks_payload(
            intent=INTENT_CORE_KNOWLEDGE, detected_language=detected_language,
            standalone_query=standalone_query, sources=sources, chunks=enriched_chunks,
        )
    finally:
        timer.summary()


async def _arun_overall_search_chunks(
    llm: AzureChatClient,
    embedder: AzureEmbeddingClient,
    standalone_query: str,
    detected_language: str,
) -> Dict[str, Any]:
    timer = StepTimer(f"overall_search_chunks:{OVERALL_COLLECTION_NAME}")

    try:
        qdrant = get_qdrant_client(OVERALL_COLLECTION_NAME)

        async with timer.astep("clarity_check"):
            clarity = await _acheck_input_clarity(llm, standalone_query, detected_language)
        if not clarity.get("clear", True):
            response_parts = clarity.get("response", [])
            response_str = (
                response_parts if isinstance(response_parts, str)
                else "\n".join(r.strip() for r in response_parts if r) if isinstance(response_parts, list)
                else str(response_parts)
            )
            return _static(
                intent=INTENT_OVERALL_COURSE_KNOWLEDGE, response=response_str,
                detected_language=detected_language, answer_satisfied=False,
                standalone_query=standalone_query,
            )

        async with timer.astep("query_embedding"):
            dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, standalone_query)
        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, dense_vec, colbert_vec, standalone_query, top_k=VECTOR_SEARCH_TOP_K,
            )

        if not sorted_chunks:
            no_result = NO_RESULT_RESPONSE_MAP.get(detected_language) or ""
            return _static(
                intent=INTENT_OVERALL_COURSE_KNOWLEDGE, response=no_result,
                detected_language=detected_language, answer_satisfied=False,
                standalone_query=standalone_query,
            )

        return _chunks_payload(
            intent=INTENT_OVERALL_COURSE_KNOWLEDGE, detected_language=detected_language,
            standalone_query=standalone_query, sources=[], chunks=sorted_chunks,
        )
    finally:
        timer.summary()


async def async_pipeline_dispatch_chunks(
    user_input: str,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Chunks-only equivalent of async_pipeline_dispatch. See module docstring."""
    timer = StepTimer("dispatch_chunks")
    try:
        llm = get_openai_chat_client()
        embedder = get_openai_embedding_client()

        async with timer.astep("validate_and_prepare"):
            detected_language, standalone_query, error_response = (
                await _avalidate_and_prepare(user_input, conversation_id, llm)
            )
            if error_response:
                return _static(
                    intent=INTENT_OFF_TOPIC, response=error_response,
                    detected_language="English", standalone_query=user_input,
                    answer_satisfied=False,
                )

        async with timer.astep("quiz_keyword_check"):
            if is_quiz_intent(user_input) or is_quiz_intent(standalone_query):
                return _static(
                    intent=INTENT_QUIZ,
                    response="Quiz feature is not enabled in this demo.",
                    detected_language=detected_language, answer_satisfied=False,
                    standalone_query=standalone_query,
                )

        async with timer.astep("intent_routing"):
            intent = await _aroute_intent(embedder, standalone_query)

        if intent == INTENT_OFF_TOPIC:
            return _static(
                intent=INTENT_OFF_TOPIC,
                response=OFF_TOPIC_RESPONSE_MAP.get(detected_language) or "",
                detected_language=detected_language, answer_satisfied=False,
                standalone_query=standalone_query,
            )

        if intent == INTENT_OVERALL_COURSE_KNOWLEDGE:
            return await _arun_overall_search_chunks(llm, embedder, standalone_query, detected_language)

        return await _arun_core_search_chunks(llm, embedder, standalone_query, detected_language)
    finally:
        timer.summary()
