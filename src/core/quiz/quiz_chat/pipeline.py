"""Quiz-chat dispatcher and quiz-aware question pipeline.

Mirrors the structure of `src/rag/search/pipeline_chunks.py` so the codebase
has one consistent pattern for "wrap the existing dispatch with a custom
final step." Imports private helpers from `src/rag/search/pipeline.py`
without modifying them.
"""
import asyncio
from typing import Any, Dict, List, Optional

from src.apis.app_model import ChatSourceModel
from src.config.app_config import AppConfig, get_app_config
from src.constants.app_constant import (
    COLLECTION_NAME,
    CORE_RERANK_TOP_K,
    CORE_VECTOR_TOP_K,
    INTENT_OFF_TOPIC,
    INTENT_OVERALL_COURSE_KNOWLEDGE,
    NO_RESULT_RESPONSE_MAP,
    OFF_TOPIC_RESPONSE_MAP,
    OVERALL_COLLECTION_NAME,
    VECTOR_SEARCH_TOP_K,
)
from src.core.quiz.quiz_chat.intent_classifier import (
    IntentResult,
    _format_options_block,
    aclassify_intent,
)
from src.core.quiz.quiz_chat.model import (
    CurrentQuestion,
    QuizChatRequest,
    QuizChatResponse,
)
from src.core.quiz.quiz_chat.prompts import QUESTION_GUARD_INSTRUCTION
from src.rag.db_vector import get_qdrant_client
from src.rag.llm.chat_llm import AzureChatClient, get_openai_chat_client
from src.rag.llm.embedding_llm import AzureEmbeddingClient, get_openai_embedding_client
from src.rag.search.entrypoint import build_final_prompt
from src.rag.search.pipeline import (
    _acheck_input_clarity,
    _aembed_text,
    _afetch_neighbor_chunks,
    _ahyde_generate,
    _aroute_intent,
    _avalidate_and_prepare,
    _avector_search,
    _make_clarity_result,
    _merge_chunks,
    extract_sources,
)
from src.rag.search.reranker import arerank_chunks
from src.rag.search.searxng_search import web_rag_answer
from src.utils.logger_utils import StepTimer, logger


_AMBIGUOUS_ANSWER_MSG = (
    "Could not match your answer to any option. Please try again."
)


def _clarity_content(clarity: dict, detected_language: str) -> str:
    """Reuse the existing clarity-formatter so rephrase/non-rephrase types
    render the same way as the legacy /chat/ask path."""
    return _make_clarity_result(clarity, detected_language).get("response") or ""


async def _aquiz_generate_answer(
    llm: AzureChatClient,
    standalone_query: str,
    detected_language: str,
    enriched_chunks: List[Dict[str, Any]],
    current_question: CurrentQuestion,
) -> str:
    base_prompt = build_final_prompt(
        user_input=standalone_query,
        detected_language=detected_language,
        relevant_chunks=enriched_chunks,
    )
    guard = QUESTION_GUARD_INSTRUCTION.format(
        node_name=current_question.node_name,
        category=current_question.category,
        question_text=current_question.question,
        options_block=_format_options_block(current_question),
    )
    augmented = guard + "\n\n" + base_prompt
    try:
        return (await llm.ainvoke_creative(augmented)).strip()
    except Exception:
        logger.exception("Failed to generate quiz-aware answer from LLM")
        return ""


def _question_response(
    *, content: str, sources: Optional[List[ChatSourceModel]] = None,
) -> QuizChatResponse:
    return QuizChatResponse(
        intent="question",
        content=content,
        sources=sources or [],
    )


async def _arun_core_quiz_question(
    llm: AzureChatClient,
    embedder: AzureEmbeddingClient,
    standalone_query: str,
    detected_language: str,
    current_question: CurrentQuestion,
) -> QuizChatResponse:
    timer = StepTimer(f"core_quiz_question:{COLLECTION_NAME}")
    config: AppConfig = get_app_config()

    try:
        qdrant = get_qdrant_client(COLLECTION_NAME)

        async with timer.astep("clarity_and_hyde_parallel"):
            clarity_task = _acheck_input_clarity(llm, standalone_query, detected_language)
            hyde_task = _ahyde_generate(llm, standalone_query)
            clarity, hyde = await asyncio.gather(clarity_task, hyde_task)

        if not clarity.get("clear", True):
            return _question_response(
                content=_clarity_content(clarity, detected_language), sources=[],
            )

        async with timer.astep("hyde_embedding"):
            dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, hyde)

        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, dense_vec, colbert_vec, standalone_query, top_k=CORE_VECTOR_TOP_K,
            )

        if not sorted_chunks:
            async with timer.astep("web_search_fallback"):
                rs = await web_rag_answer(llm, embedder, standalone_query, detected_language)
            return _question_response(content=rs["answer"], sources=rs["sources"])

        async with timer.astep("rerank"):
            reranked = await arerank_chunks(standalone_query, sorted_chunks, CORE_RERANK_TOP_K)

        async with timer.astep("enrich_chunks"):
            neighbor_chunks = await _afetch_neighbor_chunks(qdrant, reranked)
            enriched_chunks = _merge_chunks(reranked, neighbor_chunks)

        async with timer.astep("generate_answer"):
            answer = await _aquiz_generate_answer(
                llm, standalone_query, detected_language, enriched_chunks, current_question,
            )

        sources = extract_sources(reranked, config.APP_DOMAIN)
        return _question_response(content=answer, sources=sources)
    finally:
        timer.summary()


async def _arun_overall_quiz_question(
    llm: AzureChatClient,
    embedder: AzureEmbeddingClient,
    standalone_query: str,
    detected_language: str,
    current_question: CurrentQuestion,
) -> QuizChatResponse:
    timer = StepTimer(f"overall_quiz_question:{OVERALL_COLLECTION_NAME}")

    try:
        qdrant = get_qdrant_client(OVERALL_COLLECTION_NAME)

        async with timer.astep("clarity_check"):
            clarity = await _acheck_input_clarity(llm, standalone_query, detected_language)
        if not clarity.get("clear", True):
            return _question_response(
                content=_clarity_content(clarity, detected_language), sources=[],
            )

        async with timer.astep("query_embedding"):
            dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, standalone_query)

        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, dense_vec, colbert_vec, standalone_query, top_k=VECTOR_SEARCH_TOP_K,
            )

        if not sorted_chunks:
            no_result = NO_RESULT_RESPONSE_MAP.get(detected_language) or ""
            return _question_response(content=no_result, sources=[])

        async with timer.astep("generate_answer"):
            answer = await _aquiz_generate_answer(
                llm, standalone_query, detected_language, sorted_chunks, current_question,
            )

        return _question_response(content=answer, sources=[])
    finally:
        timer.summary()


async def _arun_question_intent(
    llm: AzureChatClient,
    embedder: AzureEmbeddingClient,
    message: str,
    current_question: CurrentQuestion,
) -> QuizChatResponse:
    detected_language, standalone_query, error_response = (
        await _avalidate_and_prepare(message, None, llm)
    )
    if error_response:
        return _question_response(content=error_response, sources=[])

    intent = await _aroute_intent(embedder, standalone_query)

    if intent == INTENT_OFF_TOPIC:
        return _question_response(
            content=OFF_TOPIC_RESPONSE_MAP.get(detected_language) or "",
            sources=[],
        )

    if intent == INTENT_OVERALL_COURSE_KNOWLEDGE:
        return await _arun_overall_quiz_question(
            llm, embedder, standalone_query, detected_language, current_question,
        )

    return await _arun_core_quiz_question(
        llm, embedder, standalone_query, detected_language, current_question,
    )


async def async_quiz_chat_dispatch(payload: QuizChatRequest) -> QuizChatResponse:
    """Top-level entry: classify intent, then either return a static response
    (hint/finish/answer) or run the quiz-aware question pipeline."""
    timer = StepTimer("quiz_chat_dispatch")
    try:
        llm = get_openai_chat_client()
        embedder = get_openai_embedding_client()

        async with timer.astep("classify_intent"):
            decision: IntentResult = await aclassify_intent(
                payload.message, payload.current_question, llm,
            )

        if decision.intent == "hint":
            return QuizChatResponse(intent="hint")

        if decision.intent == "finish":
            return QuizChatResponse(intent="finish")

        if decision.intent == "answer":
            if decision.answer_index is None:
                return QuizChatResponse(
                    intent="answer",
                    answer_index=None,
                    message=_AMBIGUOUS_ANSWER_MSG,
                )
            return QuizChatResponse(
                intent="answer",
                answer_index=decision.answer_index,
            )

        async with timer.astep("question_pipeline"):
            return await _arun_question_intent(
                llm, embedder, payload.message, payload.current_question,
            )
    finally:
        timer.summary()
