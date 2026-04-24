import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np

from src.apis.app_model import ChatSourceModel
from src.config.app_config import AppConfig, get_app_config
from src.constants.app_constant import (
    COLLECTION_NAME,
    MAX_INPUT_CHARS,
    INTENT_CORE_KNOWLEDGE, INTENT_OFF_TOPIC, INTENT_QUIZ, NEIGHBOR_PREV_INDEX,
    VECTOR_SEARCH_TOP_K, CORE_VECTOR_TOP_K, CORE_RERANK_TOP_K,
    NEIGHBOR_NEXT_INDEX, UNSUPPORTED_LANGUAGE_MSG, INPUT_TOO_LONG_RESPONSE, OFF_TOPIC_RESPONSE_MAP,
    INTENT_OVERALL_COURSE_KNOWLEDGE, OVERALL_COLLECTION_NAME, NO_RESULT_RESPONSE_MAP,
)
from src.rag.db_vector import get_qdrant_client, QdrantManager
from src.rag.llm.chat_llm import AzureChatClient, get_openai_chat_client
from src.rag.llm.embedding_llm import AzureEmbeddingClient, get_openai_embedding_client
from src.rag.reflector import Reflection
from src.rag.search import prep_cache
from src.rag.search.entrypoint import build_final_prompt, afetch_chat_history
from src.rag.search.model import PipelineResult, ChunkDict
from src.rag.search.prompt import DETECT_LANGUAGE_PROMPT, CLARITY_CHECK_PROMPT, HYDE_PROMPT
from src.rag.search.reranker import arerank_chunks
from src.rag.search.searxng_search import web_rag_answer
from src.rag.semantic_router.intent_router_registry import get_intent_router
from src.utils.app_utils import is_quiz_intent, parse_json_response
from src.utils.language_utils import atranslate_to_english
from src.utils.logger_utils import logger, StepTimer


def _make_off_topic_result(
        message: Optional[str] = None,
        detected_language: Optional[str] = None,
) -> PipelineResult:
    return PipelineResult(
        intent=INTENT_OFF_TOPIC,
        response=message,
        detected_language=detected_language,
        answer_satisfied=False,
        web_search_used=False,
        sources=[],
    )


def _make_quiz_result() -> PipelineResult:
    return PipelineResult(
        intent=INTENT_QUIZ,
        response=None,
        detected_language=None,
        answer_satisfied=False,
        web_search_used=False,
        sources=[],
    )


def _make_web_search_result(
        response: str,
        sources: List[Any],
        detected_language: Optional[str],
) -> PipelineResult:
    return PipelineResult(
        intent=INTENT_CORE_KNOWLEDGE,
        response=response,
        detected_language=detected_language,
        answer_satisfied=True,
        web_search_used=True,
        sources=sources,
    )


def _make_rag_result(
        response: str,
        sources: List[Any],
        detected_language: Optional[str],
        answer_satisfied: bool,
) -> PipelineResult:
    return PipelineResult(
        intent=INTENT_CORE_KNOWLEDGE,
        response=response,
        detected_language=detected_language,
        answer_satisfied=answer_satisfied,
        web_search_used=False,
        sources=sources,
    )


def _make_clarity_result(
        clarity: dict,
        detected_language: Optional[str],
) -> PipelineResult:
    response_parts = clarity.get("response", [])

    if clarity.get("type") == "rephrase":
        response_str = (
            response_parts
            if isinstance(response_parts, str)
            else str(response_parts)
        )
    else:
        if isinstance(response_parts, list):
            response_str = "\n".join(r.strip() for r in response_parts if r)
        else:
            response_str = str(response_parts)

    return PipelineResult(
        intent=INTENT_CORE_KNOWLEDGE,
        response=response_str,
        detected_language=detected_language,
        answer_satisfied=False,
        web_search_used=False,
        sources=[],
    )


async def _acheck_input_clarity(
        llm: AzureChatClient,
        standalone_query: str,
        response_language: str,
) -> dict:
    prompt = CLARITY_CHECK_PROMPT.format(
        standalone_query=standalone_query,
        response_language=response_language,
    )
    try:
        raw = (await llm.ainvoke(prompt)).strip()
        return parse_json_response(raw)
    except Exception:
        logger.exception("Clarity check failed, assuming clear")
        return {"clear": True}


async def _avalidate_and_prepare(
        user_input: str,
        conversation_id: Optional[str],
        llm: AzureChatClient,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if len(user_input) > MAX_INPUT_CHARS:
        return None, None, INPUT_TOO_LONG_RESPONSE.format(max_chars=MAX_INPUT_CHARS)

    has_history = bool(conversation_id and conversation_id.strip())
    if not has_history:
        cached = prep_cache.get(user_input)
        if cached is not None:
            return cached

    # LLM language detection runs in parallel with the reflection step
    # (history fetch + translate/rewrite) — they share no state.
    language_task = asyncio.create_task(_adetect_language_llm(llm, user_input))

    if has_history:
        chat_history = await afetch_chat_history(llm, conversation_id)
        standalone_query = await Reflection(llm).areflect(chat_history, user_input)
    else:
        standalone_query = await Reflection(llm).areflect("", user_input)

    detected_language = await language_task
    if not detected_language:
        return None, None, UNSUPPORTED_LANGUAGE_MSG

    result = (detected_language, standalone_query, None)
    if not has_history:
        prep_cache.put(user_input, result)
    return result


async def _aensure_english_query(text: str) -> str:
    """Guard: HyDE must see English input. If the reflected query still has
    non-ASCII characters (VN/JP/etc.), translate it before prompting."""
    if not text or all(ord(c) < 128 for c in text):
        return text
    return await atranslate_to_english(text)


async def _ahyde_generate(
        llm: AzureChatClient,
        standalone_query: str,
) -> str:
    english_query = await _aensure_english_query(standalone_query)
    prompt = HYDE_PROMPT.format(standalone_query=english_query)
    try:
        raw = (await llm.ainvoke(prompt)).strip()
        parsed = parse_json_response(raw)
        hyde = parsed.get("hyde", "")
        if isinstance(hyde, str) and hyde.strip():
            return hyde
        return english_query
    except Exception:
        logger.exception("HyDE generation failed, using standalone_query")
        return english_query


async def _adetect_language_llm(
        llm: AzureChatClient,
        text: str,
):
    prompt = DETECT_LANGUAGE_PROMPT.format(text=text[:300])
    try:
        raw = (await llm.ainvoke(prompt)).strip()
        parsed = parse_json_response(raw)
        language: str = parsed.get("language", "")
        return language
    except Exception:
        logger.exception("LLM language detection failed, defaulting to English")
        return ""


def extract_sources(
        chunks: List[ChunkDict],
        app_domain: str,
) -> List[ChatSourceModel]:
    seen: Dict[str, ChatSourceModel] = {}
    for chunk in chunks:
        source_file: str = chunk.get("metadata", {}).get("file_name", "")
        source = None
        if source_file:
            source = ChatSourceModel(
                name=source_file,
                url=f"{app_domain}/api/v1/file/{source_file}.pdf",
                page_number=chunk.get("metadata", {}).get("page_number", 1),
                total_pages=chunk.get("metadata", {}).get("total_pages", 1),
            )
        if source and source.name not in seen:
            seen[source.name] = source
    return list(seen.values())


async def _aroute_intent(embedder: AzureEmbeddingClient, standalone_query: str) -> str:
    router = get_intent_router()
    query_vec = await embedder.aembed_query(standalone_query)  # List[float] shape: (dim,)
    query_vec = np.array(query_vec)[np.newaxis, :]  # → shape: (1, dim)
    best_score, best_intent = await router.aguide(query_vec)
    logger.info(f"Intent routing — score: {best_score}, intent: {best_intent}")
    return best_intent


async def _aembed_text(
        embedder: AzureEmbeddingClient,
        qdrant: QdrantManager,
        text: str,
) -> tuple[List[float], List[List[float]]]:
    dense_task = embedder.aembed_documents([text])
    colbert_task = asyncio.to_thread(qdrant.embed_colbert_query, text)
    dense_embeds, colbert_result = await asyncio.gather(dense_task, colbert_task)

    dense_vec = dense_embeds[0]
    colbert_vec = [list(map(float, token)) for token in colbert_result]
    return dense_vec, colbert_vec


async def _avector_search(
        qdrant: QdrantManager,
        mean_dense: List[float],
        mean_colbert: List[List[float]],
        standalone_query: str,
        top_k: int = VECTOR_SEARCH_TOP_K,
) -> List[ChunkDict]:
    try:
        points = await qdrant.ahybrid_search(
            dense_query_vec=mean_dense,
            colbert_query_vec=mean_colbert,
            bm25_query=standalone_query,
            top_k=top_k,
        )
        return [
            ChunkDict(
                id=str(pt.id) if hasattr(pt, "id") else "",
                metadata={k: v for k, v in (pt.payload or {}).items() if k != "text"},
                text=(pt.payload or {}).get("text", ""),
                score=pt.score if hasattr(pt, "score") else 0.0,
            )
            for pt in points
        ]
    except Exception:
        logger.exception("Vector search failed")
        return []


async def _afetch_neighbor_chunks(
        qdrant: QdrantManager,
        sorted_chunks: List[ChunkDict],
) -> List[ChunkDict]:
    existing_ids = {c["id"] for c in sorted_chunks if c.get("id")}
    neighbor_ids: List[str] = []

    for c in sorted_chunks:
        meta = c.get("metadata", {})

        prev_list = meta.get("previous", [])
        if prev_list:
            uid = prev_list[NEIGHBOR_PREV_INDEX]
            if uid and uid not in existing_ids:
                neighbor_ids.append(uid)
                existing_ids.add(uid)

        next_list = meta.get("next", [])
        if next_list:
            uid = next_list[NEIGHBOR_NEXT_INDEX]
            if uid and uid not in existing_ids:
                neighbor_ids.append(uid)
                existing_ids.add(uid)

    if not neighbor_ids:
        return []

    try:
        neighbor_points = await qdrant.aget_by_ids(neighbor_ids)
        return [
            ChunkDict(
                id=str(pt.id) if hasattr(pt, "id") else "",
                metadata={k: v for k, v in (pt.payload or {}).items() if k != "text"},
                text=(pt.payload or {}).get("text", ""),
                score=0.0,
            )
            for pt in neighbor_points
        ]
    except Exception:
        logger.exception("Fetch neighbor chunks by IDs failed")
        return []


def _merge_chunks(
        sorted_chunks: List[ChunkDict],
        neighbor_chunks: List[ChunkDict],
) -> List[ChunkDict]:
    seen_ids: set[str] = set()
    merged: List[ChunkDict] = []
    for chunk in sorted_chunks + neighbor_chunks:
        cid = chunk.get("id")
        if cid:
            if cid not in seen_ids:
                merged.append(chunk)
                seen_ids.add(cid)
        else:
            merged.append(chunk)
    return merged


async def _agenerate_answer(
        llm: AzureChatClient,
        user_input: str,
        detected_language: str,
        enriched_chunks: List[ChunkDict],
):
    final_prompt = build_final_prompt(
        user_input=user_input,
        detected_language=detected_language,
        relevant_chunks=enriched_chunks,
    )
    try:
        return (await llm.ainvoke_creative(final_prompt)).strip()
    except Exception:
        logger.exception("Failed to generate answer from LLM")


async def _arun_core_search(
        llm: AzureChatClient,
        embedder: AzureEmbeddingClient,
        standalone_query: str,
        detected_language: str,
) -> PipelineResult:
    """Core knowledge flow: HyDE → vector top 5 → rerank top 3 → enrich → answer.

    Falls back to web search when the vector store returns no candidates.
    """
    timer = StepTimer(f"core_search:{COLLECTION_NAME}")
    config: AppConfig = get_app_config()

    try:
        qdrant = get_qdrant_client(COLLECTION_NAME)

        async with timer.astep("clarity_and_hyde_parallel"):
            clarity_task = _acheck_input_clarity(llm, standalone_query, detected_language)
            hyde_task = _ahyde_generate(llm, standalone_query)
            clarity, hyde = await asyncio.gather(clarity_task, hyde_task)

        if not clarity.get("clear", True):
            logger.info(f"Input unclear (type={clarity.get('type')}) — returning clarification")
            return _make_clarity_result(clarity, detected_language)

        async with timer.astep("hyde_embedding"):
            dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, hyde)

        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, dense_vec, colbert_vec, standalone_query, top_k=CORE_VECTOR_TOP_K,
            )

        if not sorted_chunks:
            logger.info("No vector results — triggering web search fallback")
            async with timer.astep("web_search_fallback_no_chunks"):
                rs = await web_rag_answer(llm, embedder, standalone_query, detected_language)
            return _make_web_search_result(rs["answer"], rs["sources"], detected_language)

        async with timer.astep("rerank"):
            reranked = await arerank_chunks(standalone_query, sorted_chunks, CORE_RERANK_TOP_K)

        async with timer.astep("enrich_chunks"):
            neighbor_chunks = await _afetch_neighbor_chunks(qdrant, reranked)
            enriched_chunks = _merge_chunks(reranked, neighbor_chunks)

        async with timer.astep("generate_answer"):
            answer = await _agenerate_answer(
                llm, standalone_query, detected_language, enriched_chunks,
            )

        sources = extract_sources(reranked, config.APP_DOMAIN)
        return _make_rag_result(
            response=answer,
            sources=sources,
            detected_language=detected_language,
            answer_satisfied=True,
        )

    finally:
        timer.summary()


async def _arun_overall_search(
        llm: AzureChatClient,
        embedder: AzureEmbeddingClient,
        standalone_query: str,
        detected_language: str,
) -> PipelineResult:
    """Overall-course flow: no HyDE, no web search, no LLM relevance filter.

    The overall collection holds a small set of static metadata chunks, so we
    embed the query directly and pass whatever hybrid-search returns to the
    generator. When nothing is found we return a templated "no result" reply
    instead of web fallback (web content hallucinates non-existent courses).
    """
    timer = StepTimer(f"overall_search:{OVERALL_COLLECTION_NAME}")

    try:
        qdrant = get_qdrant_client(OVERALL_COLLECTION_NAME)

        async with timer.astep("clarity_check"):
            clarity = await _acheck_input_clarity(llm, standalone_query, detected_language)

        if not clarity.get("clear", True):
            logger.info(f"Input unclear (type={clarity.get('type')}) — returning clarification")
            return _make_clarity_result(clarity, detected_language)

        async with timer.astep("query_embedding"):
            dense_vec, colbert_vec = await _aembed_text(
                embedder, qdrant, standalone_query,
            )

        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, dense_vec, colbert_vec, standalone_query, top_k=VECTOR_SEARCH_TOP_K,
            )

        if not sorted_chunks:
            logger.info("Overall search: no vector results — returning no-result reply")
            no_result = NO_RESULT_RESPONSE_MAP.get(detected_language) or ""
            return _make_rag_result(
                response=no_result,
                sources=[],
                detected_language=detected_language,
                answer_satisfied=False,
            )

        async with timer.astep("generate_answer"):
            answer = await _agenerate_answer(
                llm, standalone_query, detected_language, sorted_chunks,
            )

        return _make_rag_result(
            response=answer,
            sources=[],
            detected_language=detected_language,
            answer_satisfied=True,
        )

    finally:
        timer.summary()


async def async_pipeline_dispatch(
        user_input: str,
        conversation_id: Optional[str] = None,
) -> PipelineResult:
    timer = StepTimer("dispatch")

    try:
        llm = get_openai_chat_client()
        embedder = get_openai_embedding_client()

        async with timer.astep("validate_and_prepare"):
            detected_language, standalone_query, error_response = (
                await _avalidate_and_prepare(user_input, conversation_id, llm)
            )
            if error_response:
                return _make_off_topic_result(error_response, "English")

        async with timer.astep("quiz_keyword_check"):
            if is_quiz_intent(user_input) or is_quiz_intent(standalone_query):
                return _make_quiz_result()

        async with timer.astep("intent_routing"):
            intent = await _aroute_intent(embedder, standalone_query)

        if intent == INTENT_OFF_TOPIC:
            return _make_off_topic_result(
                OFF_TOPIC_RESPONSE_MAP.get(detected_language), detected_language
            )

        if intent == INTENT_OVERALL_COURSE_KNOWLEDGE:
            return await _arun_overall_search(
                llm=llm,
                embedder=embedder,
                standalone_query=standalone_query,
                detected_language=detected_language,
            )

        return await _arun_core_search(
            llm=llm,
            embedder=embedder,
            standalone_query=standalone_query,
            detected_language=detected_language,
        )

    finally:
        timer.summary()


# ── Streaming pipeline ────────────────────────────────────────────────────────
# Events yielded by `async_pipeline_dispatch_stream`:
#   {"type": "meta",  "intent": str, "detected_language": str|None,
#                      "sources": list, "answer_satisfied": bool,
#                      "web_search_used": bool}
#   {"type": "delta", "content": str}              # one or many; concat = full answer
#   {"type": "done"}                                # terminal marker
# Non-streaming outcomes (off-topic / quiz / clarity / no-result / input-too-long)
# emit meta + a single delta with the full reply + done, so clients use one path.

async def _astream_answer(
        llm: AzureChatClient,
        user_input: str,
        detected_language: str,
        enriched_chunks: List[ChunkDict],
) -> AsyncGenerator[str, None]:
    final_prompt = build_final_prompt(
        user_input=user_input,
        detected_language=detected_language,
        relevant_chunks=enriched_chunks,
    )
    try:
        async for piece in llm.astream_creative(final_prompt):
            yield piece
    except Exception:
        logger.exception("Failed to stream answer from LLM")


def _meta_event(result: PipelineResult) -> Dict[str, Any]:
    raw_sources = result.get("sources") or []
    sources = [
        s.model_dump() if isinstance(s, ChatSourceModel) else s
        for s in raw_sources
    ]
    return {
        "type": "meta",
        "intent": result.get("intent"),
        "detected_language": result.get("detected_language"),
        "sources": sources,
        "answer_satisfied": result.get("answer_satisfied", True),
        "web_search_used": result.get("web_search_used", False),
    }


def _full_reply_events(result: PipelineResult) -> List[Dict[str, Any]]:
    """Emit a non-streaming result (clarity/off-topic/etc.) as meta+delta+done."""
    return [
        _meta_event(result),
        {"type": "delta", "content": result.get("response") or ""},
        {"type": "done"},
    ]


async def _astream_core_search(
        llm: AzureChatClient,
        embedder: AzureEmbeddingClient,
        standalone_query: str,
        detected_language: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    timer = StepTimer(f"core_search_stream:{COLLECTION_NAME}")
    config: AppConfig = get_app_config()

    try:
        qdrant = get_qdrant_client(COLLECTION_NAME)

        async with timer.astep("clarity_and_hyde_parallel"):
            clarity_task = _acheck_input_clarity(llm, standalone_query, detected_language)
            hyde_task = _ahyde_generate(llm, standalone_query)
            clarity, hyde = await asyncio.gather(clarity_task, hyde_task)

        if not clarity.get("clear", True):
            for ev in _full_reply_events(_make_clarity_result(clarity, detected_language)):
                yield ev
            return

        async with timer.astep("hyde_embedding"):
            dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, hyde)

        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, dense_vec, colbert_vec, standalone_query, top_k=CORE_VECTOR_TOP_K,
            )

        if not sorted_chunks:
            async with timer.astep("web_search_fallback_no_chunks"):
                rs = await web_rag_answer(llm, embedder, standalone_query, detected_language)
            for ev in _full_reply_events(
                _make_web_search_result(rs["answer"], rs["sources"], detected_language)
            ):
                yield ev
            return

        async with timer.astep("rerank"):
            reranked = await arerank_chunks(standalone_query, sorted_chunks, CORE_RERANK_TOP_K)

        async with timer.astep("enrich_chunks"):
            neighbor_chunks = await _afetch_neighbor_chunks(qdrant, reranked)
            enriched_chunks = _merge_chunks(reranked, neighbor_chunks)

        sources = extract_sources(reranked, config.APP_DOMAIN)
        yield _meta_event(_make_rag_result(
            response="", sources=sources,
            detected_language=detected_language, answer_satisfied=True,
        ))

        async with timer.astep("stream_answer"):
            async for piece in _astream_answer(
                llm, standalone_query, detected_language, enriched_chunks,
            ):
                yield {"type": "delta", "content": piece}

        yield {"type": "done"}

    finally:
        timer.summary()


async def _astream_overall_search(
        llm: AzureChatClient,
        embedder: AzureEmbeddingClient,
        standalone_query: str,
        detected_language: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    timer = StepTimer(f"overall_search_stream:{OVERALL_COLLECTION_NAME}")

    try:
        qdrant = get_qdrant_client(OVERALL_COLLECTION_NAME)

        async with timer.astep("clarity_check"):
            clarity = await _acheck_input_clarity(llm, standalone_query, detected_language)

        if not clarity.get("clear", True):
            for ev in _full_reply_events(_make_clarity_result(clarity, detected_language)):
                yield ev
            return

        async with timer.astep("query_embedding"):
            dense_vec, colbert_vec = await _aembed_text(
                embedder, qdrant, standalone_query,
            )

        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, dense_vec, colbert_vec, standalone_query, top_k=VECTOR_SEARCH_TOP_K,
            )

        if not sorted_chunks:
            no_result = NO_RESULT_RESPONSE_MAP.get(detected_language) or ""
            for ev in _full_reply_events(_make_rag_result(
                response=no_result, sources=[],
                detected_language=detected_language, answer_satisfied=False,
            )):
                yield ev
            return

        yield _meta_event(_make_rag_result(
            response="", sources=[],
            detected_language=detected_language, answer_satisfied=True,
        ))

        async with timer.astep("stream_answer"):
            async for piece in _astream_answer(
                llm, standalone_query, detected_language, sorted_chunks,
            ):
                yield {"type": "delta", "content": piece}

        yield {"type": "done"}

    finally:
        timer.summary()


async def async_pipeline_dispatch_stream(
        user_input: str,
        conversation_id: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Streaming dispatch. See event schema above the helpers."""
    timer = StepTimer("dispatch_stream")

    try:
        llm = get_openai_chat_client()
        embedder = get_openai_embedding_client()

        async with timer.astep("validate_and_prepare"):
            detected_language, standalone_query, error_response = (
                await _avalidate_and_prepare(user_input, conversation_id, llm)
            )
            if error_response:
                for ev in _full_reply_events(
                    _make_off_topic_result(error_response, "English")
                ):
                    yield ev
                return

        async with timer.astep("quiz_keyword_check"):
            if is_quiz_intent(user_input) or is_quiz_intent(standalone_query):
                for ev in _full_reply_events(_make_quiz_result()):
                    yield ev
                return

        async with timer.astep("intent_routing"):
            intent = await _aroute_intent(embedder, standalone_query)

        if intent == INTENT_OFF_TOPIC:
            for ev in _full_reply_events(_make_off_topic_result(
                OFF_TOPIC_RESPONSE_MAP.get(detected_language), detected_language,
            )):
                yield ev
            return

        substream = (
            _astream_overall_search if intent == INTENT_OVERALL_COURSE_KNOWLEDGE
            else _astream_core_search
        )
        async for ev in substream(
            llm=llm,
            embedder=embedder,
            standalone_query=standalone_query,
            detected_language=detected_language,
        ):
            yield ev

    finally:
        timer.summary()
