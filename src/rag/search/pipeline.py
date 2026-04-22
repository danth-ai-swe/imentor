import asyncio
import json
from typing import Any, Dict, List, Optional

import numpy as np

from src.rag.search.searxng_search import web_rag_answer
from src.apis.app_model import ChatSourceModel
from src.config.app_config import AppConfig, get_app_config
from src.constants.app_constant import (
    COLLECTION_NAME,
    MAX_INPUT_CHARS,
    INTENT_CORE_KNOWLEDGE, INTENT_OFF_TOPIC, INTENT_QUIZ, NEIGHBOR_PREV_INDEX, VECTOR_SEARCH_TOP_K,
    NEIGHBOR_NEXT_INDEX, UNSUPPORTED_LANGUAGE_MSG, INPUT_TOO_LONG_RESPONSE, OFF_TOPIC_RESPONSE_MAP,
    INTENT_OVERALL_COURSE_KNOWLEDGE, OVERALL_COLLECTION_NAME,
)
from src.rag.db_vector import get_qdrant_client, QdrantManager
from src.rag.llm.chat_llm import AzureChatClient, get_openai_chat_client
from src.rag.llm.embedding_llm import AzureEmbeddingClient, get_openai_embedding_client
from src.rag.reflector import Reflection
from src.rag.search.entrypoint import build_final_prompt, afetch_chat_history, _build_chunk_filter_payload
from src.rag.search.model import PipelineResult, ChunkDict
from src.rag.search.prompt import DETECT_LANGUAGE_PROMPT, CLARITY_CHECK_PROMPT, HYDE_VARIANTS_PROMPT, \
    CHUNK_RELEVANCE_FILTER_PROMPT
from src.rag.semantic_router.intent_router_registry import get_intent_router
from src.utils.app_utils import is_quiz_intent, parse_json_response, mean_pool_colbert, mean_pool_dense
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


async def _afilter_relevant_chunks(
        llm: AzureChatClient,
        standalone_query: str,
        chunks: List[ChunkDict],
        max_text_chars: int = 400,
) -> List[ChunkDict]:
    """
    Dùng LLM lọc chunks thực sự liên quan.
    Gửi kèm metadata (source, course, module, lesson, page, node summary)
    lẫn nội dung text để LLM có đủ context phán xét.
    """
    if not chunks:
        return []

    chunks_payload = [
        _build_chunk_filter_payload(i, chunk, max_text_chars)
        for i, chunk in enumerate(chunks)
    ]

    chunks_json = json.dumps(chunks_payload, ensure_ascii=False, indent=2)

    prompt = CHUNK_RELEVANCE_FILTER_PROMPT.format(
        standalone_query=standalone_query,
        chunks_json=chunks_json,
    )

    try:
        raw = (await llm.ainvoke(prompt)).strip()
        indices = parse_json_response(raw)

        if not isinstance(indices, dict) or "relevant_indices" not in indices:
            logger.warning("Chunk filter returned unexpected format — keeping all chunks")
            return chunks

        raw_indices = indices["relevant_indices"]
        if not isinstance(raw_indices, list):
            logger.warning("relevant_indices is not a list — keeping all chunks")
            return chunks

        valid_indices = [
            i for i in raw_indices
            if isinstance(i, int) and 0 <= i < len(chunks)
        ]

        logger.info(
            f"Chunk relevance filter: {len(chunks)} total → "
            f"{len(valid_indices)} relevant "
            f"(dropped {len(chunks) - len(valid_indices)})"
        )

        return [chunks[i] for i in valid_indices]

    except Exception:
        logger.exception("Chunk relevance filter failed — keeping all chunks as fallback")
        return chunks


async def _avalidate_and_prepare(
        user_input: str,
        conversation_id: Optional[str],
        llm: AzureChatClient,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if len(user_input) > MAX_INPUT_CHARS:
        return None, None, INPUT_TOO_LONG_RESPONSE.format(max_chars=MAX_INPUT_CHARS)

    detected_language = await _adetect_language_llm(llm, user_input)
    if not detected_language:
        return None, None, UNSUPPORTED_LANGUAGE_MSG

    chat_history = await afetch_chat_history(llm, conversation_id)
    standalone_query = await Reflection(llm).areflect(chat_history, user_input)

    return detected_language, standalone_query, None


async def _ahyde_generate_variants(
        llm: AzureChatClient,
        standalone_query: str,
        response_language: str,
) -> list[str]:
    prompt = HYDE_VARIANTS_PROMPT.format(
        standalone_query=standalone_query,
        response_language=response_language,
    )
    try:
        raw = (await llm.ainvoke(prompt)).strip()
        parsed = parse_json_response(raw)
        variants = parsed.get("variants", [])
        if isinstance(variants, list) and variants:
            return variants
        return [standalone_query]
    except Exception:
        logger.exception("HyDE variant generation failed, using standalone_query")
        return [standalone_query]


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


async def _aembed_hyde_variants(
        embedder: AzureEmbeddingClient,
        qdrant: QdrantManager,
        variants: List[str],
) -> tuple[List[float], List[List[float]]]:
    dense_embeds = await embedder.aembed_documents(variants)

    colbert_tasks = [
        asyncio.to_thread(qdrant.embed_colbert_query, v) for v in variants
    ]
    colbert_results = await asyncio.gather(*colbert_tasks)

    colbert_embeds: List[List[List[float]]] = [
        [list(map(float, token)) for token in mat] for mat in colbert_results
    ]

    mean_dense = mean_pool_dense(dense_embeds)
    mean_colbert = mean_pool_colbert(colbert_embeds)
    return mean_dense, mean_colbert


async def _avector_search(
        qdrant: QdrantManager,
        mean_dense: List[float],
        mean_colbert: List[List[float]],
        standalone_query: str,
) -> List[ChunkDict]:
    try:
        points = await qdrant.ahybrid_search(
            dense_query_vec=mean_dense,
            colbert_query_vec=mean_colbert,
            bm25_query=standalone_query,
            top_k=VECTOR_SEARCH_TOP_K,
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
        return (await llm.ainvoke(final_prompt)).strip()
    except Exception:
        logger.exception("Failed to generate answer from LLM")


async def _arun_hyde_search(
        llm: AzureChatClient,
        embedder: AzureEmbeddingClient,
        collection_name: str,
        user_input: str,
        standalone_query: str,
        detected_language: str,
) -> PipelineResult:
    timer = StepTimer(f"hyde_search:{collection_name}")
    config: AppConfig = get_app_config()

    try:
        qdrant = get_qdrant_client(collection_name)

        # ── 5a+5b. clarity_check VÀ hyde_variants chạy song song ─────────────
        async with timer.astep("clarity_and_hyde_parallel"):
            clarity_task = _acheck_input_clarity(llm, standalone_query, detected_language)
            hyde_task = _ahyde_generate_variants(llm, standalone_query, detected_language)
            clarity, variants = await asyncio.gather(clarity_task, hyde_task)

        if not clarity.get("clear", True):
            logger.info(f"Input unclear (type={clarity.get('type')}) — returning clarification")
            return _make_clarity_result(clarity, detected_language)

        async with timer.astep("hyde_embedding"):
            mean_dense, mean_colbert = await _aembed_hyde_variants(
                embedder, qdrant, variants
            )

        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, mean_dense, mean_colbert, standalone_query
            )

        if not sorted_chunks:
            logger.info("No vector results — triggering web search fallback")
            async with timer.astep("web_search_fallback_no_chunks"):
                rs = await web_rag_answer(llm, embedder, standalone_query, detected_language)
            return _make_web_search_result(rs["answer"], rs["sources"], detected_language)

        async with timer.astep("enrich_chunks"):
            neighbor_chunks = await _afetch_neighbor_chunks(qdrant, sorted_chunks)
            enriched_chunks = _merge_chunks(sorted_chunks, neighbor_chunks)

        async with timer.astep("filter_relevant_chunks"):
            filtered_chunks = await _afilter_relevant_chunks(
                llm, standalone_query, enriched_chunks
            )

        if not filtered_chunks:
            logger.info("No relevant chunks after filtering — triggering web search fallback")
            async with timer.astep("web_search_fallback_no_relevant"):
                rs = await web_rag_answer(llm, embedder, standalone_query, detected_language)
            return _make_web_search_result(rs["answer"], rs["sources"], detected_language)

        async with timer.astep("generate_answer"):
            answer = await _agenerate_answer(
                llm, standalone_query, detected_language,
                filtered_chunks,
            )

        return _make_rag_result(
            response=answer,
            sources=extract_sources(sorted_chunks, config.APP_DOMAIN),
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

        # Pick collection theo intent. Mọi intent ngoài OVERALL → core collection.
        collection_name = (
            OVERALL_COLLECTION_NAME if intent == INTENT_OVERALL_COURSE_KNOWLEDGE
            else COLLECTION_NAME
        )

        return await _arun_hyde_search(
            llm=llm,
            embedder=embedder,
            collection_name=collection_name,
            user_input=user_input,
            standalone_query=standalone_query,
            detected_language=detected_language,
        )

    finally:
        timer.summary()
