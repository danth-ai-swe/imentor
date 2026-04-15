import asyncio
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd

from src.apis.app_model import ChatSourceModel
from src.config.app_config import AppConfig, get_app_config
from src.constants.app_constant import (
    COLLECTION_NAME,
    MAX_INPUT_CHARS,
    METADATA_NODE_XLSX,
)
from src.rag.db_vector import get_qdrant_client, QdrantManager
from src.rag.llm.chat_llm import AzureChatClient, get_openai_chat_client
from src.rag.llm.embedding_llm import AzureEmbeddingClient, get_openai_embedding_client
from src.rag.reflector import Reflection
from src.rag.search.entrypoint import build_final_prompt, afetch_chat_history
from src.rag.search.prompt import (
    ANSWER_ERROR_RESPONSE,
    HYDE_PROMPT,
    INPUT_TOO_LONG_RESPONSE,
    NO_RESULT_RESPONSE,
    OFF_TOPIC_FALLBACK_RESPONSE,
    OFF_TOPIC_PROMPT, EVALUATE_CHUNK_PROMPT, EVALUATE_ANSWER_PROMPT,
)
from src.rag.search.web_search import asearch_and_extract, agenerate_web_answer
from src.rag.semantic_router.intent_router_registry import get_intent_router
from src.utils.app_utils import is_quiz_intent, parse_json_response
from src.utils.language_utils import get_detected_language
from src.utils.logger_utils import logger, StepTimer

# =============================================================================
# Constants
# =============================================================================

VECTOR_SEARCH_TOP_K: int = 2
NEIGHBOR_PREV_INDEX: int = -1  # phần tử cuối của list "previous" = chunk liền trước
NEIGHBOR_NEXT_INDEX: int = 0  # phần tử đầu của list "next"     = chunk liền sau
INTENT_CORE_KNOWLEDGE: str = "core_knowledge"
INTENT_OFF_TOPIC: str = "off_topic"
INTENT_QUIZ: str = "quiz"
UNSUPPORTED_LANGUAGE_RESPONSE: str = (
    "Ngôn ngữ này hiện không được hỗ trợ. / This language is not currently supported."
)

# Câu mở đầu cố định khi trả về clarification, dịch theo ngôn ngữ detect được
_CLARIFICATION_PREFIX: Dict[str, str] = {
    "Vietnamese": "Có phải bạn đang muốn hỏi về:",
    "English": "Are you asking about:",
    "French": "Voulez-vous demander à propos de :",
    "Spanish": "¿Está preguntando sobre:",
    "Japanese": "について質問していますか：",
    "Korean": "에 대해 질문하고 계신가요:",
    "Chinese (Simplified)": "您是想询问有关：",
    "Chinese (Traditional)": "您是想詢問有關：",
    "Thai": "คุณกำลังถามเกี่ยวกับ:",
    "German": "Meinen Sie:",
    "Portuguese": "Você está perguntando sobre:",
    "Indonesian": "Apakah Anda menanyakan tentang:",
    "Malay": "Adakah anda bertanya tentang:",
    "Arabic": "هل تسأل عن:",
    "Hindi": "क्या आप इसके बारे में पूछ रहे हैं:",
    "Russian": "Вы спрашиваете о:",
}
_CLARIFICATION_PREFIX_DEFAULT = "Are you asking about:"


# =============================================================================
# TypedDicts
# =============================================================================

class ChunkDict(TypedDict):
    id: str
    metadata: Dict[str, Any]
    text: str
    score: float


class PipelineResult(TypedDict):
    intent: Optional[str]
    response: Optional[str]
    detected_language: Optional[str]
    answer_satisfied: bool
    web_search_used: bool
    sources: List[Any]


# =============================================================================
# Clarification prefix helper
# =============================================================================

def _get_clarification_prefix(detected_language: Optional[str]) -> str:
    """Trả về câu mở đầu cố định phù hợp với ngôn ngữ detect được."""
    if not detected_language:
        return _CLARIFICATION_PREFIX_DEFAULT
    return _CLARIFICATION_PREFIX.get(detected_language, _CLARIFICATION_PREFIX_DEFAULT)


# =============================================================================
# Pipeline result factory methods
# =============================================================================

def _make_off_topic_result(
        response: str,
        detected_language: Optional[str] = None,
) -> PipelineResult:
    return PipelineResult(
        intent=INTENT_OFF_TOPIC,
        response=response,
        detected_language=detected_language,
        answer_satisfied=False,
        web_search_used=False,
        sources=[],
    )


def _make_spell_check_result(
        spell_check: str,
        response: List[str],
        detected_language: Optional[str],
) -> PipelineResult:
    """
    Xử lý spell_check: 'confirm' hoặc 'unclear' từ HYDE_PROMPT.
    - confirm: response có 4 items [intro, topic1, topic2, topic3]
    - unclear: response có 1 item [polite message]
    """
    return PipelineResult(
        intent=INTENT_CORE_KNOWLEDGE,
        response="\n".join(r.strip() for r in response if r and r.strip()),
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


def _make_clarification_result(
        questions: List[str],
        detected_language: Optional[str],
) -> PipelineResult:
    """
    Nối 3 câu hỏi 5W1H thành 1 string hoàn chỉnh theo format:

        <prefix cố định theo ngôn ngữ>\\n
        <câu hỏi 1>\\n
        <câu hỏi 2>\\n
        <câu hỏi 3>

    Ví dụ (Vietnamese):
        Có phải bạn đang muốn hỏi về:
        [What] Bạn đang hỏi về khái niệm chuyển giao rủi ro cụ thể nào?
        [Why] Bạn muốn hiểu mục đích của việc chuyển giao rủi ro là gì?
        [How] Cơ chế chuyển giao rủi ro được thực hiện thông qua hợp đồng nào?
    """
    prefix = _get_clarification_prefix(detected_language)
    valid_questions = [q.strip() for q in questions if q and q.strip()]
    response = prefix + "\n" + "\n".join(valid_questions)

    return PipelineResult(
        intent=INTENT_CORE_KNOWLEDGE,
        response=response,
        detected_language=detected_language,
        answer_satisfied=False,
        web_search_used=False,
        sources=[],
    )


# =============================================================================
# Vector pooling helpers
# =============================================================================

def _normalize_mean(arr: np.ndarray) -> np.ndarray:
    """Tính mean theo axis=0 rồi L2-normalize."""
    mean_vec = np.mean(arr, axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec


def _mean_pool_dense(embeddings: List[List[float]]) -> List[float]:
    arr = np.array(embeddings, dtype=np.float32)
    return _normalize_mean(arr).tolist()


def _mean_pool_colbert(embeddings: List[List[List[float]]]) -> List[List[float]]:
    all_tokens: List[List[float]] = [
        token for token_matrix in embeddings for token in token_matrix
    ]
    arr = np.array(all_tokens, dtype=np.float32)
    return [_normalize_mean(arr).tolist()]


# =============================================================================
# Source extraction
# =============================================================================

def _chunk_to_source(chunk: ChunkDict, app_domain: str) -> Optional[ChatSourceModel]:
    source_file: str = chunk.get("metadata", {}).get("file_name", "")
    if not source_file:
        return None
    return ChatSourceModel(
        name=source_file,
        url=f"{app_domain}/api/v1/file/{source_file}.pdf",
        page_number=chunk.get("metadata", {}).get("page_number", 1),
        total_pages=chunk.get("metadata", {}).get("total_pages", 1),
    )


def extract_sources(
        chunks: List[ChunkDict],
        app_domain: str,
) -> List[ChatSourceModel]:
    seen: Dict[str, ChatSourceModel] = {}
    for chunk in chunks:
        source = _chunk_to_source(chunk, app_domain)
        if source and source.name not in seen:
            seen[source.name] = source
    return list(seen.values())


# =============================================================================
# Thread-safe node metadata cache
# =============================================================================

_NODE_DF_CACHE: Optional[pd.DataFrame] = None
_NODE_DF_LOCK: asyncio.Lock = asyncio.Lock()


async def _aget_node_df() -> pd.DataFrame:
    """Load DataFrame từ Excel đúng 1 lần, thread-safe với asyncio.Lock."""
    global _NODE_DF_CACHE
    if _NODE_DF_CACHE is not None:
        return _NODE_DF_CACHE
    async with _NODE_DF_LOCK:
        if _NODE_DF_CACHE is None:
            _NODE_DF_CACHE = await asyncio.to_thread(pd.read_excel, METADATA_NODE_XLSX)
            logger.info(f"Loaded node metadata: {len(_NODE_DF_CACHE)} rows")
    return _NODE_DF_CACHE


async def _aload_node_data(
        relevant_chunks: List[ChunkDict],
) -> List[Dict[str, Any]]:
    df = await _aget_node_df()
    node_data_list: List[Dict[str, Any]] = []
    seen_node_ids: set[str] = set()

    for chunk in relevant_chunks:
        node_id = str(chunk.get("metadata", {}).get("node_id", ""))
        if not node_id or node_id == "N/A" or node_id in seen_node_ids:
            continue
        seen_node_ids.add(node_id)

        matched = df[df["Node ID"].astype(str) == node_id]
        if matched.empty:
            continue

        row = matched.iloc[0].drop(labels=["Node ID", "Source"], errors="ignore")
        node_data_list.append(row.to_dict())

    return node_data_list


# =============================================================================
# LLM helper functions
# =============================================================================

async def _ahyde_analyse_query(
        llm: AzureChatClient,
        standalone_query: str,
        response_language: str,
) -> dict | None:
    prompt = HYDE_PROMPT.format(
        response_language=response_language,
        standalone_query=standalone_query,
    )

    try:
        raw = (await llm.ainvoke(prompt)).strip()
        return parse_json_response(raw)

    except Exception:
        logger.exception("HyDE analyse: unexpected error")


async def _aevaluate_chunks_sufficient(
        llm: AzureChatClient,
        user_input: str,
        chunks: List[ChunkDict],
) -> bool:
    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{c['text']}" for i, c in enumerate(chunks)
    )
    prompt = EVALUATE_CHUNK_PROMPT.format(user_input=user_input, chunks=chunks_text)
    try:
        raw = (await llm.ainvoke(prompt)).strip()
        parsed = parse_json_response(raw)
        return bool(parsed.get("sufficient", True))
    except Exception:
        logger.exception("Evaluate chunks sufficient: unexpected error")
        return True


async def _agenerate_off_topic_response(
        llm: AzureChatClient,
        user_input: str,
        detected_language: str,
) -> str:
    prompt = OFF_TOPIC_PROMPT.format(
        detected_language=detected_language,
        user_input=user_input,
    )
    try:
        return (await llm.ainvoke(prompt)).strip().strip("\"'")
    except Exception:
        logger.exception("Off-topic response generation failed, using fallback")
        return OFF_TOPIC_FALLBACK_RESPONSE


# =============================================================================
# Web search fallback
# =============================================================================

async def _afallback_web_search(
        llm: AzureChatClient,
        user_input: str,
        standalone_query: str,
        detected_language: str,
) -> tuple[str, List[Any]]:
    extracted_results, search_meta = await asearch_and_extract(standalone_query)

    if not extracted_results:
        logger.warning("Web search fallback: no extracted results")
        return NO_RESULT_RESPONSE, []

    answer, source_items = await agenerate_web_answer(
        llm=llm,
        user_input=user_input,
        detected_language=detected_language,
        extracted_results=extracted_results,
        search_meta=search_meta,
    )

    return (answer or NO_RESULT_RESPONSE), source_items


# =============================================================================
# Pipeline sub-functions
# =============================================================================

async def _avalidate_and_prepare(
        user_input: str,
        conversation_id: Optional[str],
        llm: AzureChatClient,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if len(user_input) > MAX_INPUT_CHARS:
        return None, None, INPUT_TOO_LONG_RESPONSE.format(max_chars=MAX_INPUT_CHARS)

    detected_language = get_detected_language(user_input)
    if detected_language is None:
        return None, None, UNSUPPORTED_LANGUAGE_RESPONSE

    chat_history = await afetch_chat_history(conversation_id)
    standalone_query = await Reflection(llm).areflect(chat_history, user_input)

    return detected_language, standalone_query, None


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

    mean_dense = _mean_pool_dense(dense_embeds)
    mean_colbert = _mean_pool_colbert(colbert_embeds)
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
        node_data_list: List[Dict[str, Any]],
) -> str:
    system_prompt, messages = build_final_prompt(
        user_input=user_input,
        detected_language=detected_language,
        relevant_chunks=enriched_chunks,
        node_data_list=node_data_list,
    )
    try:
        return await llm.acreate_agentic_chunker_message(
            system_prompt=system_prompt,
            messages=messages,
        )
    except Exception:
        logger.exception("Failed to generate answer from LLM")
        return ANSWER_ERROR_RESPONSE


async def _aenrich_chunks(
        qdrant: QdrantManager,
        sorted_chunks: List[ChunkDict],
) -> tuple[List[ChunkDict], List[Dict[str, Any]]]:
    neighbor_chunks, node_data_list = await asyncio.gather(
        _afetch_neighbor_chunks(qdrant, sorted_chunks),
        _aload_node_data(sorted_chunks),
    )
    return _merge_chunks(sorted_chunks, neighbor_chunks), node_data_list


async def _aevaluate_answer_satisfied(
        llm: AzureChatClient,
        user_input: str,
        answer: str,
) -> bool:
    prompt = EVALUATE_ANSWER_PROMPT.format(user_input=user_input, answer=answer)
    try:
        raw = (await llm.ainvoke(prompt)).strip()
        parsed = parse_json_response(raw)
        return bool(parsed.get("satisfied", True))
    except Exception:
        logger.exception("Evaluate answer satisfied: unexpected error")
    return True


# =============================================================================
# Main pipeline
# =============================================================================

async def async_pipeline_hyde_search(
        user_input: str,
        conversation_id: Optional[str] = None,
) -> PipelineResult:
    timer = StepTimer("hyde_search")

    # Inject config 1 lần
    config: AppConfig = get_app_config()

    try:
        # ── 1. Init clients ──────────────────────────────────────────────────
        async with timer.astep("init_clients"):
            llm: AzureChatClient = get_openai_chat_client()
            embedder: AzureEmbeddingClient = get_openai_embedding_client()
            qdrant: QdrantManager = get_qdrant_client()
            qdrant.collection_name = COLLECTION_NAME

        # ── 2. Validate + detect language + reflect ──────────────────────────
        async with timer.astep("validate_and_prepare"):
            detected_language, standalone_query, error_response = (
                await _avalidate_and_prepare(user_input, conversation_id, llm)
            )
            if error_response:
                return _make_off_topic_result(error_response)
        # ── 3. Quiz intent check ─────────────────────────────────────────────
        async with timer.astep("quiz_keyword_check"):
            if is_quiz_intent(user_input) or is_quiz_intent(standalone_query):
                return _make_quiz_result()

        # ── 4. Semantic intent routing ───────────────────────────────────────
        async with timer.astep("intent_routing"):
            intent = await _aroute_intent(embedder, standalone_query)

        if intent == INTENT_OFF_TOPIC:
            async with timer.astep("off_topic_response"):
                response = await _agenerate_off_topic_response(
                    llm, user_input, detected_language
                )
            return _make_off_topic_result(response, detected_language)

        # ── 5. HyDE analysis ─────────────────────────────────────────────────
        async with timer.astep("hyde_analysis"):
            hyde_result = await _ahyde_analyse_query(llm, standalone_query, detected_language)

        # search=False → kiểm tra spell_check hoặc clarification
        if not hyde_result["search"]:
            spell_check = hyde_result.get("spell_check")

            if spell_check in ("confirm", "unclear"):
                return _make_spell_check_result(
                    spell_check=spell_check,
                    response=hyde_result.get("response", []),
                    detected_language=detected_language,
                )

            # CASE B / CASE C: 3 câu hỏi 5W1H
            return _make_clarification_result(hyde_result["response"], detected_language)

        # CASE A: dùng corrected_query thay original nếu có
        corrected_query = hyde_result.get("corrected_query")
        embed_query = corrected_query if corrected_query else standalone_query

        variants: List[str] = hyde_result["response"] or [embed_query]

        # ── 6. HyDE embedding ────────────────────────────────────────────────
        async with timer.astep("hyde_embedding"):
            mean_dense, mean_colbert = await _aembed_hyde_variants(embedder, qdrant, variants)

        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, mean_dense, mean_colbert, embed_query
            )

        # ── Fallback A: không có chunk ─────────────────────────────────────────
        if not sorted_chunks:
            logger.info("No vector results — triggering web search fallback")
            async with timer.astep("web_search_fallback_no_chunks"):
                answer, source_items = await _afallback_web_search(
                    llm, standalone_query, standalone_query, detected_language
                )
            return _make_web_search_result(answer, source_items, detected_language)

        # ── 8. Evaluate chunks ─────────────────────────────────────────────────
        # async with timer.astep("evaluate_chunks"):
        #     chunks_sufficient = await _aevaluate_chunks_sufficient(
        #         llm, user_input, sorted_chunks
        #     )

        # ── Fallback B: chunk không đủ liên quan ──────────────────────────────
        # if not chunks_sufficient:
        #     logger.info("Chunks not sufficient — triggering web search fallback")
        #     async with timer.astep("web_search_fallback_insufficient_chunks"):
        #         answer, source_items = await _afallback_web_search(
        #             llm, user_input, standalone_query, detected_language
        #         )
        #     return _make_web_search_result(answer, source_items, detected_language)

        # ── 9+10. Fetch neighbors + node metadata (parallel) ──────────────────
        async with timer.astep("enrich_chunks"):
            enriched_chunks, node_data_list = await _aenrich_chunks(qdrant, sorted_chunks)

        # ── 11. Generate answer ────────────────────────────────────────────────
        async with timer.astep("generate_answer"):
            answer = await _agenerate_answer(
                llm,
                standalone_query,
                detected_language,
                enriched_chunks,
                node_data_list
            )
            # ── 11. Evaluate answer ──────────────────────────────────────────────
            async with timer.astep("evaluate_answer"):
                answer_satisfied = await _aevaluate_answer_satisfied(llm, standalone_query, answer)

            # ── Fallback B: answer không thoả mãn ───────────────────────────────
            if not answer_satisfied:
                logger.info("Answer not satisfied — triggering web search fallback")
                async with timer.astep("web_search_fallback_unsatisfied"):
                    answer, source_items = await _afallback_web_search(
                        llm, standalone_query, standalone_query, detected_language
                    )
                return _make_web_search_result(answer, source_items, detected_language)

        # ── Normal RAG response ────────────────────────────────────────────────
        return _make_rag_result(
            response=answer,
            sources=extract_sources(sorted_chunks, config.APP_DOMAIN),
            detected_language=detected_language,
            answer_satisfied=True,
        )

    finally:
        timer.summary()
