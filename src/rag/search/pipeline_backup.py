from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import pandas as pd

from src.constants.app_constant import (
    COLLECTION_NAME,
    MAX_INPUT_CHARS,
    METADATA_NODE_XLSX,
)
from src.rag.db_vector import get_qdrant_client, QdrantManager
from src.rag.llm.chat_llm import AzureChatClient, get_openai_chat_client
from src.rag.llm.embedding_llm import AzureEmbeddingClient, get_openai_embedding_client
from src.rag.reflector import Reflection
from src.rag.search.entrypoint import (
    afetch_chat_history,
    build_final_prompt,
)
from src.rag.search.prompt import (
    ANSWER_ERROR_RESPONSE,
    EVALUATE_ANSWER_PROMPT,
    HYDE_ANALYSE_PROMPT,
    INPUT_TOO_LONG_RESPONSE,
    NO_RESULT_RESPONSE,
    OFF_TOPIC_FALLBACK_RESPONSE,
    OFF_TOPIC_PROMPT
)
from src.rag.semantic_router.router import SemanticRouter, Route, load_precomputed_embeddings
from src.rag.semantic_router.samples import coreKnowledgeSamples, offTopicSamples
from src.utils.app_utils import is_quiz_intent, parse_json_response
from src.utils.language_utils import get_detected_language
from src.utils.logger_utils import logger, StepTimer


@dataclass
class SearchResult:
    metadata: Dict[str, Any]
    text: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


async def _ahyde_analyse_query(
        llm: AzureChatClient,
        standalone_query: str,
        user_input: str,
) -> Dict[str, Any]:
    response_language = get_detected_language(user_input)
    prompt = HYDE_ANALYSE_PROMPT.format(
        response_language=response_language,
        standalone_query=standalone_query,
    )
    try:
        raw = (await llm.ainvoke(prompt)).strip()
        parsed = parse_json_response(raw)
        return {
            "search": bool(parsed.get("search", True)),
            "response": str(parsed.get("response", "")),
        }
    except Exception:
        logger.exception("HyDE analyse: unexpected error")
    return {"search": True, "response": ""}


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


def _build_pipeline_result(
        intent: str | None,
        response: str | None,
        *,
        results: list | None = None,
        detected_language: str | None = None,
        node_data: list | None = None,
        answer_satisfied: bool = False,
) -> Dict[str, Any]:
    return {
        "intent": intent,
        "response": response,
        "results": results or [],
        "suggested_questions": None,
        "detected_language": detected_language,
        "node_data": node_data,
        "answer_satisfied": answer_satisfied,
    }


async def async_pipeline_hyde_search(
        user_input: str,
        conversation_id: str | None = None,
) -> Dict[str, Any]:
    timer = StepTimer("hyde_search")

    if len(user_input) > MAX_INPUT_CHARS:
        return _build_pipeline_result(
            intent="off_topic",
            response=INPUT_TOO_LONG_RESPONSE.format(max_chars=MAX_INPUT_CHARS),
        )

    try:
        async with timer.astep("detect_language"):
            detected_language = get_detected_language(user_input)
            if detected_language is None:
                return _build_pipeline_result(
                    intent="off_topic",
                    response="Ngôn ngữ này hiện không được hỗ trợ. / This language is not currently supported.",
                )

        async with timer.astep("init_clients"):
            llm: AzureChatClient = get_openai_chat_client()
            embedder: AzureEmbeddingClient = get_openai_embedding_client()
            qdrant: QdrantManager = get_qdrant_client()
            qdrant.collection_name = COLLECTION_NAME

        async with timer.astep("fetch_chat_history"):
            chat_history = await afetch_chat_history(conversation_id)

        async with timer.astep("reflection"):
            standalone_query: str = await Reflection(llm).areflect(chat_history, user_input)

        async with timer.astep("quiz_keyword_check"):
            if is_quiz_intent(user_input) or is_quiz_intent(standalone_query):
                return _build_pipeline_result(
                    intent="quiz",
                    response=None,
                )

        async with timer.astep("intent_routing"):
            routes = [
                Route(name="core_knowledge", samples=coreKnowledgeSamples),
                Route(name="off_topic", samples=offTopicSamples),
            ]
            router = SemanticRouter(
                embedding=embedder,
                routes=routes,
                precomputed_embeddings=load_precomputed_embeddings(),
            )

            best_score, best_intent = await router.aguide(standalone_query)
            print(best_score)

            if best_intent == "off_topic":
                async with timer.astep("off_topic_response"):
                    response = await _agenerate_off_topic_response(
                        llm, user_input, detected_language
                    )
                return _build_pipeline_result(
                    intent="off_topic",
                    response=response,
                    detected_language=detected_language,
                )
        # Step 2: HYDE analysis
        async with timer.astep("hyde_analysis"):
            hyde_result = await _ahyde_analyse_query(
                llm, standalone_query, user_input
            )

        if not hyde_result["search"]:
            return _build_pipeline_result(
                intent="core_knowledge",
                response=hyde_result["response"],
                detected_language=detected_language,
            )

        async with timer.astep("vector_search"):
            try:
                points = await qdrant.ahybrid_search(query=standalone_query, top_k=3)
                sorted_chunks: List[Dict[str, Any]] = [
                    {
                        "id": str(pt.id) if hasattr(pt, "id") else "",
                        "metadata": {k: v for k, v in (pt.payload or {}).items() if k != "text"},
                        "text": (pt.payload or {}).get("text", ""),
                        "score": pt.score if hasattr(pt, "score") else 0.0,
                    }
                    for pt in points
                ]
            except Exception:
                logger.exception("Vector search failed")
                sorted_chunks = []

        if not sorted_chunks:
            return _build_pipeline_result(
                intent="core_knowledge",
                response=NO_RESULT_RESPONSE,
                results=[
                    SearchResult(
                        metadata=c.get("metadata", {}),
                        text=c.get("text", ""),
                        score=c.get("score", 0.0),
                    ).to_dict()
                    for c in sorted_chunks[:5]
                ],
                detected_language=detected_language,
            )

        # ── Fetch 1 nearest neighbor (previous + next) per relevant chunk ──
        async with timer.astep("fetch_neighbor_chunks"):
            existing_ids = {c["id"] for c in sorted_chunks if c.get("id")}
            neighbor_ids: List[str] = []
            for c in sorted_chunks:
                meta = c.get("metadata", {})
                # last element of previous = immediately preceding chunk
                prev_list = meta.get("previous", [])
                if prev_list:
                    uid = prev_list[-1]
                    if uid and uid not in existing_ids:
                        neighbor_ids.append(uid)
                        existing_ids.add(uid)
                # first element of next = immediately following chunk
                next_list = meta.get("next", [])
                if next_list:
                    uid = next_list[0]
                    if uid and uid not in existing_ids:
                        neighbor_ids.append(uid)
                        existing_ids.add(uid)

            neighbor_chunks: List[Dict[str, Any]] = []
            if neighbor_ids:
                try:
                    neighbor_points = await qdrant.aget_by_ids(neighbor_ids)
                    neighbor_chunks = [
                        {
                            "id": str(pt.id) if hasattr(pt, "id") else "",
                            "metadata": {
                                k: v for k, v in (pt.payload or {}).items() if k != "text"
                            },
                            "text": (pt.payload or {}).get("text", ""),
                            "score": 0.0,
                        }
                        for pt in neighbor_points
                    ]
                except Exception:
                    logger.exception("Fetch neighbor chunks by IDs failed")

        merged_chunks = sorted_chunks + neighbor_chunks
        seen_ids = set()
        enriched_chunks = []
        for chunk in merged_chunks:
            cid = chunk.get("id", None)
            if cid and cid not in seen_ids:
                enriched_chunks.append(chunk)
                seen_ids.add(cid)
            elif not cid:
                # If no id, include anyway (rare, fallback)
                enriched_chunks.append(chunk)

        async with timer.astep("load_node_data"):
            node_data_list = _load_node_data(sorted_chunks)

        async with timer.astep("generate_answer"):
            system_prompt, messages = build_final_prompt(
                user_input=user_input,
                detected_language=detected_language,
                relevant_chunks=enriched_chunks,
                node_data_list=node_data_list,
                chat_history_summary=chat_history
            )
            try:
                answer: str = await llm.acreate_agentic_chunker_message(
                    system_prompt=system_prompt,
                    messages=messages,
                )
            except Exception:
                logger.exception("Failed to generate answer from LLM")
                answer = ANSWER_ERROR_RESPONSE

        async with timer.astep("evaluate_answer"):
            answer_satisfied = await _aevaluate_answer_satisfied(llm, user_input, answer)

        return _build_pipeline_result(
            intent="core_knowledge",
            response=answer,
            results=[
                SearchResult(
                    metadata=c.get("metadata", {}),
                    text=c.get("text", ""),
                    score=c.get("score", 0.0),
                ).to_dict()
                for c in sorted_chunks
            ],
            detected_language=detected_language,
            node_data=node_data_list,
            answer_satisfied=answer_satisfied,
        )

    finally:
        timer.summary()


def _load_node_data(relevant_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Load node metadata from Excel, deduplicating by node_id."""
    df = pd.read_excel(METADATA_NODE_XLSX)
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
