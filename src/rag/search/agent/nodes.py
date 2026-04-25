"""Deterministic StateGraph nodes — everything before and after the agent.

Pre-processing (this file, top half):
    validate_input  → length check
    detect_and_rewrite → schedules LLM language-detect with asyncio.create_task
                         and runs the history-aware rewrite concurrently
    quiz_check      → keyword shortcut
    clarity_check   → existing CLARITY_CHECK_PROMPT

Post-processing (this file, bottom half — added in Task 8):
    rerank, enrich, generate, finalize
"""

import asyncio
from typing import Any, Dict

from src.apis.app_model import ChatSourceModel
from src.constants.app_constant import (
    INPUT_TOO_LONG_RESPONSE,
    INTENT_CORE_KNOWLEDGE,
    INTENT_OFF_TOPIC,
    INTENT_OVERALL_COURSE_KNOWLEDGE,
    INTENT_QUIZ,
    MAX_INPUT_CHARS,
    NO_RESULT_RESPONSE_MAP,
    OFF_TOPIC_RESPONSE_MAP,
    UNSUPPORTED_LANGUAGE_MSG,
)
from src.rag.llm.chat_llm import get_openai_chat_client
from src.rag.reflector import Reflection
from src.rag.search.agent.state import AgentState
from src.rag.search.entrypoint import afetch_chat_history
from src.rag.search.pipeline import (
    _acheck_input_clarity,
    _adetect_language_llm,
    _make_clarity_result,
)
from src.utils.app_utils import is_quiz_intent
from src.utils.logger_utils import logger


async def validate_input_node(state: AgentState) -> Dict[str, Any]:
    if len(state["user_input"]) > MAX_INPUT_CHARS:
        return {
            "early_exit_reason": "input_too_long",
            "response": INPUT_TOO_LONG_RESPONSE.format(max_chars=MAX_INPUT_CHARS),
            "intent": INTENT_OFF_TOPIC,
            "detected_language": "English",
        }
    return {}


async def detect_and_rewrite_node(state: AgentState) -> Dict[str, Any]:
    """Runs language detection and standalone-query rewrite concurrently,
    matching `_avalidate_and_prepare` in pipeline.py.

    Note: pipeline.py's prep_cache short-circuit is intentionally omitted
    — this driver is a stateless dev/test path. Add caching if/when the
    agent pipeline is promoted to production.
    """
    llm = get_openai_chat_client()
    user_input = state["user_input"]
    conversation_id = state.get("conversation_id")
    has_history = bool(conversation_id and conversation_id.strip())

    # Schedule language detection so it runs concurrently with the rewrite.
    language_task = asyncio.create_task(_adetect_language_llm(llm, user_input))

    if has_history:
        chat_history = await afetch_chat_history(llm, conversation_id)
        standalone_query = await Reflection(llm).areflect(chat_history, user_input)
    else:
        standalone_query = await Reflection(llm).areflect("", user_input)

    detected_language = await language_task
    if not detected_language:
        return {
            "early_exit_reason": "unsupported_language",
            "response": UNSUPPORTED_LANGUAGE_MSG,
            "intent": INTENT_OFF_TOPIC,
            "detected_language": "English",
        }

    return {
        "detected_language": detected_language,
        "standalone_query": standalone_query,
    }


async def quiz_check_node(state: AgentState) -> Dict[str, Any]:
    if is_quiz_intent(state["user_input"]) or is_quiz_intent(state.get("standalone_query") or ""):
        return {
            "early_exit_reason": "quiz",
            "response": None,
            "intent": INTENT_QUIZ,
            "detected_language": state.get("detected_language"),
        }
    return {}


async def clarity_check_node(state: AgentState) -> Dict[str, Any]:
    llm = get_openai_chat_client()
    clarity = await _acheck_input_clarity(
        llm,
        state["standalone_query"],
        state.get("detected_language") or "English",
    )

    if clarity.get("clear", True):
        return {}

    # Reuse pipeline's response shaping for the clarification message.
    placeholder = _make_clarity_result(clarity, state.get("detected_language"))
    return {
        "early_exit_reason": "clarification",
        "response": placeholder["response"],
        "intent": INTENT_CORE_KNOWLEDGE,
        "answer_satisfied": False,
        "detected_language": state.get("detected_language"),
    }


# ── Post-processing ─────────────────────────────────────────────────────────

from src.config.app_config import AppConfig, get_app_config
from src.constants.app_constant import COLLECTION_NAME, CORE_RERANK_TOP_K
from src.rag.db_vector import get_qdrant_client
from src.rag.search.entrypoint import build_final_prompt
from src.rag.search.pipeline import (
    _afetch_neighbor_chunks,
    _merge_chunks,
    extract_sources,
)
from src.rag.search.reranker import arerank_chunks


async def rerank_node(state: AgentState) -> Dict[str, Any]:
    chunks = state.get("chunks") or []
    if not chunks:
        return {}
    reranked = await arerank_chunks(state["standalone_query"], chunks, CORE_RERANK_TOP_K)
    return {"chunks": reranked}


async def enrich_node(state: AgentState) -> Dict[str, Any]:
    chunks = state.get("chunks") or []
    if not chunks:
        return {}
    qdrant = get_qdrant_client(COLLECTION_NAME)
    neighbors = await _afetch_neighbor_chunks(qdrant, chunks)
    merged = _merge_chunks(chunks, neighbors)
    return {"chunks": merged}


async def generate_node(state: AgentState) -> Dict[str, Any]:
    llm = get_openai_chat_client()
    config: AppConfig = get_app_config()
    final_prompt = build_final_prompt(
        user_input=state["standalone_query"],
        detected_language=state.get("detected_language") or "English",
        relevant_chunks=state.get("chunks") or [],
    )
    try:
        answer = (await llm.ainvoke_creative(final_prompt)).strip()
    except Exception:
        logger.exception("[generate_node] LLM failed")
        answer = ""

    sources = []
    if state.get("selected_collection") == "core":
        sources = extract_sources(state.get("chunks") or [], config.APP_DOMAIN)

    return {
        "response": answer,
        "sources": sources,
        "answer_satisfied": bool(answer),
    }


def finalize_node(state: AgentState) -> Dict[str, Any]:
    """Map AgentState to PipelineResult-shaped fields. Pure function — no I/O."""
    early = state.get("early_exit_reason")
    detected_language = state.get("detected_language") or "English"

    # Early-exit branches set their own response/intent already; just pass through.
    if early in ("input_too_long", "unsupported_language", "quiz", "clarification"):
        return {
            "intent": state.get("intent"),
            "response": state.get("response"),
            "detected_language": detected_language,
            "answer_satisfied": False,
            "web_search_used": False,
            "sources": state.get("sources") or [],
        }

    # Tool-driven outcomes:
    clarification = state.get("clarification")
    if clarification:
        intent = INTENT_OFF_TOPIC if clarification.get("type") == "off_topic" else INTENT_CORE_KNOWLEDGE
        return {
            "intent": intent,
            "response": clarification.get("response"),
            "detected_language": detected_language,
            "answer_satisfied": False,
            "web_search_used": False,
            "sources": [],
        }

    web_answer = state.get("web_answer")
    if web_answer:
        return {
            "intent": INTENT_CORE_KNOWLEDGE,
            "response": web_answer,
            "detected_language": detected_language,
            "answer_satisfied": True,
            "web_search_used": True,
            "sources": state.get("sources") or [],
        }

    selected = state.get("selected_collection")
    response = state.get("response")
    chunks = state.get("chunks") or []

    if selected == "overall":
        if response:
            return {
                "intent": INTENT_OVERALL_COURSE_KNOWLEDGE,
                "response": response,
                "detected_language": detected_language,
                "answer_satisfied": True,
                "web_search_used": False,
                "sources": [],
            }
        # No chunks found — return templated no-result.
        return {
            "intent": INTENT_OVERALL_COURSE_KNOWLEDGE,
            "response": NO_RESULT_RESPONSE_MAP.get(detected_language) or "",
            "detected_language": detected_language,
            "answer_satisfied": False,
            "web_search_used": False,
            "sources": [],
        }

    if selected == "core":
        if response and chunks:
            return {
                "intent": INTENT_CORE_KNOWLEDGE,
                "response": response,
                "detected_language": detected_language,
                "answer_satisfied": True,
                "web_search_used": False,
                "sources": state.get("sources") or [],
            }

    # Fallback: agent stopped without producing usable output.
    return {
        "intent": INTENT_CORE_KNOWLEDGE,
        "response": NO_RESULT_RESPONSE_MAP.get(detected_language) or "",
        "detected_language": detected_language,
        "answer_satisfied": False,
        "web_search_used": False,
        "sources": [],
    }
