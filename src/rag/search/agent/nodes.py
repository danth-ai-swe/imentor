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
