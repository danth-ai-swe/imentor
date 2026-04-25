"""SSE event generator for the agent pipeline.

Compiles a copy of the agent StateGraph with `interrupt_before=["generate"]`
plus a MemorySaver checkpointer so the agent runs every node up to (but
not including) answer generation. Once the graph stops, this module
inspects final state to decide which branch the request took, emits the
appropriate `meta` event, and either:

  - emits a single `delta` with the prebuilt response (clarification,
    web, no-result, early-exit), or
  - calls `llm.astream_creative(...)` directly to forward token-level
    chunks as `delta` events (core / overall RAG paths).

This mirrors the dual-path pattern already in `pipeline.py`'s
`async_pipeline_dispatch_stream`.
"""

import threading
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver

from src.apis.app_model import ChatSourceModel
from src.config.app_config import get_app_config
from src.constants.app_constant import (
    INTENT_CORE_KNOWLEDGE,
    INTENT_OFF_TOPIC,
    INTENT_OVERALL_COURSE_KNOWLEDGE,
    NO_RESULT_RESPONSE_MAP,
)
from src.rag.llm.chat_llm import get_openai_chat_client
from src.rag.search.agent.graph import build_agent_graph
from src.rag.search.agent.state import AgentState, make_initial_state
from src.rag.search.entrypoint import build_final_prompt
from src.rag.search.pipeline import extract_sources
from src.utils.logger_utils import logger

config = get_app_config()

_streaming_graph = None
_streaming_lock = threading.Lock()


def get_streaming_graph():
    """Compiled graph with `interrupt_before=["generate"]` so the streaming
    module can take over for the final answer LLM call. Cached per process."""
    global _streaming_graph
    if _streaming_graph is None:
        with _streaming_lock:
            if _streaming_graph is None:
                _streaming_graph = build_agent_graph(
                    checkpointer=MemorySaver(),
                    interrupt_before=["generate"],
                )
    return _streaming_graph


def _meta_event(
    state: AgentState,
    intent: Optional[str],
    sources: list,
    answer_satisfied: bool,
    web_search_used: bool,
) -> Dict[str, Any]:
    serialized_sources = [
        s.model_dump() if isinstance(s, ChatSourceModel) else s
        for s in sources
    ]
    return {
        "type": "meta",
        "intent": intent,
        "detected_language": state.get("detected_language"),
        "sources": serialized_sources,
        "answer_satisfied": answer_satisfied,
        "web_search_used": web_search_used,
        "tool_call_count": state.get("tool_call_count", 0),
    }


def _full_reply_events(
    state: AgentState,
    intent: Optional[str],
    response: str,
    answer_satisfied: bool,
    web_search_used: bool,
    sources: Optional[list] = None,
) -> list:
    """Non-streaming branches emit meta + a single delta + done."""
    return [
        _meta_event(state, intent, sources or [], answer_satisfied, web_search_used),
        {"type": "delta", "content": response or ""},
        {"type": "done"},
    ]


async def agent_stream_events(
    user_input: str,
    conversation_id: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run the agent graph up to the answer node, then stream the final
    answer's tokens (when applicable). Yields SSE-shaped event dicts."""
    graph = get_streaming_graph()
    initial = make_initial_state(user_input, conversation_id)
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

    state: AgentState = await graph.ainvoke(initial, config=cfg)

    # Branch 1: pre-processing early exit (input_too_long / unsupported_language /
    # quiz / clarification handled inside clarity-equivalent dispatcher path
    # before agent_decide). The pre-processing nodes set early_exit_reason and
    # the appropriate response; finalize already ran since the graph routes
    # straight to finalize on early-exit, no interrupt fires.
    early = state.get("early_exit_reason")
    if early in ("input_too_long", "unsupported_language", "quiz", "clarification"):
        for ev in _full_reply_events(
            state,
            intent=state.get("intent"),
            response=state.get("response") or "",
            answer_satisfied=False,
            web_search_used=False,
        ):
            yield ev
        return

    # Branch 2: tool-driven clarification (agent called ask_clarification)
    clarification = state.get("clarification")
    if clarification:
        intent = (
            INTENT_OFF_TOPIC
            if clarification.get("type") == "off_topic"
            else INTENT_CORE_KNOWLEDGE
        )
        for ev in _full_reply_events(
            state,
            intent=intent,
            response=clarification.get("response") or "",
            answer_satisfied=False,
            web_search_used=False,
        ):
            yield ev
        return

    # Branch 3: web answer (search_web tool returned a finished answer)
    web_answer = state.get("web_answer")
    if web_answer:
        for ev in _full_reply_events(
            state,
            intent=INTENT_CORE_KNOWLEDGE,
            response=web_answer,
            answer_satisfied=True,
            web_search_used=True,
            sources=state.get("sources") or [],
        ):
            yield ev
        return

    # Branch 4: core / overall RAG with usable chunks → stream the answer
    selected = state.get("selected_collection")
    chunks = state.get("chunks") or []
    detected_language = state.get("detected_language") or "English"

    if not chunks:
        # Agent stopped without producing usable output — return templated no-result
        no_result = NO_RESULT_RESPONSE_MAP.get(detected_language) or ""
        for ev in _full_reply_events(
            state,
            intent=INTENT_CORE_KNOWLEDGE,
            response=no_result,
            answer_satisfied=False,
            web_search_used=False,
        ):
            yield ev
        return

    if selected == "core":
        sources = extract_sources(chunks, config.APP_DOMAIN)
        intent = INTENT_CORE_KNOWLEDGE
    else:  # "overall"
        sources = []
        intent = INTENT_OVERALL_COURSE_KNOWLEDGE

    yield _meta_event(
        state,
        intent=intent,
        sources=sources,
        answer_satisfied=True,
        web_search_used=False,
    )

    llm = get_openai_chat_client()
    final_prompt = build_final_prompt(
        user_input=state["standalone_query"],
        detected_language=detected_language,
        relevant_chunks=chunks,
    )

    try:
        async for chunk in llm.astream_creative(final_prompt):
            yield {"type": "delta", "content": chunk}
    except Exception:
        logger.exception("[agent_stream] LLM token stream failed mid-flight")
        # Caller emits an `error` event; we just stop yielding deltas.
        return

    yield {"type": "done"}
