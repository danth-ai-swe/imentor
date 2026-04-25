"""StateGraph assembly for the hybrid agent pipeline.

Topology (matches docs/superpowers/specs/2026-04-25-langgraph-agent-search-design.md §4):

    validate_input → detect_and_rewrite → quiz_check → clarity_check
        → agent_decide ⇄ tool_executor (loop, max 3 tool calls)
        → post_router
              → rerank → enrich → generate → finalize    (core path)
              → generate → finalize                       (overall path)
              → finalize                                  (web/clarification/no-result)
"""

from typing import Optional

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.rag.search.agent.agent_node import agent_decide_node
from src.rag.search.agent.nodes import (
    clarity_check_node,
    detect_and_rewrite_node,
    enrich_node,
    finalize_node,
    generate_node,
    quiz_check_node,
    rerank_node,
    validate_input_node,
)
from src.rag.search.agent.state import AgentState, make_initial_state
from src.rag.search.agent.tools import ALL_TOOLS

MAX_TOOL_CALLS = 3


# ── Edge predicates ─────────────────────────────────────────────────────────

def _route_after_validate(state: AgentState) -> str:
    return "finalize" if state.get("early_exit_reason") else "detect_and_rewrite"


def _route_after_detect(state: AgentState) -> str:
    return "finalize" if state.get("early_exit_reason") else "quiz_check"


def _route_after_quiz(state: AgentState) -> str:
    return "finalize" if state.get("early_exit_reason") else "clarity_check"


def _route_after_clarity(state: AgentState) -> str:
    return "finalize" if state.get("early_exit_reason") else "agent_decide"


def _route_after_agent(state: AgentState) -> str:
    """If the agent's last message has tool_calls and we're under the limit,
    execute tools; otherwise move to post-processing."""
    messages = state.get("messages") or []
    if not messages:
        return "post_router"

    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls and state.get("tool_call_count", 0) < MAX_TOOL_CALLS:
        return "tools"
    return "post_router"


def _post_router(state: AgentState) -> str:
    """After the agent loop, decide which post-processing path to take."""
    if state.get("clarification") or state.get("web_answer"):
        return "finalize"

    selected = state.get("selected_collection")
    chunks = state.get("chunks") or []

    if selected == "core" and chunks:
        return "rerank"
    if selected == "overall" and chunks:
        return "generate"
    return "finalize"  # no usable result


# ── Builder ─────────────────────────────────────────────────────────────────

def build_agent_graph():
    builder = StateGraph(AgentState)

    builder.add_node("validate_input", validate_input_node)
    builder.add_node("detect_and_rewrite", detect_and_rewrite_node)
    builder.add_node("quiz_check", quiz_check_node)
    builder.add_node("clarity_check", clarity_check_node)
    builder.add_node("agent_decide", agent_decide_node)
    # handle_tool_errors=True (the default in langgraph 1.x) means a tool
    # exception or schema-validation failure becomes a ToolMessage error
    # appended to state["messages"], not a raise. The agent then sees the
    # error on its next turn and can switch tools — this implements the
    # spec §9 "retry once with feedback" behavior. MAX_TOOL_CALLS=3 is the
    # absolute safety cap.
    builder.add_node("tools", ToolNode(ALL_TOOLS, handle_tool_errors=True))
    builder.add_node("rerank", rerank_node)
    builder.add_node("enrich", enrich_node)
    builder.add_node("generate", generate_node)
    builder.add_node("finalize", finalize_node)
    # post_router is a routing-only node; declare a no-op passthrough.
    builder.add_node("post_router", lambda s: {})

    builder.add_edge(START, "validate_input")
    builder.add_conditional_edges("validate_input", _route_after_validate,
                                  {"finalize": "finalize", "detect_and_rewrite": "detect_and_rewrite"})
    builder.add_conditional_edges("detect_and_rewrite", _route_after_detect,
                                  {"finalize": "finalize", "quiz_check": "quiz_check"})
    builder.add_conditional_edges("quiz_check", _route_after_quiz,
                                  {"finalize": "finalize", "clarity_check": "clarity_check"})
    builder.add_conditional_edges("clarity_check", _route_after_clarity,
                                  {"finalize": "finalize", "agent_decide": "agent_decide"})
    builder.add_conditional_edges("agent_decide", _route_after_agent,
                                  {"tools": "tools", "post_router": "post_router"})
    builder.add_edge("tools", "agent_decide")  # loop back

    builder.add_conditional_edges("post_router", _post_router,
                                  {"rerank": "rerank", "generate": "generate", "finalize": "finalize"})
    builder.add_edge("rerank", "enrich")
    builder.add_edge("enrich", "generate")
    builder.add_edge("generate", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph()
    return _compiled_graph


async def run_agent_pipeline(user_input: str, conversation_id: Optional[str] = None) -> dict:
    """Convenience wrapper. Returns a dict with the same fields as PipelineResult.
    Wraps the graph in a top-level try/except so unexpected errors return a
    safe off_topic response instead of bubbling up."""
    from src.constants.app_constant import INTENT_OFF_TOPIC, OFF_TOPIC_RESPONSE_MAP
    from src.utils.logger_utils import logger

    try:
        graph = get_compiled_graph()
        initial = make_initial_state(user_input, conversation_id)
        final_state = await graph.ainvoke(initial)
        return {
            "intent": final_state.get("intent"),
            "response": final_state.get("response"),
            "detected_language": final_state.get("detected_language"),
            "answer_satisfied": final_state.get("answer_satisfied", False),
            "web_search_used": final_state.get("web_search_used", False),
            "sources": final_state.get("sources") or [],
        }
    except Exception:
        logger.exception("[agent_pipeline] top-level failure")
        return {
            "intent": INTENT_OFF_TOPIC,
            "response": OFF_TOPIC_RESPONSE_MAP.get("English"),
            "detected_language": "English",
            "answer_satisfied": False,
            "web_search_used": False,
            "sources": [],
        }
