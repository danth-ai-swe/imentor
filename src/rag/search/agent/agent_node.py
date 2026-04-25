"""LLM tool-calling node — the dispatcher brain.

Uses `langchain_openai.AzureChatOpenAI.bind_tools` because the existing
custom AzureChatClient does not expose tool-calling. We construct it once
(singleton) using the same Azure config as the custom client, so the model
and credentials match production.
"""

import threading
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from src.config.app_config import get_app_config
from src.rag.search.agent.prompts import AGENT_DISPATCHER_SYSTEM_PROMPT
from src.rag.search.agent.state import AgentState
from src.rag.search.agent.tools import ALL_TOOLS
from src.utils.logger_utils import logger

config = get_app_config()

_llm_instance: AzureChatOpenAI | None = None
_bound_llm = None
_llm_lock = threading.Lock()


def _get_bound_llm():
    """Cached `AzureChatOpenAI.bind_tools(ALL_TOOLS)` so the binding is built
    once per process. Uses OPENAI_DISPATCH_MODEL if set (typically a faster
    smaller deployment like gpt-4o-mini) and falls back to OPENAI_CHAT_MODEL
    otherwise. max_tokens=512 caps dispatcher output — it only emits
    tool_calls + a one-sentence reasoning string, not a full answer."""
    global _bound_llm
    if _bound_llm is None:
        with _llm_lock:
            if _bound_llm is None:
                deployment = config.OPENAI_DISPATCH_MODEL or config.OPENAI_CHAT_MODEL
                llm = AzureChatOpenAI(
                    azure_endpoint=config.OPENAI_API_BASE,
                    api_key=config.OPENAI_API_KEY,
                    api_version=config.OPENAI_API_VERSION,
                    azure_deployment=deployment,
                    temperature=0.0,
                    timeout=config.GPT_TIMEOUT,
                    max_retries=config.GPT_MAX_RETRIES,
                    max_tokens=512,
                )
                _bound_llm = llm.bind_tools(ALL_TOOLS)
    return _bound_llm


def _summarize_tool_history(messages: list) -> str:
    """One-line summary of which tools have been called so the agent
    knows what it has already tried."""
    seen = []
    for m in messages:
        # Tool messages from prior turns
        if isinstance(m, dict):
            content = m.get("content", "")
            role = m.get("role")
        else:
            content = getattr(m, "content", "")
            role = getattr(m, "type", None)
        if role == "tool":
            seen.append(content[:120])
    return " | ".join(seen) if seen else "(none yet)"


async def agent_decide_node(state: AgentState) -> Dict[str, Any]:
    """Run one turn of the dispatcher LLM. Returns a state update with the
    LLM's reply (which may contain tool_calls) appended to messages.

    On the first turn we also seed `messages` with the HumanMessage so that
    on subsequent turns the chain has the original question for the LLM to
    anchor against. add_messages dedupes by message id, so the seed is
    persisted once."""
    llm = _get_bound_llm()

    system = AGENT_DISPATCHER_SYSTEM_PROMPT.format(
        detected_language=state.get("detected_language") or "English",
        standalone_query=state.get("standalone_query") or state.get("user_input"),
        tool_history=_summarize_tool_history(state.get("messages") or []),
    )

    if not state.get("messages"):
        human = HumanMessage(content=state.get("standalone_query") or state["user_input"])
        prompt_messages = [SystemMessage(content=system), human]
        response = await llm.ainvoke(prompt_messages)
        update = {"messages": [human, response]}
    else:
        # Subsequent turns: re-prepend the system prompt + replay full history
        # (which already contains the seeded HumanMessage + prior AIMessage(s)
        # + ToolMessages from ToolNode).
        prompt_messages = [SystemMessage(content=system)] + list(state["messages"])
        response = await llm.ainvoke(prompt_messages)
        update = {"messages": [response]}

    tool_call_names = [tc.get("name") for tc in (getattr(response, "tool_calls", None) or [])]
    logger.info("[agent_decide] tool_calls=%s text_len=%d", tool_call_names, len(response.content or ""))

    return update
