from typing import Annotated, Any, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.rag.search.model import ChunkDict


class AgentState(TypedDict, total=False):
    user_input: str
    conversation_id: Optional[str]

    detected_language: Optional[str]
    standalone_query: Optional[str]

    messages: Annotated[List[BaseMessage], add_messages]
    tool_call_count: int

    chunks: List[ChunkDict]
    sources: List[Any]
    web_answer: Optional[str]
    selected_collection: Optional[str]   # "core" | "overall" | "web"
    clarification: Optional[dict]

    response: Optional[str]
    answer_satisfied: bool
    web_search_used: bool
    intent: Optional[str]

    early_exit_reason: Optional[str]


def make_initial_state(user_input: str, conversation_id: Optional[str] = None) -> AgentState:
    return AgentState(
        user_input=user_input,
        conversation_id=conversation_id,
        detected_language=None,
        standalone_query=None,
        messages=[],
        tool_call_count=0,
        chunks=[],
        sources=[],
        web_answer=None,
        selected_collection=None,
        clarification=None,
        response=None,
        answer_satisfied=False,
        web_search_used=False,
        intent=None,
        early_exit_reason=None,
    )
