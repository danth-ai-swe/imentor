# LangGraph Hybrid Agent Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parallel LangGraph-based search pipeline in `src/rag/search/agent/` that mirrors `src/rag/search/pipeline.py` routing decisions but replaces the semantic-router + if/else dispatch with an LLM tool-calling agent. Produce a `PipelineResult`-compatible output and a CLI driver for testing.

**Architecture:** Hybrid B1 — pre-processing (validate/detect-lang/rewrite/quiz/clarity) and post-processing (rerank/enrich/generate) stay deterministic as `StateGraph` nodes. Between them, a single agent node uses `AzureChatOpenAI.bind_tools` to pick from 4 tools (`search_core_collection`, `search_overall_collection`, `search_web`, `ask_clarification`) in a multi-step ReAct loop (max 3 tool calls). State flows through `AgentState` TypedDict; tools receive state via LangGraph's `InjectedState`.

**Tech Stack:** Python 3.10+, `langgraph` (1.1.8), `langchain-openai` (1.1.14), `langchain-core`, existing `AzureChatClient`, existing Qdrant/embedder/reranker helpers from `src/rag/`.

**Reference spec:** `docs/superpowers/specs/2026-04-25-langgraph-agent-search-design.md`

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `src/rag/search/agent/__init__.py` | Create | Package marker; re-export `build_agent_graph`, `run_agent_pipeline` |
| `src/rag/search/agent/state.py` | Create | `AgentState` TypedDict + `make_initial_state` helper |
| `src/rag/search/agent/prompts.py` | Create | `AGENT_DISPATCHER_SYSTEM_PROMPT` template |
| `src/rag/search/agent/tools.py` | Create | 4 `@tool` definitions wrapping existing helpers |
| `src/rag/search/agent/agent_node.py` | Create | `AzureChatOpenAI` factory + `agent_decide` node |
| `src/rag/search/agent/nodes.py` | Create | Deterministic nodes (pre + post processing) |
| `src/rag/search/agent/graph.py` | Create | `build_agent_graph()` + `run_agent_pipeline()` wrapper |
| `scripts/run_agent.py` | Create | T2 CLI driver — REPL, `--compare`, `--questions` modes |
| `tests/manual_smoke_agent.py` | Create | Pure-logic assertions for `finalize()` mapping (no pytest needed) |

The existing `src/rag/search/pipeline.py`, `entrypoint.py`, `prompt.py`, `reranker.py`, `searxng_search.py`, `prep_cache.py`, `model.py` are **read-only reuse only** — no edits.

---

## Task 1: Scaffold the agent package & AgentState

**Files:**
- Create: `src/rag/search/agent/__init__.py`
- Create: `src/rag/search/agent/state.py`

- [ ] **Step 1: Create the empty package file**

`src/rag/search/agent/__init__.py`:

```python
"""LangGraph hybrid agent pipeline — parallel to src.rag.search.pipeline.

The agent replaces the semantic-router + if/else dispatch in pipeline.py
with an LLM tool-calling node, while keeping pre- and post-processing
deterministic. See docs/superpowers/specs/2026-04-25-langgraph-agent-search-design.md.
"""
```

- [ ] **Step 2: Write `AgentState` TypedDict**

`src/rag/search/agent/state.py`:

```python
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
```

- [ ] **Step 3: Verify the module imports cleanly**

Run:

```bash
python -c "from src.rag.search.agent.state import AgentState, make_initial_state; s = make_initial_state('hello', None); print(s['user_input'], s['tool_call_count'])"
```

Expected output: `hello 0`

- [ ] **Step 4: Commit**

```bash
git add src/rag/search/agent/__init__.py src/rag/search/agent/state.py
git commit -m "feat(agent): scaffold agent package + AgentState schema"
```

---

## Task 2: Agent dispatcher system prompt

**Files:**
- Create: `src/rag/search/agent/prompts.py`

- [ ] **Step 1: Write `AGENT_DISPATCHER_SYSTEM_PROMPT`**

`src/rag/search/agent/prompts.py`:

```python
AGENT_DISPATCHER_SYSTEM_PROMPT = """You are the dispatcher for **Insuripedia**, an LOMA281/LOMA291 insurance study assistant. You do NOT answer the user yourself — you choose which search tool to call. The system handles answer generation after your tools return data.

# Your tools (call exactly ONE per step; you may call up to 3 in total)

1. search_core_collection — textbook content (concepts, definitions, formulas, contract terms, regulations). DEFAULT choice for substantive insurance questions.
2. search_overall_collection — course METADATA only (module count, lesson list, syllabus structure). Use ONLY when the question is about course structure, NOT about insurance content itself.
3. search_web — public web fallback. Use ONLY when:
     a. You already called search_core and `found` was false, OR
     b. The question is time-sensitive (e.g., 'latest 2026 regulation update') and clearly outside the static textbook scope.
4. ask_clarification — terminate with a message to the user. Use when:
     a. Question is off-topic (not insurance/risk-management at all): reason="off_topic", message=warm rejection in the user's language.
     b. Question is too vague to search effectively even after rewrite: reason="vague", message=one warm rephrase prompt in the user's language.

# Decision flow

1. Is the question clearly insurance/risk-management related?
   * No → ask_clarification(reason="off_topic")
2. Is it about course STRUCTURE (modules, lessons, syllabus)?
   * Yes → search_overall_collection
3. Otherwise:
   * First try search_core_collection.
   * If `found` is false OR `preview` clearly does not address the query → search_web as a follow-up.
   * If you have good results → STOP calling tools (the system will finalize the answer from accumulated chunks).

# Hard rules

- NEVER call the same tool twice with the same query.
- NEVER call search_web before search_core unless the question is clearly time-sensitive and outside textbook scope.
- NEVER write the final user-facing answer yourself. Your role ends when you stop calling tools.
- The user's question is provided in {detected_language}. Tool `query` arguments must be in ENGLISH (the standalone_query has already been rewritten for you).
- The `reasoning` field on each tool call is for logging only — keep it to one short sentence.

# Context for this turn

- standalone_query (English): {standalone_query}
- detected_language: {detected_language}
- previous_tool_calls: {tool_history}
"""
```

- [ ] **Step 2: Verify the prompt formats without errors**

Run:

```bash
python -c "from src.rag.search.agent.prompts import AGENT_DISPATCHER_SYSTEM_PROMPT; print(AGENT_DISPATCHER_SYSTEM_PROMPT.format(detected_language='Vietnamese', standalone_query='What is endowment insurance?', tool_history='[]')[:200])"
```

Expected: prints the first 200 chars of the formatted prompt with no `KeyError`.

- [ ] **Step 3: Commit**

```bash
git add src/rag/search/agent/prompts.py
git commit -m "feat(agent): add dispatcher system prompt"
```

---

## Task 3: `search_core_collection` tool

**Files:**
- Create: `src/rag/search/agent/tools.py`

- [ ] **Step 1: Write the tool with state injection**

`src/rag/search/agent/tools.py`:

```python
"""LangChain @tool wrappers used by the agent dispatcher.

Each tool wraps an existing helper from src.rag.search.pipeline /
searxng_search and writes results back into AgentState via the Command
return type so reducers in StateGraph pick them up.
"""

from typing import Annotated, Any, Dict, List, Literal

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.constants.app_constant import (
    COLLECTION_NAME,
    CORE_VECTOR_TOP_K,
    OVERALL_COLLECTION_NAME,
    VECTOR_SEARCH_TOP_K,
)
from src.rag.db_vector import get_qdrant_client
from src.rag.llm.chat_llm import get_openai_chat_client
from src.rag.llm.embedding_llm import get_openai_embedding_client
from src.rag.search.agent.state import AgentState
from src.rag.search.model import ChunkDict
from src.rag.search.pipeline import _aembed_text, _ahyde_generate, _avector_search, extract_sources
from src.rag.search.searxng_search import web_rag_answer
from src.config.app_config import get_app_config
from src.utils.logger_utils import logger


def _chunks_preview(chunks: List[ChunkDict], n: int = 3, chars: int = 300) -> str:
    parts = []
    for c in chunks[:n]:
        text = (c.get("text") or "")[:chars].replace("\n", " ")
        parts.append(text)
    return " ||| ".join(parts) if parts else "(no chunks)"


@tool
async def search_core_collection(
    query: str,
    reasoning: str,
    state: Annotated[AgentState, InjectedState],
) -> Command:
    """Search the LOMA281/LOMA291 textbook chunks (core knowledge collection).
    Use for substantive insurance questions: concepts, definitions, formulas,
    contract terms, or any textbook content.

    Args:
        query: English search query (already rewritten as a standalone question).
        reasoning: One short sentence explaining why this tool was chosen (logged).
    """
    logger.info("[tool:search_core] query=%r reasoning=%r", query, reasoning)

    llm = get_openai_chat_client()
    embedder = get_openai_embedding_client()
    qdrant = get_qdrant_client(COLLECTION_NAME)

    try:
        hyde = await _ahyde_generate(llm, query)
        dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, hyde)
        chunks = await _avector_search(
            qdrant, dense_vec, colbert_vec, query, top_k=CORE_VECTOR_TOP_K,
        )
    except Exception as e:
        logger.exception("[tool:search_core] failed")
        return Command(update={
            "messages": [{
                "role": "tool",
                "content": f"search_core_collection error: {e}. Try a different tool.",
            }],
        })

    found = bool(chunks)
    preview = _chunks_preview(chunks)
    payload = {"found": found, "chunk_count": len(chunks), "preview": preview}

    return Command(update={
        "chunks": (state.get("chunks") or []) + chunks,
        "selected_collection": "core",
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "messages": [{"role": "tool", "content": str(payload)}],
    })
```

> **Implementer note:** LangGraph's `Command(update=...)` is how a tool both returns content to the LLM AND mutates state. The `messages` key uses LangGraph's `add_messages` reducer (declared in `AgentState`). The `tool_call_count` is incremented manually because there's no built-in counter.
>
> The `messages` content placed by the tool is what the LLM sees on its next turn — keep it short. The actual chunks live in `state["chunks"]` and are not shown to the LLM (we don't want it summarizing chunks itself).

- [ ] **Step 2: Verify the tool registers with `bind_tools`**

Run:

```bash
python -c "
from src.rag.search.agent.tools import search_core_collection
print('name:', search_core_collection.name)
print('description (first 80):', search_core_collection.description[:80])
print('args_schema fields:', list(search_core_collection.args_schema.model_fields.keys()) if search_core_collection.args_schema else 'none')
"
```

Expected: name = `search_core_collection`, args fields include `query` and `reasoning` but NOT `state` (LangGraph strips InjectedState from the LLM-visible schema).

- [ ] **Step 3: Commit**

```bash
git add src/rag/search/agent/tools.py
git commit -m "feat(agent): add search_core_collection tool"
```

---

## Task 4: `search_overall_collection` tool

**Files:**
- Modify: `src/rag/search/agent/tools.py` (append)

- [ ] **Step 1: Add tool to `tools.py`**

Append to `src/rag/search/agent/tools.py` (after `search_core_collection`):

```python
@tool
async def search_overall_collection(
    query: str,
    reasoning: str,
    state: Annotated[AgentState, InjectedState],
) -> Command:
    """Search course-level metadata (syllabus, module structure, lesson counts,
    course descriptions). Use ONLY for questions like 'how many modules in
    LOMA 281?', 'what does LOMA 291 cover?', 'list lessons in module 3'.

    Args:
        query: English search query.
        reasoning: One short sentence explaining the choice (logged).
    """
    logger.info("[tool:search_overall] query=%r reasoning=%r", query, reasoning)

    embedder = get_openai_embedding_client()
    qdrant = get_qdrant_client(OVERALL_COLLECTION_NAME)

    try:
        dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, query)
        chunks = await _avector_search(
            qdrant, dense_vec, colbert_vec, query, top_k=VECTOR_SEARCH_TOP_K,
        )
    except Exception as e:
        logger.exception("[tool:search_overall] failed")
        return Command(update={
            "messages": [{
                "role": "tool",
                "content": f"search_overall_collection error: {e}. Try a different tool.",
            }],
        })

    found = bool(chunks)
    preview = _chunks_preview(chunks)
    payload = {"found": found, "chunk_count": len(chunks), "preview": preview}

    return Command(update={
        "chunks": (state.get("chunks") or []) + chunks,
        "selected_collection": "overall",
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "messages": [{"role": "tool", "content": str(payload)}],
    })
```

- [ ] **Step 2: Verify the tool loads**

Run:

```bash
python -c "from src.rag.search.agent.tools import search_overall_collection; print(search_overall_collection.name)"
```

Expected: `search_overall_collection`

- [ ] **Step 3: Commit**

```bash
git add src/rag/search/agent/tools.py
git commit -m "feat(agent): add search_overall_collection tool"
```

---

## Task 5: `search_web` tool

**Files:**
- Modify: `src/rag/search/agent/tools.py` (append)

- [ ] **Step 1: Add tool to `tools.py`**

Append to `src/rag/search/agent/tools.py`:

```python
@tool
async def search_web(
    query: str,
    reasoning: str,
    state: Annotated[AgentState, InjectedState],
) -> Command:
    """Search the public web (SearXNG). Use only when:
    (1) you already searched the textbook collection and `found` was false, OR
    (2) the question is clearly time-sensitive (recent events, dates, news)
        and outside the LOMA textbook scope.

    Args:
        query: English search query.
        reasoning: One short sentence explaining the choice (logged).
    """
    logger.info("[tool:search_web] query=%r reasoning=%r", query, reasoning)

    llm = get_openai_chat_client()
    embedder = get_openai_embedding_client()
    detected_language = state.get("detected_language") or "English"

    try:
        rs = await web_rag_answer(llm, embedder, query, detected_language)
    except Exception as e:
        logger.exception("[tool:search_web] failed")
        return Command(update={
            "messages": [{
                "role": "tool",
                "content": f"search_web error: {e}",
            }],
        })

    answer = rs.get("answer") or ""
    sources = rs.get("sources") or []
    found = bool(answer)
    payload = {"found": found, "answer_preview": answer[:200]}

    return Command(update={
        "web_answer": answer,
        "sources": sources,
        "selected_collection": "web",
        "web_search_used": True,
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "messages": [{"role": "tool", "content": str(payload)}],
    })
```

- [ ] **Step 2: Verify**

Run:

```bash
python -c "from src.rag.search.agent.tools import search_web; print(search_web.name)"
```

Expected: `search_web`

- [ ] **Step 3: Commit**

```bash
git add src/rag/search/agent/tools.py
git commit -m "feat(agent): add search_web tool"
```

---

## Task 6: `ask_clarification` tool

**Files:**
- Modify: `src/rag/search/agent/tools.py` (append)

- [ ] **Step 1: Add tool to `tools.py`**

Append to `src/rag/search/agent/tools.py`:

```python
@tool
async def ask_clarification(
    reason: Literal["off_topic", "vague"],
    message: str,
    state: Annotated[AgentState, InjectedState],
) -> Command:
    """Terminate the loop with a message to the user. Use for off-topic
    questions or for questions too vague to search effectively.

    Args:
        reason: "off_topic" (not insurance-related) or "vague" (unclear).
        message: Text to show the user (must be in their language).
    """
    logger.info("[tool:ask_clarification] reason=%s", reason)

    return Command(update={
        "clarification": {"type": reason, "response": message},
        "tool_call_count": state.get("tool_call_count", 0) + 1,
        "messages": [{"role": "tool", "content": f"clarification_sent: {reason}"}],
    })


ALL_TOOLS = [
    search_core_collection,
    search_overall_collection,
    search_web,
    ask_clarification,
]
```

- [ ] **Step 2: Verify all 4 tools load and `ALL_TOOLS` is intact**

Run:

```bash
python -c "from src.rag.search.agent.tools import ALL_TOOLS; print(len(ALL_TOOLS), [t.name for t in ALL_TOOLS])"
```

Expected: `4 ['search_core_collection', 'search_overall_collection', 'search_web', 'ask_clarification']`

- [ ] **Step 3: Commit**

```bash
git add src/rag/search/agent/tools.py
git commit -m "feat(agent): add ask_clarification tool + ALL_TOOLS export"
```

---

## Task 7: Pre-processing nodes

**Files:**
- Create: `src/rag/search/agent/nodes.py`

- [ ] **Step 1: Write pre-processing nodes**

`src/rag/search/agent/nodes.py`:

```python
"""Deterministic StateGraph nodes — everything before and after the agent.

Pre-processing (this file, top half):
    validate_input  → length check
    detect_and_rewrite → asyncio.gather over LLM language-detect and history-aware rewrite
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
    """Runs language detection and standalone-query rewrite in parallel,
    matching `_avalidate_and_prepare` in pipeline.py."""
    llm = get_openai_chat_client()
    user_input = state["user_input"]
    conversation_id = state.get("conversation_id")
    has_history = bool(conversation_id and conversation_id.strip())

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
    }
```

- [ ] **Step 2: Verify the module imports**

Run:

```bash
python -c "from src.rag.search.agent.nodes import validate_input_node, detect_and_rewrite_node, quiz_check_node, clarity_check_node; print('pre-processing nodes loaded')"
```

Expected: `pre-processing nodes loaded`

- [ ] **Step 3: Commit**

```bash
git add src/rag/search/agent/nodes.py
git commit -m "feat(agent): add pre-processing nodes (validate/detect/rewrite/quiz/clarity)"
```

---

## Task 8: Post-processing nodes (rerank / enrich / generate / finalize)

**Files:**
- Modify: `src/rag/search/agent/nodes.py` (append)

- [ ] **Step 1: Append post-processing nodes to `nodes.py`**

Append to `src/rag/search/agent/nodes.py`:

```python
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
```

- [ ] **Step 2: Write smoke test for `finalize_node` mapping**

`tests/manual_smoke_agent.py`:

```python
"""Pure-logic smoke checks for the agent finalize() mapping.

Run: python tests/manual_smoke_agent.py
Exits 0 on success, raises AssertionError on failure.
"""

from src.constants.app_constant import (
    INTENT_CORE_KNOWLEDGE,
    INTENT_OFF_TOPIC,
    INTENT_OVERALL_COURSE_KNOWLEDGE,
    INTENT_QUIZ,
)
from src.rag.search.agent.nodes import finalize_node
from src.rag.search.agent.state import make_initial_state


def case(name, **state_overrides):
    s = make_initial_state("q")
    s.update(state_overrides)
    out = finalize_node(s)
    print(f"  [{name}] intent={out['intent']} satisfied={out['answer_satisfied']} web={out['web_search_used']} resp_len={len(out.get('response') or '')}")
    return out


def main():
    print("finalize_node smoke checks:")

    # 1. Input too long
    out = case("input_too_long",
               early_exit_reason="input_too_long",
               response="too long",
               intent=INTENT_OFF_TOPIC,
               detected_language="English")
    assert out["intent"] == INTENT_OFF_TOPIC
    assert out["answer_satisfied"] is False

    # 2. Unsupported language
    out = case("unsupported_language",
               early_exit_reason="unsupported_language",
               response="lang msg",
               intent=INTENT_OFF_TOPIC,
               detected_language="English")
    assert out["intent"] == INTENT_OFF_TOPIC

    # 3. Quiz
    out = case("quiz",
               early_exit_reason="quiz",
               response=None,
               intent=INTENT_QUIZ,
               detected_language="Vietnamese")
    assert out["intent"] == INTENT_QUIZ
    assert out["response"] is None

    # 4. Pre-processing clarity exit
    out = case("pre_clarity",
               early_exit_reason="clarification",
               response="please rephrase",
               intent=INTENT_CORE_KNOWLEDGE,
               detected_language="Vietnamese")
    assert out["intent"] == INTENT_CORE_KNOWLEDGE
    assert out["response"] == "please rephrase"

    # 5. Tool-driven clarification (off_topic)
    out = case("tool_off_topic",
               clarification={"type": "off_topic", "response": "we only do insurance"},
               detected_language="English")
    assert out["intent"] == INTENT_OFF_TOPIC
    assert "insurance" in out["response"]

    # 6. Tool-driven clarification (vague)
    out = case("tool_vague",
               clarification={"type": "vague", "response": "could you clarify?"},
               detected_language="English")
    assert out["intent"] == INTENT_CORE_KNOWLEDGE

    # 7. Web answer
    out = case("web_done",
               web_answer="web answer text",
               sources=[{"url": "x"}],
               detected_language="English")
    assert out["intent"] == INTENT_CORE_KNOWLEDGE
    assert out["web_search_used"] is True
    assert out["response"] == "web answer text"

    # 8. Core RAG with answer
    out = case("core_ok",
               selected_collection="core",
               response="core answer",
               chunks=[{"id": "c1", "text": "x", "metadata": {}, "score": 0.5}],
               sources=[],
               detected_language="English")
    assert out["intent"] == INTENT_CORE_KNOWLEDGE
    assert out["answer_satisfied"] is True

    # 9. Overall RAG with answer
    out = case("overall_ok",
               selected_collection="overall",
               response="overall answer",
               detected_language="English")
    assert out["intent"] == INTENT_OVERALL_COURSE_KNOWLEDGE
    assert out["answer_satisfied"] is True

    # 10. Overall with no chunks
    out = case("overall_empty",
               selected_collection="overall",
               response=None,
               detected_language="English")
    assert out["intent"] == INTENT_OVERALL_COURSE_KNOWLEDGE
    assert out["answer_satisfied"] is False

    # 11. No collection selected at all (agent gave up)
    out = case("agent_gave_up",
               selected_collection=None,
               detected_language="English")
    assert out["answer_satisfied"] is False

    print("\nAll 11 cases passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the smoke test**

Run:

```bash
python tests/manual_smoke_agent.py
```

Expected: prints 11 case lines and `All 11 cases passed.`

- [ ] **Step 4: Commit**

```bash
git add src/rag/search/agent/nodes.py tests/manual_smoke_agent.py
git commit -m "feat(agent): add post-processing nodes + finalize smoke test"
```

---

## Task 9: Agent decision node (LLM tool-calling)

**Files:**
- Create: `src/rag/search/agent/agent_node.py`

- [ ] **Step 1: Write the agent node + AzureChatOpenAI factory**

`src/rag/search/agent/agent_node.py`:

```python
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
_llm_lock = threading.Lock()


def get_agent_llm() -> AzureChatOpenAI:
    global _llm_instance
    if _llm_instance is None:
        with _llm_lock:
            if _llm_instance is None:
                _llm_instance = AzureChatOpenAI(
                    azure_endpoint=config.OPENAI_API_BASE,
                    api_key=config.OPENAI_API_KEY,
                    api_version=config.OPENAI_API_VERSION,
                    azure_deployment=config.OPENAI_CHAT_MODEL,
                    temperature=0.0,
                    timeout=config.GPT_TIMEOUT,
                    max_retries=config.GPT_MAX_RETRIES,
                )
    return _llm_instance


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
    llm = get_agent_llm().bind_tools(ALL_TOOLS)

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
```

- [ ] **Step 2: Verify the LLM factory & node import**

Run:

```bash
python -c "
from src.rag.search.agent.agent_node import get_agent_llm, agent_decide_node
llm = get_agent_llm()
print('LLM type:', type(llm).__name__)
print('agent_decide_node:', agent_decide_node.__name__)
"
```

Expected:

```
LLM type: AzureChatOpenAI
agent_decide_node: agent_decide_node
```

- [ ] **Step 3: Commit**

```bash
git add src/rag/search/agent/agent_node.py
git commit -m "feat(agent): add agent_decide node + AzureChatOpenAI factory"
```

---

## Task 10: Graph assembly

**Files:**
- Create: `src/rag/search/agent/graph.py`

- [ ] **Step 1: Write `build_agent_graph()` + `run_agent_pipeline()`**

`src/rag/search/agent/graph.py`:

```python
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
```

- [ ] **Step 2: Verify the graph compiles**

Run:

```bash
python -c "
from src.rag.search.agent.graph import build_agent_graph
g = build_agent_graph()
print('compiled OK, nodes:', list(g.get_graph().nodes.keys()))
"
```

Expected: prints `compiled OK, nodes: ['__start__', 'validate_input', 'detect_and_rewrite', 'quiz_check', 'clarity_check', 'agent_decide', 'tools', 'rerank', 'enrich', 'generate', 'finalize', 'post_router', '__end__']` (order may vary).

- [ ] **Step 3: Re-export from `__init__.py`**

Edit `src/rag/search/agent/__init__.py` to add:

```python
from src.rag.search.agent.graph import build_agent_graph, get_compiled_graph, run_agent_pipeline

__all__ = ["build_agent_graph", "get_compiled_graph", "run_agent_pipeline"]
```

(Keep the existing module docstring above it.)

- [ ] **Step 4: Verify the public API**

Run:

```bash
python -c "from src.rag.search.agent import run_agent_pipeline, build_agent_graph; print('public API OK')"
```

Expected: `public API OK`

- [ ] **Step 5: Commit**

```bash
git add src/rag/search/agent/graph.py src/rag/search/agent/__init__.py
git commit -m "feat(agent): assemble StateGraph + run_agent_pipeline wrapper"
```

---

## Task 11: T2 CLI driver — REPL mode

**Files:**
- Create: `scripts/run_agent.py`

- [ ] **Step 1: Write the basic REPL**

`scripts/run_agent.py`:

```python
"""T2 CLI driver for the LangGraph hybrid agent pipeline.

Modes:
    python scripts/run_agent.py                       # interactive REPL
    python scripts/run_agent.py --compare             # REPL, runs both old + new pipelines side-by-side
    python scripts/run_agent.py --questions FILE      # batch over one-question-per-line file → CSV stdout

Run with PYTHONPATH set to the project root, e.g.:
    PYTHONPATH=. python scripts/run_agent.py
"""

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.db_vector import get_qdrant_client
from src.rag.llm.chat_llm import get_openai_chat_client
from src.rag.llm.embedding_llm import get_async_client, get_openai_embedding_client, get_sync_client
from src.rag.search.agent import run_agent_pipeline
from src.rag.search.pipeline import async_pipeline_dispatch
from src.rag.search.reranker import _get_reranker


async def warm_up():
    """Mirror main.py lifespan warm-up (minus FastAPI router pieces)."""
    qdrant = get_qdrant_client()
    try:
        qdrant.client.get_collections()
    except Exception:
        pass
    get_sync_client()
    get_async_client()
    get_openai_chat_client()
    get_openai_embedding_client()
    # Warm reranker in a thread so first agent call doesn't pay model-load cost.
    await asyncio.get_running_loop().run_in_executor(None, _get_reranker)


def print_result(label: str, result: dict, latency: float):
    print(f"\n--- {label} ({latency:.2f}s) ---")
    print(f"  intent           : {result.get('intent')}")
    print(f"  detected_language: {result.get('detected_language')}")
    print(f"  answer_satisfied : {result.get('answer_satisfied')}")
    print(f"  web_search_used  : {result.get('web_search_used')}")
    print(f"  sources          : {len(result.get('sources') or [])}")
    response = result.get("response") or ""
    print(f"  response ({len(response)} chars):")
    print("  " + (response.replace("\n", "\n  ") if response else "(none)"))


async def run_one_agent(question: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    result = await run_agent_pipeline(question)
    return result, time.perf_counter() - t0


async def run_one_old(question: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    result = await async_pipeline_dispatch(question)
    return dict(result), time.perf_counter() - t0


async def repl_loop(compare: bool):
    print("Insuripedia agent REPL. Type a question (Ctrl-C / Ctrl-D to exit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
        if not question:
            continue

        if compare:
            (agent_res, agent_t), (old_res, old_t) = await asyncio.gather(
                run_one_agent(question), run_one_old(question),
            )
            print_result("AGENT", agent_res, agent_t)
            print_result("OLD  ", old_res, old_t)
        else:
            agent_res, agent_t = await run_one_agent(question)
            print_result("AGENT", agent_res, agent_t)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Also run pipeline.py for side-by-side comparison.")
    parser.add_argument("--questions", type=str, default=None, help="Path to a file with one question per line; emit CSV summary.")
    args = parser.parse_args()

    await warm_up()

    if args.questions:
        # Implemented in Task 13.
        print("--questions mode not implemented yet (Task 13).")
        return

    await repl_loop(compare=args.compare)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Smoke-run the REPL with one question**

Run (provide one question and Ctrl-D / `exit` to leave):

```bash
PYTHONPATH=. python scripts/run_agent.py
```

In the prompt, type something like `What is the law of large numbers in insurance?` and press Enter. Expect:
- Pre-processing logs.
- One or more `[tool:search_*]` log lines.
- A `--- AGENT (X.XXs) ---` block with non-empty response.

Then exit with Ctrl-D.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_agent.py
git commit -m "feat(agent): add T2 CLI REPL driver"
```

---

## Task 12: T2 CLI — `--compare` mode verification

**Files:**
- Modify (verification only): `scripts/run_agent.py` already supports `--compare` from Task 11.

- [ ] **Step 1: Smoke-run the compare mode**

Run:

```bash
PYTHONPATH=. python scripts/run_agent.py --compare
```

Type 3 different question types and verify both panels print:

1. Substantive insurance: `What is adverse selection?` → both should hit the core knowledge path.
2. Course structure: `How many modules are in LOMA 281?` → AGENT should pick `search_overall_collection`; OLD should route to `INTENT_OVERALL_COURSE_KNOWLEDGE`.
3. Off-topic: `What is the weather in Tokyo today?` → AGENT should call `ask_clarification(off_topic)` or fall back to web; OLD should return `INTENT_OFF_TOPIC`.

Compare latency and intent column. Note any cases where the two diverge — these are the candidates for prompt-tuning later.

- [ ] **Step 2: No code change needed; document any prompt issues found**

If the agent picks the wrong tool repeatedly on a category of questions, note it in a follow-up issue or in `prompts.py` as a `# TODO(prompt-tuning)` comment with the failing example. Do NOT change the prompt in this plan — that's a separate iteration.

- [ ] **Step 3: Commit (only if you actually edited a comment in `prompts.py`)**

If no edits, skip the commit. Otherwise:

```bash
git add src/rag/search/agent/prompts.py
git commit -m "docs(agent): note prompt-tuning candidates from compare run"
```

---

## Task 13: T2 CLI — `--questions` batch mode

**Files:**
- Modify: `scripts/run_agent.py` (replace the placeholder branch)

- [ ] **Step 1: Replace the placeholder in `main()` with batch logic**

In `scripts/run_agent.py`, find this block:

```python
    if args.questions:
        # Implemented in Task 13.
        print("--questions mode not implemented yet (Task 13).")
        return
```

Replace it with:

```python
    if args.questions:
        await run_batch(args.questions)
        return
```

And add the `run_batch` function above `main()`:

```python
async def run_batch(path: str):
    """Read one question per line, run both pipelines, emit CSV to stdout."""
    questions = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not questions:
        print("No questions found.", file=sys.stderr)
        return

    writer = csv.writer(sys.stdout)
    writer.writerow([
        "question",
        "agent_intent", "agent_satisfied", "agent_web", "agent_sources", "agent_resp_len", "agent_latency_s",
        "old_intent",   "old_satisfied",   "old_web",   "old_sources",   "old_resp_len",   "old_latency_s",
    ])

    for q in questions:
        (agent_res, agent_t), (old_res, old_t) = await asyncio.gather(
            run_one_agent(q), run_one_old(q),
        )
        writer.writerow([
            q,
            agent_res.get("intent"),
            agent_res.get("answer_satisfied"),
            agent_res.get("web_search_used"),
            len(agent_res.get("sources") or []),
            len(agent_res.get("response") or ""),
            f"{agent_t:.2f}",
            old_res.get("intent"),
            old_res.get("answer_satisfied"),
            old_res.get("web_search_used"),
            len(old_res.get("sources") or []),
            len(old_res.get("response") or ""),
            f"{old_t:.2f}",
        ])
        sys.stdout.flush()
```

- [ ] **Step 2: Smoke-run the batch mode**

Create a small `/tmp/sample_questions.txt` (or `data/sample_questions.txt`):

```
What is the law of large numbers?
How many modules are in LOMA 281?
What is the weather today?
```

Run:

```bash
PYTHONPATH=. python scripts/run_agent.py --questions data/sample_questions.txt
```

Expected: 1 CSV header line + 3 data rows on stdout. Latency columns should be > 0.5s. No tracebacks.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_agent.py
git commit -m "feat(agent): add --questions batch CSV mode to CLI"
```

---

## Done — verification checklist

After Task 13:

- [ ] `python tests/manual_smoke_agent.py` passes (11 cases).
- [ ] `PYTHONPATH=. python scripts/run_agent.py` REPL answers a substantive question end-to-end.
- [ ] `PYTHONPATH=. python scripts/run_agent.py --compare` shows AGENT and OLD side-by-side, both producing answers.
- [ ] `PYTHONPATH=. python scripts/run_agent.py --questions <file>` emits valid CSV.
- [ ] `git log --oneline` shows 11 commits added by this plan, none touching `pipeline.py` / `entrypoint.py` / `prompt.py` / `searxng_search.py` / `reranker.py` / `model.py`.

After all checks pass, the spec's Phase-1 (T2) deliverables are complete. Phase 2 (T1 FastAPI streaming route) is deferred to a separate plan as documented in §12 of the spec.
