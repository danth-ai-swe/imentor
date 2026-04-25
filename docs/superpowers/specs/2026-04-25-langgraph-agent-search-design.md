# LangGraph Hybrid Agent for RAG Search — Design

**Date:** 2026-04-25
**Status:** Spec — pending implementation
**Scope:** New parallel pipeline. The existing `src/rag/search/pipeline.py` is **not** modified. The new code lives in `src/rag/search/agent/` and a CLI driver in `scripts/run_agent.py`. Once stable, it will be wired into FastAPI as a second engine option.

## 1. Goal & Non-Goals

### Goal
Build a LangGraph-based pipeline that mirrors the routing decisions of `pipeline.py` (when to RAG-search the core collection, when to search the overall collection, when to fall back to web, when to ask for clarification, when to reject for unsupported language) but replaces the deterministic semantic-router + `if/else` dispatch with an **LLM tool-calling agent** for the search-selection step. Pre-processing and post-processing remain deterministic.

The new pipeline must produce the same `PipelineResult` shape as the old one, so we can A/B compare them on the same input set.

### Non-Goals
- Not removing or refactoring `pipeline.py`. It stays as the production engine.
- Not changing the existing FastAPI routes in phase 1.
- Not changing prompt templates already in `src/rag/search/prompt.py` (`CLARITY_CHECK_PROMPT`, `HYDE_PROMPT`, `DETECT_LANGUAGE_PROMPT`, `SYSTEM_PROMPT_TEMPLATE`, `SUMMARIZE_PROMPT_TEMPLATE`). The agent reuses them as-is.
- Not adding new third-party dependencies. `langgraph` and `langchain-openai` are already in `requirements.txt`.

## 2. Architectural Style — B1 Hybrid

The pipeline is mostly deterministic with **one LLM tool-calling agent node** in the middle, acting as a search dispatcher.

- **Pre-processing (deterministic):** validate input length → detect language + rewrite query (in parallel) → quiz keyword check → clarity check.
- **Agent (LLM tool-calling, multi-step ReAct):** picks 1–3 tools from `{search_core_collection, search_overall_collection, search_web, ask_clarification}`.
- **Post-processing (deterministic):** rerank → enrich (neighbor chunks) → generate final answer. Skipped when the agent terminated via `search_web` (which already returns a full answer) or `ask_clarification`.

Rationale: keep the well-tuned rerank/enrich/generate path untouched, but replace the routing brain (semantic router + `if/else`) with an LLM that can self-correct (e.g., search core, see no useful chunks, fall back to web within the same turn).

## 3. Module Layout

```
src/rag/search/
├── pipeline.py              # unchanged — production engine
├── entrypoint.py            # unchanged — reused
├── prompt.py                # unchanged — reused
├── reranker.py              # unchanged
├── searxng_search.py        # unchanged — wrapped by search_web tool
├── prep_cache.py            # unchanged
├── model.py                 # unchanged — reused (PipelineResult, ChunkDict)
└── agent/                   # NEW
    ├── __init__.py
    ├── state.py             # AgentState TypedDict
    ├── nodes.py             # deterministic nodes (validate, detect_lang, rewrite, quiz, clarity, rerank, enrich, generate, finalize)
    ├── tools.py             # 4 @tool definitions (with InjectedState)
    ├── agent_node.py        # LLM tool-calling node (AzureChatOpenAI.bind_tools)
    ├── graph.py             # build_agent_graph() — assembles StateGraph
    └── prompts.py           # AGENT_DISPATCHER_SYSTEM_PROMPT

scripts/
└── run_agent.py             # NEW — T2 CLI driver
```

Deleting `src/rag/search/agent/` reverts the project to the unchanged pipeline.

## 4. Graph Topology

```
┌─────────────────┐
│ validate_input  │  (length check)
└────────┬────────┘
         │ ok                          fail → finalize(input_too_long)
         │
┌────────▼────────────────────┐
│ detect_lang + rewrite_query │  (asyncio.gather inside one node)
└────────┬────────────────────┘
         │ language detected            "" → finalize(unsupported_language)
         │
┌────────▼────────┐
│ quiz_check      │
└────────┬────────┘
         │ not quiz                     quiz → finalize(quiz)
         │
┌────────▼────────┐
│ clarity_check   │
└────────┬────────┘
         │ clear                        not clear → finalize(clarification)
         │
┌────────▼─────────┐  ◄────────────────────┐
│  AGENT (LLM      │                       │
│  tool-calling)   │                       │
└────────┬─────────┘                       │
         │                                  │
         ├── tool_calls present → execute → loop back to AGENT (≤ 3 total)
         │
         │ no tool_calls / forced exit
         │
   ┌─────┴────────┐
   │ post_router  │
   └──┬───┬───┬───┘
      │   │   │
      │   │   └─ web_answer set → finalize(web)
      │   │
      │   └─ clarification set → finalize(clarification)
      │
      └─ chunks present
            │
            ├─ collection=="core" → rerank → enrich → generate → finalize
            └─ collection=="overall" → generate (no rerank) → finalize

      └─ no chunks → finalize(no_result)
```

**Termination rules:**
- `search_web` returning a full answer → END (skip rerank/enrich/generate).
- `ask_clarification` → END.
- Agent stops calling tools voluntarily → if state has chunks, run post-processing; else `NO_RESULT_RESPONSE_MAP[lang]`.
- `tool_call_count >= 3` → forced exit; same logic as voluntary stop.

## 5. State Schema

```python
class AgentState(TypedDict):
    # Input
    user_input: str
    conversation_id: Optional[str]

    # Pre-processing outputs
    detected_language: Optional[str]
    standalone_query: Optional[str]

    # Agent loop
    messages: Annotated[list[BaseMessage], add_messages]
    tool_call_count: int

    # Search outputs (set by tools)
    chunks: List[ChunkDict]
    sources: List[Any]
    web_answer: Optional[str]
    selected_collection: Optional[str]   # "core" | "overall" | "web"
    clarification: Optional[dict]        # {"type": "off_topic"|"vague", "response": str}

    # Final
    response: Optional[str]
    answer_satisfied: bool
    web_search_used: bool
    intent: Optional[str]                # for PipelineResult parity

    # Termination signal
    early_exit_reason: Optional[str]     # "input_too_long"|"unsupported_language"|"quiz"|"clarification"|"web_done"|"no_result"|"forced_exit"
```

## 6. Tools

Each tool is a LangChain `@tool` with a Pydantic-validated input schema. State is injected via an additional `state: Annotated[AgentState, InjectedState]` parameter (LangGraph 0.2+ pattern); LangGraph strips this parameter from the schema sent to the LLM, so the agent only sees the explicit `query`/`reasoning`/etc. fields below.

### 6.1 `search_core_collection(query: str, reasoning: str) -> dict`

Reuses `_ahyde_generate` → `_aembed_text` → `_avector_search` against `COLLECTION_NAME` with `top_k=CORE_VECTOR_TOP_K`.

Returns `{"chunks": [...], "found": bool, "preview": "<first 3 chunk excerpts, 300 chars each>"}`.

Side effects on state: appends to `chunks`, sets `selected_collection="core"`.

### 6.2 `search_overall_collection(query: str, reasoning: str) -> dict`

Skips HyDE. Embeds `query` directly, searches `OVERALL_COLLECTION_NAME` with `top_k=VECTOR_SEARCH_TOP_K`.

Returns same shape as 6.1.

Side effects: appends to `chunks`, sets `selected_collection="overall"`.

### 6.3 `search_web(query: str, reasoning: str) -> dict`

Calls `web_rag_answer(llm, embedder, query, detected_language)`. This already produces a final user-facing answer.

Returns `{"answer": str, "sources": [...], "found": bool}`.

Side effects: sets `web_answer`, `sources`, `web_search_used=True`, `selected_collection="web"`. Routes to END after agent stops.

### 6.4 `ask_clarification(reason: Literal["off_topic","vague"], message: str) -> dict`

No external call. The `message` argument is the literal text shown to the user (must be in the user's language; the system prompt instructs the agent on this).

Returns `{"clarification_sent": True}`.

Side effects: sets `clarification={"type": reason, "response": message}`. Routes to END after agent stops.

## 7. Agent System Prompt (skeleton)

The system prompt instructs the LLM to:

1. Call exactly one tool per step, up to 3 total.
2. Default to `search_core_collection` for substantive insurance questions.
3. Use `search_overall_collection` ONLY for course-structure questions (modules, lessons, syllabus).
4. Use `search_web` only as a fallback after `search_core` returns nothing useful, or for clearly time-sensitive questions outside textbook scope.
5. Use `ask_clarification` for off-topic or unsalvageably vague questions.
6. Never call the same tool twice with the same query.
7. Never write the final user-facing answer (the system handles that).
8. Tool `query` arguments must be in English (the rewritten standalone query is provided).
9. The `reasoning` field on each tool call is for logging only.

The full prompt is maintained in `src/rag/search/agent/prompts.py` as `AGENT_DISPATCHER_SYSTEM_PROMPT`.

## 8. PipelineResult Mapping

`finalize` node converts `AgentState` to `PipelineResult` so the new engine is a drop-in replacement at the API layer.

| Path | intent | answer_satisfied | web_search_used | response source |
|---|---|---|---|---|
| `ask_clarification(off_topic)` | `INTENT_OFF_TOPIC` | False | False | `message` arg |
| `ask_clarification(vague)` | `INTENT_CORE_KNOWLEDGE` | False | False | `message` arg |
| `search_web` success | `INTENT_CORE_KNOWLEDGE` | True | True | `web_rag_answer.answer` |
| `search_core` → generate | `INTENT_CORE_KNOWLEDGE` | True | False | LLM `ainvoke_creative` |
| `search_overall` → generate | `INTENT_OVERALL_COURSE_KNOWLEDGE` | True | False | LLM `ainvoke_creative` |
| No chunks anywhere | `INTENT_CORE_KNOWLEDGE` | False | False | `NO_RESULT_RESPONSE_MAP[lang]` |
| Input too long | `INTENT_OFF_TOPIC` | False | False | `INPUT_TOO_LONG_RESPONSE` |
| Unsupported language | `INTENT_OFF_TOPIC` | False | False | `UNSUPPORTED_LANGUAGE_MSG` |
| Quiz | `INTENT_QUIZ` | False | False | `None` (consumer handles) |

## 9. Error Handling

| Condition | Handling |
|---|---|
| LLM tool_call with unknown tool name | Append validation error to `messages`, loop back to agent. After 1 retry attempt, force exit with `NO_RESULT_RESPONSE_MAP[lang]`. |
| LLM tool_call with malformed args (Pydantic fail) | Same as above — feed validation error back, retry once. |
| Tool raises exception (Qdrant timeout, embed fail, web crawl fail) | Caught inside the tool; returns `{"found": False, "error": str(e)}`. Agent can switch to a different tool on the next step. |
| Agent stops without calling any tool and no chunks accumulated | Treat as voluntary stop with no result → `NO_RESULT_RESPONSE_MAP[lang]`. |
| `tool_call_count >= 3` | Force exit; if chunks exist run rerank/generate, else `NO_RESULT_RESPONSE_MAP[lang]`. |
| Unhandled exception in graph execution | Top-level `try/except` in the graph wrapper returns `_make_off_topic_result(generic_error_msg, detected_language)`. |
| Conversation history | `rewrite_query` node calls `afetch_chat_history(llm, conversation_id)` + `Reflection.areflect`, identical to `_avalidate_and_prepare` in `pipeline.py`. |

## 10. Logging & Observability

Each node and each tool call is wrapped with `StepTimer` (already in `src/utils/logger_utils.py`). Additional log fields:

- `agent_decisions=[search_core, search_web]` — sequence of tool names called.
- `agent_reasoning=[...]` — the `reasoning` string each tool call provided.
- `tool_call_count` at exit.

This makes A/B comparison with `pipeline.py` straightforward (we already log `intent_router score/intent` there).

## 11. LLM for the Agent Node

Uses `langchain_openai.AzureChatOpenAI` (already in dependencies) constructed with the same Azure endpoint/key/model as `AzureChatClient`, `temperature=0`. This client is constructed once and reused across calls (singleton pattern, parallel to `get_openai_chat_client()`).

The deterministic nodes (clarity, hyde, rewrite, generate) keep using the existing `AzureChatClient` so we don't fragment the codebase.

## 12. Test Strategy

### Phase 1 — T2 (CLI driver)

`scripts/run_agent.py`:

- Warm-up: Qdrant, embedder, chat client, reranker (no semantic router needed — the agent replaces it).
- Interactive REPL: read line, run agent graph, print events as they happen.
- Per-turn output:

  ```
  [user] bảo hiểm sinh kỳ là gì?
  [pre]  detected_language=Vietnamese  standalone_query="What is endowment insurance?"
  [agent] tool: search_core_collection(query="endowment insurance definition", reasoning="…")
  [tool]  search_core → found=True, 5 chunks
  [agent] no more tool calls
  [post]  rerank → 3 chunks → enrich → 5 chunks → generate
  [answer] Bảo hiểm sinh kỳ (endowment) là …
  [meta]  intent=core_knowledge  web_used=False  tools_called=1  latency=3.2s
  ```

- Flag `--compare`: runs both the old pipeline and the new agent on the same input, prints results side-by-side.
- Flag `--questions <file>`: batch mode — runs every line from the file, writes a CSV with `(question, intent, latency, tools_called, sources_count, answer_length)` rows for both engines. Used for regression / quality comparison.

### Phase 2 — T1 (FastAPI route)

Once T2 confirms behavior, add a new route `POST /api/v1/chat/agent/stream` parallel to the existing chat route (the choice between a separate route vs. a `?engine=agent` query param is itself a Phase 2 design call and not part of this spec). Streaming uses `graph.astream(state, stream_mode="updates")`; events are converted to the existing `{type: meta|delta|done}` shape used by `async_pipeline_dispatch_stream`, so no FE change is needed. The `generate` node uses `llm.astream_creative` to stream tokens.

Phase 2 is out of scope for this initial implementation but the design is future-proof for it.

## 13. Dependencies

- `langgraph >= 0.2` (required for `InjectedState`). The current `requirements.txt` does not pin a version; the implementation step verifies the installed version and bumps if needed.
- `langchain-openai` (already present) for `AzureChatOpenAI` with `bind_tools`.
- No other new dependencies.

## 14. Out-of-Scope / Deferred

- Multi-collection parallel search (e.g., search core + web in the same step). Possible on top of this design but not built in v1.
- Streaming support and FastAPI wiring (Phase 2).
- Replacing the semantic router globally — the existing pipeline still uses it.
- Changing rerank/enrich/generate behavior.
- Adding tool-call traces to Langfuse spans (existing logger covers basic timing; deeper tracing can come later).
