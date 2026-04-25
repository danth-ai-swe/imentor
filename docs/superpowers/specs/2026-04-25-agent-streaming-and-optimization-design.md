# Agent Streaming Endpoint + Latency Optimization — Design

**Date:** 2026-04-25
**Status:** Spec — pending implementation
**Builds on:** `docs/superpowers/specs/2026-04-25-langgraph-agent-search-design.md` (Phase 2 §12 referred Phase 1 streaming and FastAPI wiring as deferred work — this spec executes that).

## 1. Goal & Non-Goals

### Goal

Three coordinated changes to the LangGraph hybrid agent pipeline that ships a usable, comparable, faster agent over HTTP:

1. **Streaming SSE endpoint** at `POST /api/v1/chat/ask/stream/agent`, parallel to the existing `POST /api/v1/chat/ask/stream`. Same SSE event shape (`meta` / `delta` / `done` / `error`) so the FE needs no changes when switching engines.
2. **Two latency optimizations** with realistic gains:
   - Skip `clarity_check_node` (the dispatcher prompt already handles off-topic and vague queries via `ask_clarification`).
   - Use a separate, smaller Azure deployment for the dispatcher LLM (e.g. `gpt-4o-mini`) while keeping the answer-generation LLM on the production model (e.g. `gpt-4o`).
3. **A 10-case benchmark fixture** that exercises every path of the agent graph and is reusable for regression and latency comparison.

Combined realistic target for substantive questions: **agent total latency drops from ~10s to ~7-8s** (saves ≈ 1 s from skipping `clarity_check_node` plus ≈ 1-1.5 s from the smaller dispatcher model). That puts the agent on par with the existing pipeline (~8.7 s on the same query) while preserving the multi-step ReAct flexibility.

### Non-Goals

- No changes to `pipeline.py`, `entrypoint.py`, `prompt.py`, `searxng_search.py`, `reranker.py`, `model.py`, or any of `tools.py`/`prompts.py`/`state.py`/`nodes.py` beyond what the optimizations explicitly require.
- No speculative parallelism (e.g. running HyDE in parallel with the dispatcher LLM). That option was discussed during brainstorming and explicitly deferred to a follow-up spec if benchmarks show it is still needed.
- No changes to the existing `/ask` and `/ask/stream` routes. They keep using the production pipeline.
- No new framework dependencies. The existing `langgraph`, `langchain-openai`, and `fastapi` install set covers everything.

## 2. Streaming Endpoint

### 2.1 Route

`POST /api/v1/chat/ask/stream/agent`

Request body matches the existing `ChatRequest`:

```json
{ "user_name": "...", "message": "...", "conversation_id": "..." }
```

Response is `text/event-stream` with `Cache-Control: no-cache` and `X-Accel-Buffering: no`, identical to the existing `/ask/stream` route.

### 2.2 Event shape

The four SSE events match `async_pipeline_dispatch_stream`'s shape for FE parity, with one additive field on `meta`:

| Event | Payload |
|---|---|
| `meta` | `{intent, detected_language, sources, answer_satisfied, web_search_used, tool_call_count}` |
| `delta` | `{content: "<token piece>"}` (one or many; concatenation = full answer) |
| `done` | `{}` |
| `error` | `{message: "<exc str>"}` |

`tool_call_count` is the only new field. It's optional from the FE's perspective (additive, not breaking).

### 2.3 Stream source

`graph.astream(state, stream_mode="updates")` emits one dict per node completion. The new module `src/rag/search/agent/streaming.py` defines `async def agent_stream_events(user_input, conversation_id) -> AsyncGenerator[Dict, None]` that translates LangGraph updates into the SSE event shape above.

Rules:

- After every pre-processing node, inspect `early_exit_reason`. If set (`input_too_long`, `unsupported_language`, `quiz`, `clarification`), short-circuit to a `meta + delta + done` triple carrying the prebuilt response (mirrors `_full_reply_events` in `pipeline.py`).
- After the agent loop terminates, inspect post-router decision:
   - `clarification`: `meta + delta(clarification.response) + done`.
   - `web_answer`: `meta + delta(web_answer) + done`.
   - `core` path: emit `meta` once `extract_sources` is known (after rerank/enrich), then forward `llm.astream_creative` chunks as `delta` events, then `done`.
   - `overall` path: same as core, but skip rerank/enrich (matches the deterministic graph).
   - No usable result: `meta + delta(NO_RESULT_RESPONSE_MAP[lang]) + done`.
- The streaming generator itself runs inside a top-level `try/except` in the route handler that emits `event: error` and exits cleanly on any uncaught exception.

### 2.4 Token streaming for `generate_node`

`graph.astream(stream_mode="updates")` streams node-completion events but not LLM token chunks. For real token-level streaming inside `generate_node`, the streaming module bypasses the graph's `generate_node` invocation and runs `llm.astream_creative(prompt)` directly after consuming the graph's pre-`generate` updates. The graph still runs end-to-end via `astream` for state correctness, but the final answer node is delegated to a dedicated streaming call so chunks reach the client live.

This dual-path approach mirrors how `async_pipeline_dispatch_stream` runs the deterministic graph then calls `_astream_answer` for the final LLM. The trade-off is one extra prompt build, which is cheap.

## 3. Optimization A — Drop `clarity_check_node`

### 3.1 What changes

Remove `clarity_check_node` from the graph in `src/rag/search/agent/graph.py`. Conditional edge after `quiz_check` routes directly to `agent_decide` (or `finalize` on early-exit).

The function `clarity_check_node` itself stays in `nodes.py` (no need to delete; it's no longer wired in). Removing the node from the builder is a one-edge rewire.

### 3.2 Why this is safe

The dispatcher's system prompt (`src/rag/search/agent/prompts.py`, finalised in commit `fe36b04`) already contains explicit rules:

- "Question off-topic → ask_clarification(reason='off_topic')"
- "Question vague → ask_clarification(reason='vague')"

This is the same functionality `clarity_check_node` provides via `_acheck_input_clarity`. Removing the standalone node eliminates one LLM call (~700-1000ms cold path) and trusts the dispatcher to handle the same cases.

### 3.3 Risk

The standalone clarity prompt includes a "Knowledge Scope" section listing 100+ LOMA topics (see `CLARITY_CHECK_PROMPT` in `prompt.py`). The dispatcher's prompt does not include this list. The dispatcher might be marginally less accurate at detecting off-topic questions outside obviously off-topic categories.

Test cases 5 and 6 in §5 directly exercise this. The acceptance criteria require both to terminate via `ask_clarification` (case 5) and `ask_clarification(rephrase)` (case 6). If they fail, the rollback is to add the Knowledge Scope list to the dispatcher prompt before re-running the benchmark.

### 3.4 Save

~1 LLM call eliminated per request → ~700-1000ms saved on cold path. Higher when prompt cache misses.

## 4. Optimization C — Dedicated dispatcher deployment

### 4.1 Config addition

`src/config/app_config.py` adds one optional field:

```python
OPENAI_DISPATCH_MODEL: Optional[str] = None   # falls back to OPENAI_CHAT_MODEL if unset
```

The user's `.env` is expected to set `OPENAI_DISPATCH_MODEL=gpt-4o-mini` (same Azure endpoint, key, and API version as the existing chat deployment). When unset, behaviour is unchanged from today: dispatcher uses `OPENAI_CHAT_MODEL`.

### 4.2 Wiring

`src/rag/search/agent/agent_node.py` `_get_bound_llm`:

```python
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
```

The dispatcher needs only enough output for `tool_calls + reasoning` strings, so `max_tokens=512` is a safe cap (the default `GPT_MAX_TOKENS=3500` is wasteful). `_get_bound_llm` is the only consumer; `get_agent_llm` is unaffected.

### 4.3 Save

`gpt-4o-mini` is roughly 2-3× faster than `gpt-4o` in TPS and has lower TTFT. Per-dispatcher-call savings: ~1-1.5s (from ~2s down to ~0.7-1s). With one dispatcher call per typical request, that's a flat ~1-1.5s win.

### 4.4 Risk

A small model can mis-classify edge cases (off-topic vs vague vs substantive). The 10-case fixture exercises this; the acceptance criteria below catch regressions.

## 5. Test Fixture

### 5.1 File

`data/agent_test_questions.txt` (replaces `data/agent_smoke_questions.txt` from the previous spec):

```
# === Substantive insurance ===
What is adverse selection in insurance?
Bảo hiểm sinh kỳ là gì và khác bảo hiểm tử kỳ thế nào?
保険における逆選択とは何ですか

# === Course structure (overall collection) ===
How many modules are in LOMA 281?

# === Off-topic (should trigger ask_clarification) ===
What's the weather in Tokyo today?

# === Vague (should trigger ask_clarification rephrase) ===
tell me the thing about that risk stuff

# === Time-sensitive / outside textbook (search_web) ===
What is the latest 2026 update to U.S. life insurance regulation?

# === Quiz keyword (early-exit pre-processing) ===
generate a quiz on annuities

# === Unsupported language (early-exit pre-processing) ===
apa kah Risiko dalam asuransi

# === Input too long (early-exit pre-processing) ===
[A single line of >2000 characters; the implementation step generates this line by repeating a template phrase rather than checking in 2 KB of literal lorem ipsum.]
```

The previous fixture `data/agent_smoke_questions.txt` is deleted in the same commit that adds the new file.

### 5.2 Expected behaviour table

| # | Case | agent_intent | tool_calls | web | satisfied |
|---|---|---|---|---|---|
| 1 | English substantive | `core_knowledge` | 1 | False | True |
| 2 | Vietnamese substantive | `core_knowledge` | 1 | False | True |
| 3 | Japanese substantive | `core_knowledge` | 1 | False | True |
| 4 | Course structure | `overall_course_knowledge` | 1 | False | True |
| 5 | Off-topic | `off_topic` | 1 | False | False |
| 6 | Vague | `core_knowledge` | 1 | False | False |
| 7 | Time-sensitive web | `core_knowledge` | 1-2 | True | True |
| 8 | Quiz keyword | `quiz` | 0 | False | False |
| 9 | Unsupported language | `off_topic` | 0 | False | False |
| 10 | Input too long | `off_topic` | 0 | False | False |

This table is the regression contract for the optimization. The benchmark step compares actual CSV output against these rows.

### 5.3 Runner

`PYTHONPATH=. python scripts/run_agent.py --questions data/agent_test_questions.txt > bench/results.csv`

Runs all 10 cases sequentially through both the agent and the old pipeline (the `--questions` mode wires both already), emits a 14-column CSV per question.

## 6. Benchmark Methodology

### 6.1 Capture before-state

Before any code change, run the fixture against the current `main` (commit `37930bf`):

```
PYTHONPATH=. python scripts/run_agent.py --questions data/agent_test_questions.txt > bench/before.csv
```

This produces the baseline `agent_latency_s`, `old_latency_s`, intents, and `tool_call_count` for all 10 cases.

### 6.2 Capture after-state

After applying A and C, re-run the same fixture:

```
PYTHONPATH=. python scripts/run_agent.py --questions data/agent_test_questions.txt > bench/after.csv
```

### 6.3 Acceptance criteria

- **Latency:** mean `agent_latency_s` over cases 1, 2, 3, 4 (substantive + course-structure paths that exercise the dispatcher + tool execution) drops by **≥ 30 %** vs `before.csv`.
- **Behaviour parity:** every case's `agent_intent` matches the table in §5.2. `tool_call_count` matches within ±1 (the dispatcher may occasionally retry).
- **No exceptions:** no row has `agent_intent == "error"` from the top-level `try/except` in `run_agent_pipeline`.
- **Pre-processing paths unaffected:** cases 8, 9, 10 latency stays under 1 s (these never reach the dispatcher).

If any criterion fails:

- Off-topic regress (case 5 doesn't yield `off_topic` intent) → revert §3 (re-add `clarity_check_node` to the graph), commit, re-bench.
- Latency target missed → keep the optimization, document the gap, defer speculative-HyDE work to a follow-up spec.

### 6.4 Storage

`bench/` is a new directory. `bench/.gitignore` ignores `*.csv` so benchmark outputs are never committed. The fixture file (`data/agent_test_questions.txt`) is committed.

## 7. File Changes

| Path | Action | Reason |
|---|---|---|
| `src/config/app_config.py` | Modify | Add `OPENAI_DISPATCH_MODEL` optional field |
| `src/rag/search/agent/agent_node.py` | Modify | Use dispatcher deployment + `max_tokens=512` |
| `src/rag/search/agent/graph.py` | Modify | Remove `clarity_check_node` node + rewire conditional edge |
| `src/rag/search/agent/streaming.py` | Create | `agent_stream_events()` async generator producing SSE event dicts |
| `src/apis/app_controller.py` | Modify | Add `POST /api/v1/chat/ask/stream/agent` route handler |
| `data/agent_test_questions.txt` | Create | 10-case fixture |
| `data/agent_smoke_questions.txt` | Delete | Superseded by the 10-case fixture |
| `bench/.gitignore` | Create | Ignore `*.csv` outputs |
| `docs/superpowers/specs/2026-04-25-langgraph-agent-search-design.md` | No change | The Phase 2 deferral note in §12 stays accurate; this spec is the follow-up |

Files NOT touched: `pipeline.py`, `entrypoint.py`, `prompt.py`, `searxng_search.py`, `reranker.py`, `model.py`, `nodes.py`, `tools.py`, `prompts.py`, `state.py`, `__init__.py`, `tests/manual_smoke_agent.py`.

## 8. Error Handling

| Failure | Response |
|---|---|
| Streaming generator raises before any event | Route handler's outer `try/except` emits `event: error` + done |
| Streaming generator raises mid-stream (e.g. LLM mid-token) | Outer handler catches it, emits `event: error` + done; client sees a partial answer plus an error event |
| `OPENAI_DISPATCH_MODEL` set to a non-existent deployment | `AzureChatOpenAI` raises on first call. `agent_decide_node` doesn't catch this — it propagates to `run_agent_pipeline`'s top-level `except`, which returns the safe `INTENT_OFF_TOPIC` fallback for non-streaming and emits `event: error` for streaming. The user sees a generic refusal; logs show the deployment-name failure. Fix: correct the env var. |
| Dispatcher LLM mis-classifies off-topic as substantive | Agent calls `search_core` with no useful results, then likely `search_web`. Acceptance criterion §6.3 catches this regression. |
| Concurrent streaming requests hitting the same compiled graph | `get_compiled_graph()` is thread-safe (commit `b215330` added the lock); LangGraph `astream` is per-invocation, no shared mutable state. |

## 9. Out of Scope / Deferred

- Speculative HyDE (run `_ahyde_generate` in parallel with `agent_decide`). Discussed during brainstorming, deferred. Trigger condition: after this spec's optimization lands and benchmarks show the dispatcher is still the dominant latency contributor for substantive questions.
- Replacing the existing `/ask/stream` with the agent. The agent route ships separately so the production pipeline stays untouched until A/B comparison data justifies a default switch.
- Per-tool latency instrumentation in the CSV (e.g. breaking out `dispatcher_latency_s` from `agent_latency_s`). Useful for follow-up debugging; out of scope here.
- Token-by-token streaming for the `web_answer` path. `web_rag_answer` returns the full answer non-streamed; the streaming endpoint emits it as a single `delta` for now. A follow-up could refactor `web_rag_answer` to stream.
