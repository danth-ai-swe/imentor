# Agent Streaming Endpoint + Latency Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a parallel SSE streaming endpoint for the LangGraph agent (`POST /api/v1/chat/ask/stream/agent`), apply two latency optimizations (drop `clarity_check_node`; route the dispatcher LLM to `gpt-4o-mini` via a new env var), and add a 10-case test fixture used to gate the optimizations against an acceptance contract.

**Architecture:** The fixture goes in first so a pre-change baseline can be captured. Optimizations are applied in two small, independently revertable commits. The streaming module reuses the existing compiled graph with `interrupt_before=["generate"]` plus a `MemorySaver` checkpointer so the agent runs all logic up to (but not including) answer generation; the streaming module then calls `llm.astream_creative` directly to forward token-level chunks as SSE `delta` events. The new FastAPI route is a thin wrapper that JSON-encodes those event dicts.

**Tech Stack:** Python 3.10, `langgraph 1.1.8` (with `MemorySaver`, `interrupt_before`), `langchain-openai` (`AzureChatOpenAI`), `fastapi`, existing `AzureChatClient.astream_creative`, existing pipeline helpers (`extract_sources`, `build_final_prompt`).

**Reference spec:** `docs/superpowers/specs/2026-04-25-agent-streaming-and-optimization-design.md`

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `data/agent_test_questions.txt` | Create | 10-case fixture (replaces `agent_smoke_questions.txt`) |
| `data/agent_smoke_questions.txt` | Delete | Superseded |
| `bench/.gitignore` | Create | Ignore `*.csv` outputs |
| `src/config/app_config.py` | Modify | Add `OPENAI_DISPATCH_MODEL: Optional[str] = None` field |
| `src/rag/search/agent/agent_node.py` | Modify | Use dispatch deployment + `max_tokens=512` |
| `src/rag/search/agent/graph.py` | Modify | Drop `clarity_check_node` from builder; add `checkpointer`/`interrupt_before` kwargs to `build_agent_graph` |
| `src/rag/search/agent/streaming.py` | Create | `agent_stream_events()` async generator + streaming-graph singleton |
| `src/apis/app_controller.py` | Modify | Add `POST /api/v1/chat/ask/stream/agent` handler |

Files NOT touched by this plan: `pipeline.py`, `entrypoint.py`, `prompt.py`, `searxng_search.py`, `reranker.py`, `model.py`, `nodes.py`, `tools.py`, `prompts.py`, `state.py`, `__init__.py`, `tests/manual_smoke_agent.py`.

---

## Task 1: Add the 10-case test fixture

**Files:**
- Create: `data/agent_test_questions.txt`
- Delete: `data/agent_smoke_questions.txt`

- [ ] **Step 1: Create `data/agent_test_questions.txt` with cases 1-9**

`data/agent_test_questions.txt`:

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
```

- [ ] **Step 2: Append the long-input case (case 10) programmatically**

The 10th case is a single line over 2000 characters. To avoid embedding 2 KB of literal text, append it via a one-line Python script:

```bash
PYTHONPATH=. python -c "
line = ('Insurance fundamentals require a thorough understanding of risk pooling, underwriting principles, and policyholder protection. ' * 22).strip()
assert len(line) > 2000, f'len={len(line)}'
with open('data/agent_test_questions.txt', 'a', encoding='utf-8') as f:
    f.write('\n# === Input too long (early-exit pre-processing) ===\n' + line + '\n')
print('appended; total file length:', sum(1 for _ in open('data/agent_test_questions.txt', encoding='utf-8')), 'lines')
"
```

Expected: `appended; total file length: 21 lines` (or thereabouts — the exact count depends on blank lines between sections).

- [ ] **Step 3: Verify line 10 (after comment-stripping) is over 2000 chars**

```bash
PYTHONPATH=. python -c "
with open('data/agent_test_questions.txt', encoding='utf-8') as f:
    qs = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
print('non-comment question count:', len(qs))
print('case 10 length:', len(qs[9]))
assert len(qs) == 10, 'expected 10 cases'
assert len(qs[9]) > 2000, 'case 10 should be over MAX_INPUT_CHARS'
"
```

Expected output: `non-comment question count: 10`, `case 10 length: 2530` (or any number > 2000).

- [ ] **Step 4: Delete the old smoke fixture**

```bash
rm data/agent_smoke_questions.txt
```

- [ ] **Step 5: Commit**

```bash
git add data/agent_test_questions.txt
git rm data/agent_smoke_questions.txt
git commit -m "feat(agent): replace 1-question smoke fixture with 10-case test fixture"
```

---

## Task 2: Add `bench/.gitignore`

**Files:**
- Create: `bench/.gitignore`

- [ ] **Step 1: Create directory + gitignore**

`bench/.gitignore`:

```
# Benchmark CSV outputs are local-only; the questions fixture is
# committed in data/agent_test_questions.txt.
*.csv
```

- [ ] **Step 2: Verify directory is tracked but contents excluded**

```bash
mkdir -p bench
ls bench/
```

Expected: only `.gitignore` listed.

- [ ] **Step 3: Commit**

```bash
git add bench/.gitignore
git commit -m "chore(bench): add bench directory + gitignore for CSV outputs"
```

---

## Task 3: Capture baseline benchmark

**Files:** None (verification step)

**Goal:** Produce `bench/before.csv` against the current `main` HEAD (commit produced by Task 2). This is the latency baseline the optimizations will be measured against. **No code changes in this task.**

- [ ] **Step 1: Confirm working tree is clean and on main**

```bash
git status --short
git log --oneline -1
```

Expected: no uncommitted changes; HEAD is the commit from Task 2.

- [ ] **Step 2: Run the fixture against the current main**

```bash
PYTHONPATH=. python scripts/run_agent.py --questions data/agent_test_questions.txt > bench/before.csv 2> bench/before.log
```

This takes 60-180 s (10 questions × both pipelines × ~6-10 s/question). The shell does not return until completion. If the command hangs > 5 min on any single question, kill it and investigate (most likely the `AzureChatOpenAI` deadlock fixed in commit `37930bf` regressed, or the dispatcher deployment is unreachable).

- [ ] **Step 3: Verify all 10 rows are present**

```bash
wc -l bench/before.csv
head -1 bench/before.csv
cut -d, -f1 bench/before.csv | tail -10
```

Expected: 11 lines (1 header + 10 data rows). Header starts with `question,agent_intent,...`. The 10 question column values match (in order) the cases in `data/agent_test_questions.txt`.

- [ ] **Step 4: Verify `bench/before.csv` is gitignored**

```bash
git status --short bench/
```

Expected: empty output (the CSV is matched by `bench/.gitignore`).

- [ ] **Step 5: Record the baseline median substantive-question latency**

```bash
PYTHONPATH=. python -c "
import csv
with open('bench/before.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
substantive = [float(r['agent_latency_s']) for r in rows[:4]]  # cases 1-4
print('baseline agent latency for cases 1-4:', substantive)
print('mean:', sum(substantive) / len(substantive))
"
```

Expected: 4 latency values (in seconds), mean roughly 8-12 s. Record this number; the acceptance criterion in Task 6 requires the post-optimization mean to drop ≥ 30 % from this baseline.

- [ ] **Step 6: No commit**

`bench/before.csv` is gitignored on purpose — keep it local for the duration of this plan.

---

## Task 4: Optimization A — drop `clarity_check_node` from the graph

**Files:**
- Modify: `src/rag/search/agent/graph.py`

- [ ] **Step 1: Edit `_route_after_quiz` to route directly to `agent_decide`**

In `src/rag/search/agent/graph.py`, change:

```python
def _route_after_quiz(state: AgentState) -> str:
    return "finalize" if state.get("early_exit_reason") else "clarity_check"
```

to:

```python
def _route_after_quiz(state: AgentState) -> str:
    return "finalize" if state.get("early_exit_reason") else "agent_decide"
```

- [ ] **Step 2: Remove the `clarity_check` node and its conditional edge from `build_agent_graph`**

Find these two lines in `build_agent_graph`:

```python
    builder.add_node("clarity_check", clarity_check_node)
```

Delete that line.

Find this block:

```python
    builder.add_conditional_edges("quiz_check", _route_after_quiz,
                                  {"finalize": "finalize", "clarity_check": "clarity_check"})
    builder.add_conditional_edges("clarity_check", _route_after_clarity,
                                  {"finalize": "finalize", "agent_decide": "agent_decide"})
```

Replace it with:

```python
    builder.add_conditional_edges("quiz_check", _route_after_quiz,
                                  {"finalize": "finalize", "agent_decide": "agent_decide"})
```

(The second `add_conditional_edges` call is removed entirely. `_route_after_clarity` becomes dead code — remove its definition too.)

- [ ] **Step 3: Remove the dead `_route_after_clarity` function**

In the same file, delete:

```python
def _route_after_clarity(state: AgentState) -> str:
    return "finalize" if state.get("early_exit_reason") else "agent_decide"
```

- [ ] **Step 4: Remove the dead import of `clarity_check_node`**

In the imports block of `graph.py`, find:

```python
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
```

Remove the `clarity_check_node,` line:

```python
from src.rag.search.agent.nodes import (
    detect_and_rewrite_node,
    enrich_node,
    finalize_node,
    generate_node,
    quiz_check_node,
    rerank_node,
    validate_input_node,
)
```

- [ ] **Step 5: Verify the graph still compiles and `clarity_check` is gone**

```bash
PYTHONPATH=. python -c "
from src.rag.search.agent.graph import build_agent_graph
g = build_agent_graph()
nodes = sorted(g.get_graph().nodes.keys())
print('nodes:', nodes)
assert 'clarity_check' not in nodes, 'clarity_check should be removed'
assert 'agent_decide' in nodes
"
```

Expected: prints the node list (12 entries: `__end__`, `__start__`, `agent_decide`, `detect_and_rewrite`, `enrich`, `finalize`, `generate`, `post_router`, `quiz_check`, `rerank`, `tools`, `validate_input`) and exits 0.

- [ ] **Step 6: Verify the existing finalize smoke test still passes**

```bash
PYTHONPATH=. python tests/manual_smoke_agent.py
```

Expected: `All 11 cases passed.`

- [ ] **Step 7: Commit**

```bash
git add src/rag/search/agent/graph.py
git commit -m "perf(agent): drop clarity_check_node — dispatcher prompt already handles off_topic/vague via ask_clarification"
```

---

## Task 5: Optimization C — dedicated dispatcher deployment

**Files:**
- Modify: `src/config/app_config.py`
- Modify: `src/rag/search/agent/agent_node.py`

- [ ] **Step 1: Add `OPENAI_DISPATCH_MODEL` field to `AppConfig`**

In `src/config/app_config.py`, find this block:

```python
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_API_VERSION: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
```

Insert one new line after `OPENAI_CHAT_MODEL`:

```python
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_DISPATCH_MODEL: Optional[str] = None
    OPENAI_API_VERSION: Optional[str] = None
    OPENAI_API_BASE: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
```

- [ ] **Step 2: Verify `AppConfig` accepts the new field**

```bash
PYTHONPATH=. python -c "
from src.config.app_config import get_app_config
c = get_app_config()
print('OPENAI_DISPATCH_MODEL:', c.OPENAI_DISPATCH_MODEL)
print('OPENAI_CHAT_MODEL:', c.OPENAI_CHAT_MODEL)
"
```

Expected: `OPENAI_DISPATCH_MODEL: None` (assuming the user hasn't yet set the env var) and the chat model from `.env`. No `ValidationError`.

- [ ] **Step 3: Update `_get_bound_llm` in `agent_node.py` to use the dispatch deployment**

In `src/rag/search/agent/agent_node.py`, find `_get_bound_llm`:

```python
def _get_bound_llm():
    global _bound_llm
    if _bound_llm is None:
        with _llm_lock:
            if _bound_llm is None:
                # Build AzureChatOpenAI inline, không qua get_agent_llm()
                llm = _llm_instance or AzureChatOpenAI(
                    azure_endpoint=config.OPENAI_API_BASE,
                    api_key=config.OPENAI_API_KEY,
                    api_version=config.OPENAI_API_VERSION,
                    azure_deployment=config.OPENAI_CHAT_MODEL,
                    temperature=0.0,
                    timeout=config.GPT_TIMEOUT,
                    max_retries=config.GPT_MAX_RETRIES,
                )
                _bound_llm = llm.bind_tools(ALL_TOOLS)
    return _bound_llm
```

Replace it with:

```python
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
```

(The change drops the `_llm_instance or ...` fallback — that pattern was a workaround for the deadlock fixed in commit `37930bf` and is no longer needed since `_get_bound_llm` no longer touches `_llm_instance`.)

- [ ] **Step 4: Verify the bound LLM constructs cleanly**

```bash
PYTHONPATH=. python -c "
from src.rag.search.agent.agent_node import _get_bound_llm, get_agent_llm
b = _get_bound_llm()
print('bound type:', type(b).__name__)
print('base singleton still works:', type(get_agent_llm()).__name__)
"
```

Expected: `bound type: RunnableBinding` (or similar — `bind_tools` returns a wrapper). Second line: `AzureChatOpenAI`.

- [ ] **Step 5: Set the env var (manual user step)**

The user is expected to add this line to `.env`:

```
OPENAI_DISPATCH_MODEL=gpt-4o-mini
```

If they skip this, the dispatcher falls back to `OPENAI_CHAT_MODEL` (no behaviour change). The benchmark in Task 6 measures the optimization's effect; if the user has not set the env var, the latency improvement will come from Task 4 alone.

The plan does NOT modify `.env` — that file is environment-specific and not under version control.

- [ ] **Step 6: Commit**

```bash
git add src/config/app_config.py src/rag/search/agent/agent_node.py
git commit -m "perf(agent): route dispatcher LLM to OPENAI_DISPATCH_MODEL when set + cap max_tokens=512"
```

---

## Task 6: Capture optimized benchmark + verify acceptance

**Files:** None (verification + decision step)

- [ ] **Step 1: Re-run the fixture against the post-optimization HEAD**

```bash
PYTHONPATH=. python scripts/run_agent.py --questions data/agent_test_questions.txt > bench/after.csv 2> bench/after.log
```

Expected runtime: faster than the baseline (the savings are ≈ 1 s for clarity-check elimination + ≈ 1-1.5 s for the smaller dispatcher × 10 questions = roughly 20-25 s shorter total).

- [ ] **Step 2: Confirm intent parity vs the spec table**

```bash
PYTHONPATH=. python -c "
import csv
expected = [
    ('core_knowledge', True, False),    # 1 EN substantive
    ('core_knowledge', True, False),    # 2 VN substantive
    ('core_knowledge', True, False),    # 3 JP substantive
    ('overall_course_knowledge', True, False),  # 4 course
    ('off_topic', False, False),        # 5 off-topic
    ('core_knowledge', False, False),   # 6 vague
    ('core_knowledge', True, True),     # 7 web (intent core, web=True)
    ('quiz', False, False),             # 8 quiz keyword
    ('off_topic', False, False),        # 9 unsupported language
    ('off_topic', False, False),        # 10 input too long
]
with open('bench/after.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
mismatches = []
for i, (row, exp) in enumerate(zip(rows, expected)):
    got = (row['agent_intent'], row['agent_satisfied'] == 'True', row['agent_web'] == 'True')
    if got != exp:
        mismatches.append((i + 1, exp, got))
if mismatches:
    print('intent mismatches:', mismatches)
else:
    print('all 10 cases match expected intent / satisfied / web flags')
"
```

Expected: `all 10 cases match expected intent / satisfied / web flags`. If there are mismatches, see Step 5 for the rollback decision.

- [ ] **Step 3: Confirm latency improvement on substantive cases (≥ 30 % drop)**

```bash
PYTHONPATH=. python -c "
import csv
def mean(rows): return sum(float(r['agent_latency_s']) for r in rows) / len(rows)
with open('bench/before.csv', encoding='utf-8') as f:
    before = list(csv.DictReader(f))[:4]
with open('bench/after.csv', encoding='utf-8') as f:
    after = list(csv.DictReader(f))[:4]
mb, ma = mean(before), mean(after)
drop = (mb - ma) / mb * 100
print(f'baseline mean: {mb:.2f}s')
print(f'optimized mean: {ma:.2f}s')
print(f'drop: {drop:.1f}%')
assert drop >= 30, f'expected ≥ 30%% drop, got {drop:.1f}%%'
print('PASS: latency criterion met')
"
```

Expected: prints baseline + optimized + drop, exits 0 with `PASS: latency criterion met`. If `drop < 30 %`, see Step 5.

- [ ] **Step 4: Confirm no error rows**

```bash
grep -c '^[^,]*,error,' bench/after.csv || true
```

Expected: 0 (no row has `agent_intent` set to `error` from the top-level `try/except` in `run_agent_pipeline`).

- [ ] **Step 5: Decision branch**

If all three checks (Steps 2, 3, 4) pass: proceed to Task 7.

If any check fails:

- **Intent mismatch on case 5 (off-topic detected as something else)**: revert Task 4. Run `git revert <task-4-commit>`, re-bench, commit the revert. Then proceed to Task 7. Document the regression in the commit message.
- **Latency improvement < 30 %**: keep both optimizations, document in the commit message that the gain came primarily from one of them (you can identify which by reverting Task 5 alone and re-benching), and proceed to Task 7. The streaming endpoint work is independent of this gate.
- **Error rows present**: investigate the `bench/after.log` for tracebacks. The most common cause is the user setting `OPENAI_DISPATCH_MODEL` to a deployment that doesn't exist on their Azure resource. Roll back Task 5's `OPENAI_DISPATCH_MODEL` env var (have the user unset it; the code stays). Re-bench. Then proceed to Task 7.

- [ ] **Step 6: No commit**

`bench/*.csv` and `bench/*.log` are gitignored. The decision in Step 5 is recorded in the commit message of any rollback (if needed) or simply by the absence of rollbacks.

---

## Task 7: Add `checkpointer` and `interrupt_before` kwargs to `build_agent_graph`

**Files:**
- Modify: `src/rag/search/agent/graph.py`

The streaming module (Task 8) needs to compile a second copy of the graph with `interrupt_before=["generate"]` and a `MemorySaver` checkpointer so the agent runs every node up to but not including answer generation. Today's `build_agent_graph` accepts no parameters.

- [ ] **Step 1: Update the function signature**

In `src/rag/search/agent/graph.py`, find:

```python
def build_agent_graph():
    builder = StateGraph(AgentState)
```

Replace with:

```python
def build_agent_graph(checkpointer=None, interrupt_before=None):
    """Compile the agent StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. MemorySaver).
            Required if interrupt_before is provided.
        interrupt_before: Optional list of node names to halt before.
            Used by the streaming endpoint to stop before `generate` so
            tokens can be streamed manually.
    """
    builder = StateGraph(AgentState)
```

- [ ] **Step 2: Pass the kwargs into `builder.compile()`**

At the end of the same function, find:

```python
    builder.add_edge("finalize", END)

    return builder.compile()
```

Replace with:

```python
    builder.add_edge("finalize", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )
```

- [ ] **Step 3: Verify the existing default invocation still works**

```bash
PYTHONPATH=. python -c "
from src.rag.search.agent.graph import build_agent_graph, get_compiled_graph
g = build_agent_graph()
print('default compile OK, nodes:', len(g.get_graph().nodes))
g2 = get_compiled_graph()
print('cached singleton OK:', type(g2).__name__)
"
```

Expected: `default compile OK, nodes: 12` (after Task 4 dropped clarity_check). `cached singleton OK: CompiledStateGraph`.

- [ ] **Step 4: Verify the new kwargs work**

```bash
PYTHONPATH=. python -c "
from langgraph.checkpoint.memory import MemorySaver
from src.rag.search.agent.graph import build_agent_graph
g = build_agent_graph(checkpointer=MemorySaver(), interrupt_before=['generate'])
print('streaming compile OK, nodes:', len(g.get_graph().nodes))
"
```

Expected: `streaming compile OK, nodes: 12`. No errors.

- [ ] **Step 5: Commit**

```bash
git add src/rag/search/agent/graph.py
git commit -m "feat(agent): add checkpointer + interrupt_before kwargs to build_agent_graph"
```

---

## Task 8: Create the streaming module

**Files:**
- Create: `src/rag/search/agent/streaming.py`

- [ ] **Step 1: Write `streaming.py`**

`src/rag/search/agent/streaming.py`:

```python
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
    return {
        "type": "meta",
        "intent": intent,
        "detected_language": state.get("detected_language"),
        "sources": sources,
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
```

- [ ] **Step 2: Verify the module imports and the streaming graph compiles**

```bash
PYTHONPATH=. python -c "
from src.rag.search.agent.streaming import agent_stream_events, get_streaming_graph
g = get_streaming_graph()
print('streaming graph compiled, node count:', len(g.get_graph().nodes))
print('agent_stream_events callable:', callable(agent_stream_events))
"
```

Expected: `streaming graph compiled, node count: 12`. `agent_stream_events callable: True`.

- [ ] **Step 3: Commit**

```bash
git add src/rag/search/agent/streaming.py
git commit -m "feat(agent): add streaming.py — agent_stream_events generator + interrupted graph"
```

---

## Task 9: Add the `/ask/stream/agent` route

**Files:**
- Modify: `src/apis/app_controller.py`

- [ ] **Step 1: Add the import for `agent_stream_events`**

In `src/apis/app_controller.py`, find the existing import block:

```python
from src.rag.search.pipeline import async_pipeline_dispatch, async_pipeline_dispatch_stream
```

Add a new line below it:

```python
from src.rag.search.pipeline import async_pipeline_dispatch, async_pipeline_dispatch_stream
from src.rag.search.agent.streaming import agent_stream_events
```

- [ ] **Step 2: Add the route handler**

In the same file, locate the `chat_ask_stream` handler (look for `@chat_router.post("/ask/stream")`). Add a new handler immediately after its closing `)` of `StreamingResponse(...)` (i.e., after the existing handler, before the `@file_router.get("/t")` line):

```python
@chat_router.post("/ask/stream/agent")
async def chat_ask_stream_agent(payload: ChatRequest) -> StreamingResponse:
    """SSE endpoint for the LangGraph hybrid agent.

    Same event shape as /ask/stream:
      event: meta  → {intent, detected_language, sources, answer_satisfied,
                      web_search_used, tool_call_count}
      event: delta → {content: "<token piece>"}
      event: done  → {}
      event: error → {message: "<exc str>"}  (on uncaught failure)
    """
    logger.info(
        "chat_ask_stream_agent called | user_name=%s | conversation_id=%s",
        payload.user_name, payload.conversation_id,
    )

    async def event_source():
        try:
            async for ev in agent_stream_events(
                user_input=payload.message,
                conversation_id=payload.conversation_id,
            ):
                event_type = ev.get("type", "message")
                data = {k: v for k, v in ev.items() if k != "type"}
                yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("chat_ask_stream_agent failed")
            err = json.dumps({"message": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

(This is a near-clone of the existing `chat_ask_stream` handler with the function name and the underlying generator swapped.)

- [ ] **Step 3: Verify the controller imports cleanly**

```bash
PYTHONPATH=. python -c "
from src.apis.app_controller import chat_ask_stream_agent
print('handler registered:', chat_ask_stream_agent.__name__)
"
```

Expected: `handler registered: chat_ask_stream_agent`. No import errors.

- [ ] **Step 4: Commit**

```bash
git add src/apis/app_controller.py
git commit -m "feat(api): add POST /api/v1/chat/ask/stream/agent SSE route"
```

---

## Task 10: Smoke-test the streaming endpoint

**Files:** None (verification step)

**Goal:** Start the FastAPI server, hit the new route with a real question, confirm SSE events stream end-to-end.

- [ ] **Step 1: Start the server in the background**

```bash
PYTHONPATH=. python main.py > /tmp/agent_server.log 2>&1 &
SERVER_PID=$!
echo "server PID: $SERVER_PID"
```

The server takes ~10-15 s to warm up (Qdrant + reranker + semantic router setup happens in lifespan). Wait until the log contains `Application startup complete`:

```bash
until grep -q 'Application startup complete' /tmp/agent_server.log 2>/dev/null; do sleep 1; done
echo "server ready"
```

- [ ] **Step 2: Hit the new endpoint and observe the SSE stream**

```bash
curl -N -s -X POST http://localhost:8083/api/v1/chat/ask/stream/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "What is adverse selection in insurance?", "conversation_id": null, "user_name": "smoke"}' \
  | head -50
```

Expected: a stream of lines like:

```
event: meta
data: {"intent": "core_knowledge", "detected_language": "English", "sources": [...], "answer_satisfied": true, "web_search_used": false, "tool_call_count": 1}

event: delta
data: {"content": "Adverse"}

event: delta
data: {"content": " selection"}

...

event: done
data: {}
```

`-N` disables curl's output buffering so events appear live. `head -50` caps the output.

- [ ] **Step 3: Confirm no `event: error` line in the output**

```bash
curl -N -s -X POST http://localhost:8083/api/v1/chat/ask/stream/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "What is adverse selection in insurance?", "conversation_id": null, "user_name": "smoke"}' \
  > /tmp/agent_stream.out
grep -c 'event: meta' /tmp/agent_stream.out
grep -c 'event: delta' /tmp/agent_stream.out
grep -c 'event: done' /tmp/agent_stream.out
grep -c 'event: error' /tmp/agent_stream.out
```

Expected: `meta` count = 1, `delta` count > 5 (token-by-token), `done` count = 1, `error` count = 0.

- [ ] **Step 4: Stop the server**

```bash
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
```

- [ ] **Step 5: No commit**

This task only verifies behaviour. If any step failed, investigate before declaring the plan complete.

---

## Done — verification checklist

After Task 10:

- [ ] `bench/before.csv` and `bench/after.csv` both exist locally with 11 lines each.
- [ ] Task 6 acceptance criteria all green (or rollback documented).
- [ ] `tests/manual_smoke_agent.py` still passes 11/11 cases.
- [ ] `git log --oneline` shows the 7 task commits since the start of this plan (Tasks 1, 2, 4, 5, 7, 8, 9 — Tasks 3, 6, 10 are verification-only).
- [ ] Server smoke test (Task 10) emits valid SSE for one substantive question.

After all checks pass, the new `/ask/stream/agent` route is live and the agent latency target (~30 % drop on substantive questions) is verified by data.
