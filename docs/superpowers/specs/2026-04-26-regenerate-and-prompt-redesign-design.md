# Regenerate flow + Recomp summarize + softened clarity prompt — Design

**Date:** 2026-04-26
**Branch:** `feat/java-rabbitmq-react-chat`
**Author:** danth-ai-swe (paired with Claude)

## 1. Problem statement

Three independent issues addressed in one spec because they all touch the chat ask/regenerate path and share the same prompt module.

### 1.1 Regenerate duplicates the user message
On the React chat UI, clicking "Regenerate" on an assistant bubble currently:
1. Deletes the assistant message (`deleteFromMessage` — `frontend/src/components/ChatWindow.tsx:51`).
2. Calls `send(previousUser.content)`, which `appendUser`s a duplicate user bubble locally and POSTs `/ask/stream`, persisting **another** user `Message` row in the DB (`backend/.../ChatService.java:175-179`).

Result: every regenerate produces a duplicate user bubble in UI and an extra row in the DB.

### 1.2 Regenerate re-runs full retrieval
The current regenerate path goes through `/ask/stream`, which calls `rpc.requestSearch(...)` again. The original chunks were already retrieved seconds ago for the message being regenerated; re-searching wastes latency and tokens, and can return slightly different chunks (different answer is fine — different *sources* is confusing UX).

`ChunkSource` rows persist only metadata (`name`, `url`, `pageNumber`, `totalPages` — `ChatService.java:200-205`); the chunk **text payload** is ephemeral inside `SearchReplyMessage.chunks()` and lost as soon as the original `ask()` returns.

### 1.3 SUMMARIZE_PROMPT_TEMPLATE doesn't follow Recomp
Current prompt (`src/rag/search/prompt.py:23-61`) compresses turn-by-turn while preserving alternation. This is generic conversation summarization, not Recomp:
- Recomp = **abstractive, query-focused** condensation that keeps only what is relevant to the *current* query.
- Output shape `{"summary": [{role, content}, ...]}` forces preservation of every turn even when irrelevant.
- The summary feeds HyDE / answer prompts; noisy unrelated turns reduce retrieval quality.

### 1.4 CLARITY_CHECK_PROMPT is too rigid
Current prompt mixes "Decision logic", "Hard rules", and a 60+ item Knowledge Scope list as bullet rules. The form-filling tone makes the LLM lean toward false negatives (flagging clear queries as needing rephrase). The user wants a more natural, generous prompt that lets most queries pass.

## 2. Goals & non-goals

**Goals:**
- Regenerate produces exactly one new assistant message and reuses the chunks of the deleted assistant message when possible.
- User message and DB state for the user message are untouched by regenerate.
- `SUMMARIZE_PROMPT_TEMPLATE` becomes Recomp-style: query-focused, abstractive, single-string output, optionally empty.
- `CLARITY_CHECK_PROMPT` is friendlier and biases toward CLEAR while keeping the JSON schema and Knowledge Scope guardrail.

**Non-goals:**
- Persisting chunks durably (we accept TTL-based cache; if expired, regenerate falls back to re-search).
- Changing the chunk retrieval pipeline itself (pipeline.py / pipeline_chunks.py logic unchanged).
- Schema migrations on the `messages` or `chunk_sources` tables.

## 3. Architecture overview

```
[FE] Click Regenerate on bubble A
      │
      ▼
FE: optimistic remove A from local state (keep user bubble U)
      │
FE: open SSE → POST /api/conversations/{convId}/messages/{A.id}/regenerate/stream
      │
      ▼
[Java backend]
  1. Validate A exists + role=assistant + belongs to conv
  2. Find parent user message U (latest user msg before A)
  3. Acquire rate-limit + stream-lock (per user)
  4. Read Redis chunks cache:  chat:chunks:{A.id}
       hit  → reuse {chunks, standaloneQuery, detectedLanguage, intent}
       miss → fallback rpc.requestSearch(U.content) (full pipeline)
  5. Delete A row + ChunkSources(A) + Redis chat:chunks:{A.id} + invalidate Redis history
  6. Persist new empty assistant msg A' with reused intent/lang
  7. Persist ChunkSource metadata for A'; cache chunk payload at chat:chunks:{A'.id}
  8. Emit `meta` event ({assistantMessageId: A'.id, regenerated: true, chunksReused})
  9. promptBuilder.build(chunks, standaloneQuery, detectedLanguage)
 10. openai.streamChat(prompt, temperature=0.7) — bumped to bias variation
 11. Stream `delta` tokens, persist A'.content + history on completion
```

Key invariant: **U is never re-saved or re-deleted by regenerate.**

## 4. API contract

### 4.1 New endpoint
```
POST /api/conversations/{convId}/messages/{assistantMsgId}/regenerate/stream
```
- No request body. All context comes from DB + Redis.
- Returns `text/event-stream` with the same event names as `/ask/stream`:
  - `meta` — `{ intent, detectedLanguage, sources, webSearchUsed, assistantMessageId, regenerated: true, chunksReused: bool }`
  - `delta` — `{ content: "<token>" }`
  - `done` — `{}`
  - `error` — `{ message: string }`

### 4.2 Errors
- 400 if `assistantMsgId` is not role=assistant or not in `convId`.
- 404 if `assistantMsgId` not found.
- `error` event (200 SSE) if rate-limit or stream-lock rejects, or OpenAI errors mid-stream.

## 5. Backend (Java) changes

### 5.1 Redis chunk cache
File: `backend/src/main/java/com/example/chat/redis/ChatRedisService.java`

Add:
```java
private static final String CHUNKS_KEY = "chat:chunks:%d";

@Value("${chat.chunks.ttl-seconds:3600}")
long chunksTtl = 3600;

public record ChunkContext(List<Map<String,Object>> chunks, String standaloneQuery) {}

public void cacheChunks(Long assistantMessageId, ChunkContext ctx);
public ChunkContext readChunks(Long assistantMessageId);
public void deleteChunks(Long assistantMessageId);
```

Serialization: store as JSON string at key `chat:chunks:{id}` with TTL 3600s. Object shape:
```json
{ "chunks": [ ... ], "standaloneQuery": "..." }
```

### 5.2 `ChatService.ask()` — cache chunks after persisting assistant msg
After `ChatService.java:206` (chunk source loop), add:
```java
if (reply.chunks() != null && !reply.chunks().isEmpty()) {
    redis.cacheChunks(
        assistantMsg.getId(),
        new ChunkContext(reply.chunks(), reply.standaloneQuery())
    );
}
```

### 5.3 `ChatService.regenerate(convId, oldAssistantId)` — new method
Pseudocode:
```java
public SseEmitter regenerate(Long convId, Long oldAssistantId) {
    Message oldA = require(messages.findById(oldAssistantId)).inConv(convId).asAssistant();
    Conversation conv = oldA.getConversation();
    Long userId = conv.getUser().getId();
    Message userMsg = messages.findLatestUserBefore(convId, oldAssistantId)
        .orElseThrow(() -> new IllegalStateException("orphan assistant"));

    SseEmitter emitter = new SseEmitter(60_000L);
    if (!rateLimiter.tryAcquire(userId)) { emitError; return; }
    if (!streamLock.acquire(userId))     { emitError; return; }

    ChunkContext ctx = redis.readChunks(oldAssistantId);
    boolean chunksReused = (ctx != null);
    String detectedLanguage = oldA.getDetectedLanguage();
    String intent = oldA.getIntent();
    boolean webSearchUsed = oldA.isWebSearchUsed();
    if (ctx == null) {
        SearchReplyMessage reply = rpc.requestSearch(userMsg.getContent(), convId.toString());
        ctx = new ChunkContext(reply.chunks(), reply.standaloneQuery());
        detectedLanguage = reply.detectedLanguage();
        intent = reply.intent();
        webSearchUsed = reply.webSearchUsed();
    }

    chunkSources.deleteByMessageIdIn(List.of(oldAssistantId));
    messages.deleteById(oldAssistantId);
    redis.deleteChunks(oldAssistantId);
    redis.invalidateHistory(convId);

    Message newA = messages.save(Message.builder()
        .conversation(conv).role("assistant").content("")
        .intent(intent).detectedLanguage(detectedLanguage)
        .webSearchUsed(webSearchUsed).build());
    persistChunkSourcesFromPayload(newA, ctx.chunks());
    redis.cacheChunks(newA.getId(), ctx);

    sendEvent(emitter, "meta", Map.of(
        "intent", nullSafe(intent),
        "detectedLanguage", nullSafe(detectedLanguage),
        "sources", extractSourceDtos(ctx.chunks()),
        "webSearchUsed", webSearchUsed,
        "assistantMessageId", newA.getId(),
        "regenerated", true,
        "chunksReused", chunksReused
    ));

    String prompt = promptBuilder.build(ctx.chunks(), ctx.standaloneQuery(), detectedLanguage);
    Disposable sub = openai.streamChat(prompt, /*temperature*/ 0.7).subscribe(
        token -> { ... same buffer/append/delta as ask() ... },
        error -> { ... persist + release lock ... },
        ()    -> { ... persist + release lock ... }
    );
    // emitter.onTimeout/onError/onCompletion: same pattern as ask()
    return emitter;
}
```

Helper: `persistChunkSourcesFromPayload(Message m, List<Map<String,Object>> chunks)` extracts `name/url/pageNumber/totalPages` from chunk payloads (look at how `pipeline_chunks` shapes them, see §10).

### 5.4 `OpenAiChatService.streamChat` — temperature overload
Current signature is `streamChat(String prompt)`. Add overload `streamChat(String prompt, double temperature)`. Default flow stays as-is. Regenerate path passes `0.7`.

### 5.5 `MessageRepository` — find latest user before id
```java
@Query("SELECT m FROM Message m WHERE m.conversation.id = :convId AND m.id < :before AND m.role = 'user' ORDER BY m.id DESC LIMIT 1")
Optional<Message> findLatestUserBefore(@Param("convId") Long convId, @Param("before") Long before);
```

### 5.6 `ChatController` — new mapping
```java
@PostMapping("/conversations/{convId}/messages/{assistantMsgId}/regenerate/stream")
public SseEmitter regenerate(@PathVariable Long convId, @PathVariable Long assistantMsgId) {
    return service.regenerate(convId, assistantMsgId);
}
```

### 5.7 `application.yml`
Add:
```yaml
chat:
  chunks:
    ttl-seconds: 3600
  regenerate:
    temperature: 0.7
```
Wire `@Value("${chat.regenerate.temperature:0.7}")` into `ChatService` and pass to `streamChat`.

## 6. Frontend changes

### 6.1 `frontend/src/api/chatApi.ts`
Export URL helper:
```typescript
export function regenerateStreamUrl(convId: number, assistantMsgId: number): string {
  return `${API_BASE}/conversations/${convId}/messages/${assistantMsgId}/regenerate/stream`;
}
```

### 6.2 `frontend/src/hooks/useChatStream.ts`
- Add state `regenerating: boolean`.
- Add method `regenerate(convId, assistantMsgId, appendPlaceholder, patchPlaceholder, setError)` that opens SSE on `regenerateStreamUrl`.
- The placeholder it appends is marked with `streamingMode: 'regenerate'` (new optional field on `Message` or kept in component state) so the bubble can render "regenerating…" instead of "thinking…".
- Hook surface becomes: `{ streaming, regenerating, start, regenerate, stop }`.

### 6.3 `frontend/src/components/ChatWindow.tsx`
Replace `regenerate` (lines 46-54):
```typescript
async function regenerate(assistantMsg: Message) {
  const idx = messages.findIndex((m) => m.id === assistantMsg.id);
  if (idx < 1) return;
  if (messages[idx - 1].role !== 'user') return;
  setMessages((m) => m.filter((x) => x.id !== assistantMsg.id));
  await regenerateStream(
    conversation.id,
    assistantMsg.id,
    appendPlaceholder,
    patchPlaceholder,
    setError,
  );
  const fresh = await loadMessages(conversation.id);
  setMessages(fresh);
}
```
- No `deleteFromMessage` (backend regenerate handles deletion atomically).
- No `send(...)` (which would `appendUser` a duplicate).
- User bubble untouched.

Composer disabled while `streaming || regenerating`:
```tsx
<textarea ... disabled={streaming || regenerating} />
{streaming || regenerating
  ? <button type="button" className="stop-btn" onClick={stop}>■ Stop</button>
  : <button type="submit" disabled={!input.trim()}>Send</button>}
```

### 6.4 `frontend/src/components/MessageBubble.tsx`
- New optional prop `streamingMode?: 'ask' | 'regenerate'`.
- When `streaming === true && streamingMode === 'regenerate'`, render placeholder text "regenerating…" instead of "thinking…".
- Disable `onRegenerate` button when any stream is active.

## 7. Recomp `SUMMARIZE_PROMPT_TEMPLATE`

File: `src/rag/search/prompt.py`

```python
SUMMARIZE_PROMPT_TEMPLATE = """
You are a Recomp-style abstractive summarizer for a RAG pipeline.
Your job: read the prior conversation and produce a SHORT, query-focused summary
that an AI assistant can use as memory context to answer the user's CURRENT query.

Output language: English only, regardless of the conversation language.

<current_query>
{query}
</current_query>

<conversation_history>
{conversation_history}
</conversation_history>

How to summarize:
- Keep ONLY information from history that is relevant to the current query.
- Drop greetings, filler, and turns unrelated to the query topic.
- Preserve concrete facts the user already received: numbers, definitions, named concepts, prior decisions.
- Write 1-3 sentences in third person ("The user asked about X; the assistant explained Y, including Z.").
- If nothing in the history is relevant to the current query, return an empty string for "summary".
- Never invent facts. Never copy the current query into the summary.

Return ONLY a JSON object:
{{"summary": "<concise query-focused summary, or empty string>"}}

Start with exactly: {{"summary":
"""
```

### 7.1 Caller updates (`src/rag/search/entrypoint.py`)
- `_summarize_history(llm, history_string, query)` — add `query` parameter; format both into the new template.
- `afetch_chat_history(llm, conversation_id, query)` — add `query` parameter; early-return `""` if `query` is blank.
- Update **every** call site of `afetch_chat_history` to pass the user query (grep `afetch_chat_history(`).

### 7.2 Compatibility
- Output is now a string (not a list). Existing parser `parse_json_response(raw).get("summary", "")` returns the string directly. Downstream consumers already treat the result as a string (`afetch_chat_history` returns `str`).

## 8. Softened `CLARITY_CHECK_PROMPT`

File: `src/rag/search/prompt.py` (replace lines 62-131).

```python
CLARITY_CHECK_PROMPT = """
You are **Insuripedia**, a warm LOMA281/LOMA291 study buddy. You help insurance students,
and right now you're glancing at their query to decide whether it's clear enough to answer.

<query>
{standalone_query}
</query>
<reply_language>{response_language}</reply_language>

How to judge:
- Be generous. If you can guess what the student is asking — even with typos, broken grammar, or vague phrasing — mark it CLEAR. Most queries should pass.
- Only stop the student in two cases:
    - The query has a guessable topic but is too vague to retrieve a useful answer → ask ONE warm follow-up to narrow it down (type: "rephrase").
    - The query is gibberish, random characters, or completely outside insurance → offer 3 example questions to redirect them (type: "suggestions").
- Never mix rephrase and suggestions — pick one.
- Suggestions must be in {response_language}, sound natural, no numbering, and stay within the Knowledge Scope below.

<example>
Input: "What is the Law of Large Numbers in insurance?"
Output: {{"clear": true}}
</example>

<example>
Input: "abcxyz loma"
Output: {{"clear": false, "type": "suggestions", "response": ["Các kỹ thuật quản lý rủi ro trong bảo hiểm là gì?", "Nguyên tắc bồi thường trong hợp đồng bảo hiểm hoạt động như thế nào?", "Tái bảo hiểm là gì và tại sao nó quan trọng?"]}}
</example>

Knowledge Scope:
Risk Concepts, Risk Management Techniques, Insurance Basics, Insurance Roles & Parties,
Insurance Contracts & Principles, Reinsurance Concepts, Industry Overview, Institution Types,
Insurance Fundamentals, Insurance Company Organization, Insurance Regulation, Consumer Protection,
Contract Type, Contract Validity, Valid Contract Req., Insurance Requirement, Role, Concept,
Factor, Metric, Tool, Principle, Product Type, Product Sub-Type, Product Example, Product Feature,
Policy Feature, Policy Action, Product Category, Feature, Action/Feature, Product, Annuity Basics,
Annuity Types, Contract Features, Payout Options, Insurance Type, Insurance Product,
Cost Participation Method, Regulation, Health Plan Type, Managed Care Plan, Benefit Trigger,
Disability Definition, Government Program, General Concept, Rider Category, Rider Type,
Policy Structure, Standard Provision, Beneficiary Type, Optional Provision, Assignment Type,
Policy Provision, Nonforfeiture Option, Dividend Option, Group Insurance Contract,
Group Insurance Document, Plan Administration, Underwriting, Policy Event, Plan Type,
Plan Category, Plan Sub-Category, Business Concept, Management Strategy, Stakeholder Group,
Management Principles, Management Function, Organizational Structure, Functional Area,
Core Concept, Investment Principle, Investment Component, Risk Management, Risk Management Model,
Distribution Category, Distribution System, Distribution Strategy, Core Process, Core Function,
Process Improvement, Underwriting Method, Underwriting Component, Underwriting Outcome,
Data & Analytics, Financial Transaction, Performance Management, Performance Metric,
Claim Process Step, Payment Method, Compliance, Marketing Framework, Marketing Component,
Marketing Process, Marketing Concept, Process Stage, Management Framework, Key Role,
Financial Function, Reporting Tool, Accounting Standard, Financial Goal, Solvency Component,
Regulatory Standard, Financial Reporting

Return JSON only — no markdown, no extra keys.

Schema:
  CLEAR       -> {{"clear": true}}
  rephrase    -> {{"clear": false, "type": "rephrase", "response": "<one warm sentence in {response_language}>"}}
  suggestions -> {{"clear": false, "type": "suggestions", "response": ["<q1>", "<q2>", "<q3>"]}}

Start with exactly: {{"clear":
"""
```

Schema and start marker preserved → callers `_acheck_input_clarity` (`pipeline.py:116-131`), `pipeline.py:372-379`, `pipeline.py:437-442`, `pipeline_chunks.py:132-147`, and `agent/nodes.py:107-127` need no parser changes.

## 9. Data flow summary

| Step | ask() (existing) | regenerate() (new) |
|------|------------------|---------------------|
| User msg | persist NEW row | untouched |
| RPC search | always | only on cache miss |
| Chunk cache | written for assistant msg | read for old A; rewritten for new A' |
| Old assistant msg | n/a | deleted (row + ChunkSource + chunk cache) |
| New assistant msg | persist | persist |
| Temperature | default | 0.7 |
| Redis history | append user + assistant | invalidate, then append new assistant |

## 10. Open implementation notes

- `pipeline_chunks.py` returns chunks with payloads shaped like `{ payload: { file_name, course, module_number, lesson_number, page_number }, text }` (see `entrypoint.py:99-122`). `persistChunkSourcesFromPayload` and `extractSourceDtos` must inspect the same fields the original `ask()` flow uses to map chunks → `SourceDto`.
- `RpcReplyMessage`/`SearchReplyMessage` carries `sources()` (already-derived `SourceDto`) **and** `chunks()` (raw payloads). When restoring sources from cache we'll re-derive via `extractSourceDtos(chunks)` to avoid storing two parallel structures.
- Stream-lock and rate-limit semantics: regenerate counts as one streaming session, same per-user lock as `ask()`. A user cannot ask and regenerate simultaneously — by design.
- Tests: add unit test for `ChatService.regenerate` happy path (cache hit) and cache-miss fallback. Frontend: add a UI test that clicking regenerate does not call `appendUser` and does not produce a duplicate user bubble.

## 11. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Chunk cache expires mid-conversation → user loses regenerate determinism | Fallback to RPC re-search; emit `chunksReused: false` in meta so FE could optionally show "(re-retrieved)" |
| Recomp summary returns `""` and downstream prompts feel context-less | Acceptable — Recomp explicitly trades recall for precision; if quality drops, tune by prompt iteration, not by reverting |
| Temperature 0.7 produces hallucinations | Bounded by `Answer using ONLY the Retrieved Context` system rule in `SYSTEM_PROMPT_TEMPLATE`; we are not changing system prompt |
| Schema break on SUMMARIZE callers that expect list | Grep confirms only `entrypoint.py` consumes `summary`; downstream uses it as a string |

## 12. Acceptance criteria

1. Click Regenerate → exactly one assistant bubble replaced; user bubble unchanged in UI and DB.
2. New endpoint `POST /api/conversations/{c}/messages/{a}/regenerate/stream` returns SSE with `meta.regenerated=true`.
3. When chunks cache is warm, no RPC search call observed in logs during regenerate; `chunksReused=true`.
4. Composer textarea disabled and placeholder bubble shows "regenerating…" while regenerate stream is active.
5. `SUMMARIZE_PROMPT_TEMPLATE` accepts `{query}` and `{conversation_history}` and returns `{"summary": "<string>"}`. Caller in `entrypoint.py` extracts the string.
6. `CLARITY_CHECK_PROMPT` returns the same JSON shapes (`clear=true` | `rephrase` | `suggestions`); manual eyeball shows it leans toward CLEAR for borderline queries.
