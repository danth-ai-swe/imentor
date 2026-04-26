# Redis-Backed Chat History & Streaming — Design

**Date:** 2026-04-26
**Branch context:** `feat/java-rabbitmq-react-chat`
**Scope:** Java backend (`backend/`), Python AI service (`src/`), React frontend (`frontend/`)

---

## 1. Goal

Replace the current "Java reads H2 directly + Python fetches history via HTTP every turn" with a Redis-backed hot cache for the active session, while keeping H2 (later: Postgres) as the durable source of truth.

Resulting flow per chat turn:

```
User gửi tin nhắn
  ↓
Java: rate-limit + active-stream check (Redis)
  ↓
Java: persist user message to DB + append to Redis history list
  ↓
Java → Python (RabbitMQ RPC) — Python calls Java HTTP /history-for-ai
                               (Java reads Redis first, falls back to DB)
  ↓
Java: stream tokens from Azure OpenAI → append to Redis stream buffer
  ↓
Stream done → flush full assistant message into DB + into Redis history list
              + delete stream buffer
```

## 2. Non-goals

- Multi-tab live mirroring of the same stream.
- Resuming a dropped stream from Redis (client must reissue on disconnect).
- Auto-recovering partial assistant responses after a Java crash (TTL handles cleanup).
- Replacing H2 with another DB.
- Distributed rate limiting across multiple Java instances (single-instance scope).

## 3. Architecture

### 3.1 Components

| Component | Role |
|---|---|
| **Java backend** (port 8085) | Owns Redis namespace; chat REST/SSE endpoints; rate limit; stream buffering; history-for-ai endpoint |
| **Python AI service** | Calls Java's `/history-for-ai` over HTTP (no direct Redis access) |
| **Redis** | Hot cache for active conversations; rate-limit counters; concurrent-stream lock; transient stream buffer |
| **H2 / DB** | Durable source of truth for users, conversations, messages, chunk sources |
| **React frontend** | Reads `/conversations/{id}/messages` (full schema) for rendering history |

Python does **not** connect to Redis. All Redis access flows through Java. This keeps the Redis schema owned by one service.

### 3.2 Redis key schema

All keys live under the `chat:` prefix. Java owns them.

| Key | Type | Purpose | TTL |
|---|---|---|---|
| `chat:history:{conversationId}` | LIST of JSON | Recent messages (user + assistant) for AI history | 30 min idle (refresh on read/write) |
| `chat:stream:{conversationId}` | STRING | Concatenated assistant tokens during active stream | 5 min |
| `chat:active:{userId}` | STRING (lock) | Concurrent-stream guard (SETNX) | 60 s |
| `chat:rate:{userId}:{minuteBucket}` | INTEGER | Per-minute message counter (INCR) | 60 s |

`{minuteBucket}` = `epochSeconds / 60` (integer floor). Fixed-window algorithm.

History list cap: keep at most **50** entries per conversation in Redis (LTRIM after each push). The `/history-for-ai?limit=20` endpoint will read the tail.

### 3.3 Standardized API schema

All chat endpoints return the wrapper:

```json
{ "success": true, "data": { ... } }
```

Errors use HTTP status + body `{ "success": false, "error": "..." }`.

#### `GET /api/conversations/{id}/messages` — full schema (frontend)

```json
{
  "success": true,
  "data": {
    "conversationId": 123,
    "messages": [
      {
        "id": 1,
        "role": "user",
        "content": "What is whole life insurance?",
        "intent": null,
        "detectedLanguage": null,
        "webSearchUsed": false,
        "stopped": false,
        "createdAt": "2026-04-26T08:30:00Z",
        "sources": []
      }
    ]
  }
}
```

#### `GET /api/conversations/{id}/history-for-ai?limit=20` — gọn cho Python

```json
{
  "success": true,
  "data": {
    "conversationId": 123,
    "messages": [
      { "role": "user",      "content": "...", "intent": null,            "createdAt": "..." },
      { "role": "assistant", "content": "...", "intent": "core_knowledge", "createdAt": "..." }
    ]
  }
}
```

Default `limit=20`. Maximum allowed: 50 (matches Redis cap).

### 3.4 Data flow per chat turn

```
1. POST /api/conversations/{id}/ask/stream  { message: "..." }

2. ChatService.ask:
   a. Resolve userId from conversationId.
   b. Rate-limit check:
      INCR chat:rate:{userId}:{bucket}
      EXPIRE 60 if newly created
      → if count > 15: 429 Too Many Requests, return.
   c. Active-stream lock:
      SET chat:active:{userId} 1 NX EX 60
      → if not acquired: 409 Conflict ("a stream is already active"), return.
   d. Persist user Message to DB (existing code path).
   e. RPUSH chat:history:{convId} <user-msg-json>
      LTRIM chat:history:{convId} -50 -1
      EXPIRE chat:history:{convId} 1800

3. RPC → Python (existing flow). Python calls
   GET /api/conversations/{id}/history-for-ai?limit=20
   Java's handler:
      a. LRANGE chat:history:{convId} -limit -1
      b. If non-empty → EXPIRE 1800; return.
      c. If empty → load last `limit` rows from DB, RPUSH each into Redis
         (warm cache), set EXPIRE 1800; return.

4. Java receives RPC reply (chunks/static), persists empty assistant Message in DB
   (as today). Sends `meta` SSE event.

5. For each token from Azure OpenAI stream:
   - emitter.send delta event
   - APPEND chat:stream:{convId} <token>
   - EXPIRE chat:stream:{convId} 300 (refresh)

6. On stream completion:
   - Use the existing in-memory `StringBuilder buffer` as the authoritative final text
     (Redis stream key is only for crash-safety observability, not the read path).
   - Update assistant Message.content in DB.
   - RPUSH chat:history:{convId} <assistant-msg-json>; LTRIM 50; EXPIRE 1800.
   - DEL chat:stream:{convId}.
   - DEL chat:active:{userId}  (release lock).
   - emitter.complete().

7. On stream error / timeout / client disconnect:
   - Persist whatever is in the in-memory buffer (existing behavior).
   - DEL chat:stream:{convId}.
   - DEL chat:active:{userId}.
```

### 3.5 Cache invalidation

`DELETE /conversations/{convId}/messages/from/{messageId}` (regenerate flow) →
`DEL chat:history:{convId}`. Next read warms from DB. Simpler than reconciling partial deletes inside the list.

## 4. Detailed component changes

### 4.1 Java — new files

```
backend/src/main/java/com/example/chat/
├── config/
│   └── RedisConfig.java          // RedisTemplate<String, String>, connection factory
├── redis/
│   ├── ChatRedisService.java     // history list ops, stream buffer ops, key naming
│   ├── RateLimiter.java          // INCR + EXPIRE per-user-per-minute
│   └── StreamLock.java           // SETNX active-stream lock
├── dto/
│   ├── ApiResponse.java          // generic { success, data } wrapper
│   ├── HistoryForAiDto.java      // { conversationId, messages[] }
│   └── HistoryMessageDto.java    // { role, content, intent, createdAt }
└── chat/
    └── (ChatService, ChatController updated)
```

### 4.2 Java — modified files

- `pom.xml`: add `spring-boot-starter-data-redis`.
- `application.yml`: add `spring.data.redis.host`, `port`, `password` (use env var).
- `ChatController`:
  - Wrap all responses in `ApiResponse<T>`.
  - Add `GET /conversations/{id}/history-for-ai`.
  - On 429/409 from rate limit / lock, return `ResponseEntity` with proper status.
- `ChatService.ask`: insert rate-limit + lock; push to Redis history; append to stream buffer; release lock on all exit paths.
- `ChatService.listMessages`: returns wrapped envelope (controller does the wrap; service returns DTO list).
- `ChatService.deleteFromMessage`: invalidate history Redis key.
- New `ChatService.historyForAi(convId, limit)`: Redis-then-DB read-through.

### 4.3 Python — modified files

- `src/external/fetch_history.py`:
  - Endpoint base unchanged conceptually; deployment changes `CHAT_HISTORY_API_BASE` env var to point at Java backend (`http://localhost:8085` for dev).
  - Path changes from `/api/chats/history-for-ai` → `/api/conversations/{id}/history-for-ai?limit=20`.
  - Drop the `X-Api-Key` header (Java mới không yêu cầu); revisit if/when auth is added.
  - Response parsing: `body["data"]["messages"]` (envelope unchanged in shape).
- `src/rag/search/entrypoint.py::_filter_core_knowledge_pairs`:
  - Read `m["role"]` ("user"/"assistant") instead of `m["sender"]` ("user"/"ai").
  - Read `m["content"]` (string) instead of `m["content"]["text"]`.
  - `intent` field unchanged.

### 4.4 Frontend — modified files

- `frontend/src/api/*` (REST client):
  - `listMessages`, `listConversations`, `createConversation`, `getOrCreateUser`: read `response.data.<field>` instead of top-level.
  - `MessageDto` field names unchanged (already camelCase).

### 4.5 Env / config

`.env` additions / changes:
```
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=admin123    # already present
CHAT_HISTORY_API_BASE=http://localhost:8085   # was http://10.98.36.75:8080
```

`backend/src/main/resources/application.yml`:
```yaml
spring:
  data:
    redis:
      host: ${REDIS_HOST:localhost}
      port: ${REDIS_PORT:6379}
      password: ${REDIS_PASSWORD:}
chat:
  history:
    redis-ttl-seconds: 1800
    redis-max-entries: 50
    default-limit: 20
  stream:
    redis-ttl-seconds: 300
  rate-limit:
    per-user-per-minute: 15
    active-stream-lock-seconds: 60
```

## 5. Error handling

| Failure | Behavior |
|---|---|
| Redis down at startup | Java refuses to start (Spring auto-config fails). Acceptable for demo; document the dependency. |
| Redis down mid-request | Catch `RedisConnectionFailureException` in `ChatRedisService` ops. History reads degrade to DB-only (log warning). Rate-limit + lock failures **fail open** (allow the request) with WARN log — better UX than blanket 500 if Redis flaps. |
| OpenAI stream error | Existing behavior: persist partial buffer with `stopped=true`; release lock; DEL stream key. |
| Client disconnect mid-stream | Existing emitter callbacks: persist partial; release lock; DEL stream key. |
| Java crash mid-stream | Stream buffer expires after 5 min via TTL. Active-stream lock expires after 60 s. No auto-recovery. Conversation history list survives (it's only updated on stream completion). |
| Python `/history-for-ai` 4xx/5xx | Existing `fetch_raw_chat_history` returns `None`; pipeline proceeds without history. Unchanged. |
| Rate-limit triggered | HTTP 429 with `{success: false, error: "rate limit exceeded"}`. Frontend shows toast. |
| Active-stream lock held | HTTP 409 with `{success: false, error: "a response is already streaming"}`. Frontend disables Send while streaming (already does). |

## 6. Testing

**Java unit tests** (`backend/src/test/`):

- `ChatRedisServiceTest`: use `embedded-redis` or testcontainers Redis. Verify push/trim/read, TTL refresh on read, key invalidation.
- `RateLimiterTest`: 15 within 60 s passes; 16th rejected; counter resets after window.
- `StreamLockTest`: SETNX behavior, TTL expiry releases lock.
- `ChatServiceAskTest`: mock Redis service + repos. Verify lock acquired/released on success, error, timeout. Verify history pushed in correct order.
- `ChatControllerHistoryForAiTest`: cache hit returns Redis data; cache miss warms from DB then returns.

**Python unit tests** (`tests/`):

- `test_fetch_history.py`: mock httpx response with new envelope `{success, data: {conversationId, messages}}`. Verify `_filter_core_knowledge_pairs` reads `role`/`content`. Verify backward path returns `[]` on `success=false`.

**Integration smoke** (manual, for demo):

1. Start Redis, RabbitMQ, Python, Java, Vite.
2. Create user/conversation; send 3 messages; verify all three round-trip and frontend renders.
3. Check Redis with `KEYS chat:*`: should see `chat:history:{id}` with 6 entries (3 user + 3 assistant).
4. Send 16 messages in <60 s → 16th returns 429.
5. Send a message; while streaming, send another → 409.
6. Wait 30 min idle → `chat:history:*` keys gone; next message warms from DB.
7. Kill Java mid-stream → `chat:stream:*` and `chat:active:*` expire on TTL; restart Java; new request works.

## 7. Migration / rollout

Single feature branch (`feat/java-rabbitmq-react-chat` or a child branch). All three components ship together — Java schema change forces frontend + Python updates simultaneously. No backward-compat shims (per CLAUDE.md §3).

Rollout order in implementation plan:

1. Add Redis dependency + config + connection sanity check.
2. Java: `ApiResponse` wrapper + update existing controller responses.
3. Frontend: update API client to read `response.data.*`.
4. Java: `ChatRedisService`, `RateLimiter`, `StreamLock`.
5. Java: integrate into `ChatService.ask` (rate-limit, lock, history push, stream buffer, flush).
6. Java: `/history-for-ai` endpoint + warm-from-DB.
7. Python: update `fetch_history.py` URL/path/response shape; update `_filter_core_knowledge_pairs`.
8. Manual smoke per §6.

## 8. Open items deferred

- Authentication / API-key for `/history-for-ai` (Python ↔ Java is internal for now).
- Multi-instance Java rate limiting (Lua script for atomicity if scaled out).
- Stream resume on disconnect (would require Redis Streams + offset tracking).
- Migrating H2 → Postgres (separate project).
