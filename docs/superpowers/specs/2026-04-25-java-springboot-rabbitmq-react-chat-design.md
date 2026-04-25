# Java Spring Boot + RabbitMQ + React Chat — Design Spec

**Date:** 2026-04-25
**Status:** Approved (brainstorming complete)

## Goal

Build a basic chat system with three components alongside the existing Python RAG backend:

1. **Java Spring Boot backend** (`backend/`) — REST + SSE for the React client, RabbitMQ producer/consumer for talking to Python, OpenAI streaming client, Spring Data JPA for persistence (H2 file-based).
2. **Python additions** — reuse `async_pipeline_dispatch` from `src/rag/search/pipeline.py` but skip the answer-generation step; expose it via a RabbitMQ consumer.
3. **React frontend** (`frontend/`) — Vite + TS chat UI with sidebar history, multi-conversation, markdown + syntax-highlighted code blocks, suggested prompts, streaming with typing cursor, edit / regenerate / stop.

Java owns the prompt building and the OpenAI streaming call. Python only does retrieval (chunks + metadata). The split is encoded by a `mode` field in the RabbitMQ reply.

## Architecture

```
┌────────────┐  SSE  ┌──────────────┐  AMQP  ┌──────────┐  ┌──────────────────┐
│   React    │──────→│ Spring Boot  │←──────→│ RabbitMQ │←→│ Python (existing)│
│  (5173)    │       │   (8080)     │        │  (5672)  │  │     (8083)       │
└────────────┘       └──────┬───────┘        └──────────┘  └────────┬─────────┘
                            │                                        │
                            ▼                                        │
                       ┌─────────┐                          ┌────────▼────────┐
                       │ OpenAI  │                          │  Qdrant + RAG   │
                       │ stream  │                          └─────────────────┘
                       └─────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ H2 (file)   │
                     │ JPA         │
                     └─────────────┘
```

## Java ↔ Python contract (RabbitMQ)

### Topology

- Exchange `chat` (topic, durable)
- Queue `chat.search.request` ← bind `search.request` — Python consumes
- Queue `chat.search.reply`   ← bind `search.reply`   — Java consumes
- Message converter: `Jackson2JsonMessageConverter` (both sides)

### Request (`chat.search.request`)

```json
{ "correlationId": "uuid", "userMessage": "...", "conversationId": "..." }
```

Sent by Java with AMQP property `correlation_id` set to the same UUID.

### Reply (`chat.search.reply`)

```json
{
  "mode": "chunks" | "static",
  "intent": "core_knowledge" | "off_topic" | "overall_course_knowledge" | "quiz",
  "detectedLanguage": "Vietnamese",
  "standaloneQuery": "<reflected query>",
  "answerSatisfied": true,
  "webSearchUsed": false,
  "sources": [ {"name":"...","url":"...","pageNumber":1,"totalPages":10} ],
  "chunks": [ {"text":"...","metadata":{"file_name":"...","page_number":1,"course":"...","module_number":"...","lesson_number":""}} ],
  "response": null
}
```

Sent by Python with AMQP property `correlation_id` matching the request.

### Java decision rule

| `mode`   | Java action                                                         |
|----------|---------------------------------------------------------------------|
| `chunks` | Build prompt from `chunks` + `standaloneQuery` + `detectedLanguage`, call OpenAI streaming, forward deltas to SSE. |
| `static` | Stream `response` directly to the SSE client as one (or chunked) `delta` event. No OpenAI call. |

### Branch mapping (which Python branch produces which mode)

| Branch                                            | mode     | response                             | chunks   |
|---------------------------------------------------|----------|--------------------------------------|----------|
| Input too long                                    | `static` | template (`INPUT_TOO_LONG_RESPONSE`) | `[]`     |
| Off-topic intent                                  | `static` | `OFF_TOPIC_RESPONSE_MAP`             | `[]`     |
| Quiz intent                                       | `static` | "Quiz feature is not enabled in this demo" | `[]` |
| Clarity check fails (rephrase / suggestions)      | `static` | clarity message                      | `[]`     |
| Core search — no chunks (web fallback)            | `static` | web answer                           | `[]`     |
| Overall search — no chunks                        | `static` | `NO_RESULT_RESPONSE_MAP`             | `[]`     |
| **Core search — has chunks**                      | `chunks` | `null`                               | `[...]`  |
| **Overall search — has chunks**                   | `chunks` | `null`                               | `[...]`  |

## Python changes

### New file: `src/rag/search/pipeline_chunks.py`

`async_pipeline_dispatch_chunks(user_input, conversation_id) -> dict`:

- Mirrors `async_pipeline_dispatch` (validate, quiz check, intent routing, off-topic, clarity, HyDE, embed, vector search, rerank, neighbor enrich) but **skips** the `_agenerate_answer` call.
- For the two branches that currently generate an answer (`_arun_core_search` core path, `_arun_overall_search` core path), introduce sibling helpers `_arun_core_search_chunks` and `_arun_overall_search_chunks` that perform the retrieval portion only and return `(enriched_chunks, sources, standalone_query)`.
- The original `async_pipeline_dispatch` and FastAPI endpoints stay untouched.
- Returns the dict matching the RabbitMQ reply schema (`mode`, `intent`, `detectedLanguage`, `standaloneQuery`, `answerSatisfied`, `webSearchUsed`, `sources`, `chunks`, `response`).

### New package: `src/messaging/`

- `__init__.py`
- `rabbit_consumer.py`:
  - `aio-pika` consumer for queue `chat.search.request`.
  - For each delivery: parse JSON → call `async_pipeline_dispatch_chunks` → publish JSON reply to exchange `chat` routing key `search.reply` with header `correlation_id`.
  - `prefetch_count = 10`.
  - Exposed as `start_consumer()` (returns the consumer task) and `stop_consumer()`.

### Update `main.py`

- In `lifespan`: after warm-up, `consumer_task = asyncio.create_task(start_consumer())`. On shutdown, `consumer_task.cancel()` then `await` it.

### Update `src/config/app_config.py`

Add:
```python
RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"
```

### Update `requirements.txt`

Add `aio-pika`.

## Java backend (`backend/`)

### Stack

- Spring Boot 3.2, Java 17, Maven.
- Dependencies: `spring-boot-starter-web`, `spring-boot-starter-webflux`, `spring-boot-starter-amqp`, `spring-boot-starter-data-jpa`, `spring-boot-starter-validation`, `com.h2database:h2`, `org.projectlombok:lombok`, `com.fasterxml.jackson.core:jackson-databind`.

### Package structure (`com.example.chat`)

```
ChatApplication.java                # @SpringBootApplication

config/
  RabbitConfig.java                 # exchange + 2 queues + bindings + Jackson converter
  OpenAiConfig.java                 # WebClient bean (base url + auth header)
  WebConfig.java                    # CORS for http://localhost:5173

domain/
  User.java                         # id, username (unique), createdAt
  Conversation.java                 # id, user (ManyToOne), title, createdAt, updatedAt
  Message.java                      # id, conversation (ManyToOne), role, content, intent,
                                    #   detectedLanguage, webSearchUsed, stopped, createdAt
  ChunkSource.java                  # id, message (ManyToOne), name, url, pageNumber, totalPages

repository/
  UserRepository, ConversationRepository, MessageRepository, ChunkSourceRepository

dto/
  CreateUserRequest, UserDto
  CreateConversationRequest, ConversationDto
  MessageDto, AskRequest
  SearchRequestMessage   { correlationId, userMessage, conversationId }
  SearchReplyMessage     { mode, intent, detectedLanguage, standaloneQuery,
                           answerSatisfied, webSearchUsed, sources[], chunks[], response }
  ChunkDto, SourceDto

messaging/
  SearchRpcClient.java              # publish to chat exchange routing key search.request
                                    #   ConcurrentHashMap<correlationId, CompletableFuture<SearchReplyMessage>>
                                    #   timeout = chat.search-timeout-seconds
  SearchReplyListener.java          # @RabbitListener(queues="chat.search.reply") → completes future

openai/
  OpenAiChatService.java            # streamChat(String prompt) → Flux<String>
                                    #   POST /v1/chat/completions with stream:true,
                                    #   parse SSE lines "data: {...}" → emit content deltas
  PromptBuilder.java                # build prompt from chunks + standaloneQuery + detectedLanguage
                                    #   simplified mirror of Python SYSTEM_PROMPT_TEMPLATE

chat/
  ChatController.java
  ChatService.java                  # orchestrates search RPC + OpenAI stream + persistence
```

### REST endpoints

| Method | Path                                                        | Purpose                                    |
|--------|-------------------------------------------------------------|--------------------------------------------|
| POST   | `/api/users`                                                | create user (idempotent on username)       |
| GET    | `/api/users/{username}/conversations`                       | list user's conversations                  |
| POST   | `/api/conversations`                                        | create conversation `{userId, title?}`     |
| GET    | `/api/conversations/{id}/messages`                          | load messages + sources for a conversation |
| POST   | `/api/conversations/{id}/ask/stream`                        | SSE: ask + stream answer                   |
| DELETE | `/api/conversations/{convId}/messages/from/{messageId}`     | delete message and all messages after it (used by edit/regenerate) |

### SSE event shape (matches Python convention)

```
event: meta   data: {"intent":"...","detectedLanguage":"...","sources":[...],"webSearchUsed":false,"assistantMessageId":42}
event: delta  data: {"content":"<token>"}
event: done   data: {}
event: error  data: {"message":"..."}
```

`assistantMessageId` is included so the frontend can target it for regenerate.

### `/ask/stream` flow

1. Validate conversation exists; persist user `Message`.
2. `correlationId = UUID.randomUUID()`; register `CompletableFuture<SearchReplyMessage>` in `SearchRpcClient`.
3. Publish `SearchRequestMessage` to exchange `chat` routing key `search.request` (AMQP `correlation_id` set).
4. `future.get(chat.search-timeout-seconds, SECONDS)` → `SearchReplyMessage`.
5. Create empty assistant `Message`, get its id.
6. Emit `meta` event (intent, detectedLanguage, sources, webSearchUsed, assistantMessageId).
7. Branch on `mode`:
   - `static`: emit single `delta {content: response}` → `done`. Persist `response` as the assistant message content; persist `ChunkSource[]` if `sources` non-empty.
   - `chunks`: `prompt = PromptBuilder.build(chunks, standaloneQuery, detectedLanguage)` → `OpenAiChatService.streamChat(prompt)`:
     - For each `String token` in Flux: emit `delta {content: token}`, append to `StringBuilder`.
     - `doOnComplete`: emit `done`; persist accumulated content + `ChunkSource[]`.
     - `doOnError`: emit `error`; persist accumulated content (if non-empty) with `stopped=true`.
8. Return `SseEmitter` (timeout 60s).

### Stop streaming behavior

- Frontend calls `AbortController.abort()` on the fetch reader → TCP closes.
- Spring `SseEmitter.onError` / `onCompletion` fires.
- `ChatService` keeps a `Disposable` reference to the OpenAI subscription and disposes it.
- Accumulated content in the `StringBuilder` is persisted with `stopped=true` (skip if empty).

### Edit / regenerate (frontend-orchestrated)

Both flows reuse `/ask/stream`:

- **Edit user message X with new content**:
  1. `DELETE /api/conversations/{convId}/messages/from/{X}` → cascade-deletes X and all later messages (and their `ChunkSource`).
  2. `POST /api/conversations/{convId}/ask/stream` with the new content.

- **Regenerate assistant message Y (preceded by user message X)**:
  1. `DELETE /api/conversations/{convId}/messages/from/{Y}` → cascade-deletes Y and any following messages.
  2. `POST /api/conversations/{convId}/ask/stream` with the existing content of X (frontend already has it).

### RabbitMQ topology (declared in `RabbitConfig`)

- Exchange `chat` (topic, durable)
- Queue `chat.search.request` (durable) ← binding `search.request`
- Queue `chat.search.reply` (durable) ← binding `search.reply`
- `Jackson2JsonMessageConverter`

### `application.yml`

```yaml
server:
  port: 8080
spring:
  datasource:
    url: jdbc:h2:file:./data/chatdb;AUTO_SERVER=TRUE
    driverClassName: org.h2.Driver
    username: sa
    password:
  jpa:
    hibernate:
      ddl-auto: update
    show-sql: false
  rabbitmq:
    host: localhost
    port: 5672
    username: guest
    password: guest
  h2:
    console:
      enabled: true
openai:
  api-key: ${OPENAI_API_KEY}
  base-url: https://api.openai.com
  model: gpt-4o-mini
chat:
  search-timeout-seconds: 30
```

## React frontend (`frontend/`)

### Stack

- Vite + React 18 + TypeScript + plain CSS.
- Dependencies:
  - `react`, `react-dom`
  - `vite`, `@vitejs/plugin-react`
  - `typescript`, `@types/react`, `@types/react-dom`
  - `react-markdown`, `remark-gfm`
  - `react-syntax-highlighter` + `@types/react-syntax-highlighter`

### File structure

```
frontend/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts                 # proxy /api → http://localhost:8080
└── src/
    ├── main.tsx
    ├── App.tsx                    # layout: <Sidebar /> + <ChatWindow />
    ├── App.css
    ├── api/
    │   └── chatApi.ts             # createUser, listConversations, createConversation,
    │                              #   loadMessages, deleteFromMessage,
    │                              #   askStream(convId, message, handlers, abortSignal)
    ├── components/
    │   ├── Sidebar.tsx            # username input (1st time) + list conversations + "New Chat"
    │   ├── ChatWindow.tsx         # message list + input box + send/stop button
    │   ├── MessageBubble.tsx      # markdown + code blocks + sources + edit/regenerate hover
    │   └── SuggestedPrompts.tsx   # 4 hardcoded prompts shown when conversation is empty
    ├── hooks/
    │   └── useChatStream.ts       # manages fetch+ReadableStream SSE, AbortController,
    │                              #   accumulates delta, exposes onMeta/onDelta/onDone/onError
    └── types.ts                   # Conversation, Message, Source, SseEvent types
```

### UI flow

1. **First visit**: `localStorage` has no `userId` → Sidebar shows username input → POST `/api/users` → save `{userId, username}` to localStorage.
2. Sidebar loads `GET /api/users/{username}/conversations`.
3. Click "+ New Chat" → POST `/api/conversations` → select new conversation. Empty conversation shows `<SuggestedPrompts />`.
4. Click a conversation → `GET /api/conversations/{id}/messages` → render in ChatWindow.
5. User submits a message:
   - Optimistically append user message to the list.
   - Append empty placeholder assistant message with a typing cursor (`▋` blink animation).
   - Open SSE via `fetch` + `ReadableStream` reader (POST + AbortController).
   - On `meta`: set `assistantMessageId` + `sources` on the placeholder.
   - On `delta`: append `content` to the placeholder.
   - On `done`: remove cursor, mark complete.
   - On `error`: render inline error.
6. **Suggested prompt click**: fills input + auto-submits.
7. **Edit user message**: hover → "✎ Edit" → inline `<textarea>` → Save calls `deleteFromMessage(X)` then `askStream(newContent)`.
8. **Regenerate**: hover last assistant message → "↻ Regenerate" → calls `deleteFromMessage(Y)` then `askStream(precedingUserContent)`.
9. **Stop streaming**: while streaming, "Send" button becomes "■ Stop" → click calls `controller.abort()`. Backend persists accumulated content as the assistant message with `stopped=true`.

### Markdown rendering

`MessageBubble` renders `content` through `<ReactMarkdown remarkPlugins={[remarkGfm]} components={{ code: CodeBlock }}>`. `CodeBlock` uses `react-syntax-highlighter` (Prism + `oneDark` theme) for fenced code blocks; inline code uses a `<code>` with monospace style.

### Suggested prompts (hardcoded for the LOMA insurance domain)

```ts
const SUGGESTED_PROMPTS = [
  "What is adverse selection in insurance?",
  "Explain the law of large numbers.",
  "What is reinsurance and why is it important?",
  "List the main types of insurance contracts.",
];
```

### `vite.config.ts` proxy

```ts
server: {
  port: 5173,
  proxy: { '/api': 'http://localhost:8080' }
}
```

## Out of scope (YAGNI)

- Authentication / passwords (username string only).
- Conversation rename / delete (only "delete from message" for edit/regenerate).
- Dark mode toggle.
- Docker compose (RabbitMQ already running locally).
- Migration tooling (H2 + `ddl-auto=update` is enough for the demo).

## Success criteria

- `cd backend && ./mvnw spring-boot:run` boots Spring Boot on 8080, declares RabbitMQ topology, opens a JPA connection to H2 file at `./data/chatdb`.
- `python main.py` keeps existing FastAPI behavior **and** also subscribes to `chat.search.request`.
- `cd frontend && npm install && npm run dev` serves the React app on 5173 with `/api` proxied.
- End-to-end: open `http://localhost:5173`, enter a username, create a chat, click a suggested prompt → message renders with markdown + code highlighting + sources, streaming visible token-by-token.
- Edit / regenerate / stop all work (verified manually).
- After full restart of all three processes, conversations + messages persist (H2 file).
