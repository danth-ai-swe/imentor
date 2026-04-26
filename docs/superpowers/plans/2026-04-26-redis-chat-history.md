# Redis-Backed Chat History & Streaming — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Redis hot cache for active conversations, per-user rate limiting, an active-stream lock, a transient stream buffer, and a standardized `{success, data}` API envelope across Java/Python/Frontend — implementing the design in `docs/superpowers/specs/2026-04-26-redis-chat-history-design.md`.

**Architecture:** Java owns all Redis access (key prefix `chat:`). Python keeps calling Java over HTTP for history; Java does Redis-then-DB read-through. User messages still persist to DB synchronously; assistant tokens append to a Redis stream buffer during streaming and the final assistant message is flushed to DB + Redis history list on completion.

**Tech Stack:** Spring Boot 3.2.5 (Java 17), Spring Data Redis (Lettuce), Redis 7+, FastAPI (Python 3.11+), httpx, React 18 + TypeScript + Vite. Test stack: JUnit 5 + Mockito + AssertJ (Java); pytest + unittest.mock (Python).

---

## File Structure

**Java — new files:**
- `backend/src/main/java/com/example/chat/config/RedisConfig.java` — `RedisTemplate<String, String>` bean
- `backend/src/main/java/com/example/chat/dto/ApiResponse.java` — generic `{success, data}` envelope
- `backend/src/main/java/com/example/chat/dto/ApiError.java` — generic `{success: false, error}` shape
- `backend/src/main/java/com/example/chat/dto/HistoryMessageDto.java` — `{role, content, intent, createdAt}`
- `backend/src/main/java/com/example/chat/dto/HistoryForAiResponse.java` — `{conversationId, messages[]}`
- `backend/src/main/java/com/example/chat/dto/MessagesResponse.java` — `{conversationId, messages[]}` (full schema for FE)
- `backend/src/main/java/com/example/chat/redis/ChatRedisService.java` — history list ops + stream buffer ops
- `backend/src/main/java/com/example/chat/redis/RateLimiter.java` — fixed-window per-minute INCR
- `backend/src/main/java/com/example/chat/redis/StreamLock.java` — SETNX active-stream lock
- `backend/src/main/java/com/example/chat/chat/ChatExceptionHandler.java` — `@RestControllerAdvice` mapping to `ApiError`

**Java — modified files:**
- `backend/pom.xml` — add `spring-boot-starter-data-redis`
- `backend/src/main/resources/application.yml` — Redis config + chat config block
- `backend/src/main/java/com/example/chat/chat/ChatController.java` — wrap responses + new `/history-for-ai` endpoint
- `backend/src/main/java/com/example/chat/chat/ChatService.java` — integrate Redis (rate-limit, lock, history push, stream buffer, invalidation)

**Java — new tests:**
- `backend/src/test/java/com/example/chat/redis/RateLimiterTest.java`
- `backend/src/test/java/com/example/chat/redis/StreamLockTest.java`
- `backend/src/test/java/com/example/chat/redis/ChatRedisServiceTest.java`
- `backend/src/test/java/com/example/chat/chat/ChatServiceHistoryForAiTest.java`

**Python — modified files:**
- `src/external/fetch_history.py` — endpoint path + envelope parsing
- `src/rag/search/entrypoint.py::_filter_core_knowledge_pairs` — `role`/`content` instead of `sender`/`content.text`
- `.env` — `CHAT_HISTORY_API_BASE=http://localhost:8085`

**Python — new tests:**
- `tests/test_fetch_history.py`

**Frontend — modified files:**
- `frontend/src/api/chatApi.ts` — read `response.data.*` for all wrapped endpoints

---

## Task 1: Add Spring Data Redis dependency, config, and connection check

**Files:**
- Modify: `backend/pom.xml`
- Modify: `backend/src/main/resources/application.yml`
- Create: `backend/src/main/java/com/example/chat/config/RedisConfig.java`

- [ ] **Step 1: Add Redis dependency to `backend/pom.xml`**

Add inside `<dependencies>` after the existing `spring-boot-starter-amqp` block:

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>
```

- [ ] **Step 2: Add Redis + chat config to `application.yml`**

Append to existing `spring:` block (under `rabbitmq:`):

```yaml
  data:
    redis:
      host: ${REDIS_HOST:localhost}
      port: ${REDIS_PORT:6379}
      password: ${REDIS_PASSWORD:}
      timeout: 2000ms
```

Replace the existing `chat:` block with:

```yaml
chat:
  search-timeout-seconds: 30
  history:
    redis-ttl-seconds: 1800
    redis-max-entries: 50
    default-limit: 20
    max-limit: 50
  stream:
    redis-ttl-seconds: 300
  rate-limit:
    per-user-per-minute: 15
    active-stream-lock-seconds: 60
```

- [ ] **Step 3: Create `RedisConfig.java`**

```java
package com.example.chat.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.StringRedisTemplate;

@Configuration
public class RedisConfig {

    @Bean
    public StringRedisTemplate stringRedisTemplate(RedisConnectionFactory cf) {
        return new StringRedisTemplate(cf);
    }
}
```

We use `StringRedisTemplate` (specialization of `RedisTemplate<String, String>`) — every value we store is a JSON string or a counter, both fit the String type cleanly.

- [ ] **Step 4: Build and verify Spring Boot starts**

Run:
```bash
cd backend && ./mvnw -q -DskipTests compile
```
Expected: `BUILD SUCCESS`.

Run (assume Redis is up at `localhost:6379` per docker-compose, password `admin123` from `.env`):
```bash
REDIS_PASSWORD=admin123 ./mvnw -q spring-boot:run
```
Expected: log line like `Connecting to Lettuce Redis at localhost:6379`. Stop with Ctrl+C.

If Redis is not up, expected: `RedisConnectionFailureException`. Bring Redis up first.

- [ ] **Step 5: Commit**

```bash
git add backend/pom.xml backend/src/main/resources/application.yml backend/src/main/java/com/example/chat/config/RedisConfig.java
git commit -m "chore(backend): add spring-data-redis + StringRedisTemplate config"
```

---

## Task 2: Standardize API responses with `{success, data}` envelope

**Files:**
- Create: `backend/src/main/java/com/example/chat/dto/ApiResponse.java`
- Create: `backend/src/main/java/com/example/chat/dto/ApiError.java`
- Create: `backend/src/main/java/com/example/chat/dto/MessagesResponse.java`
- Create: `backend/src/main/java/com/example/chat/chat/ChatExceptionHandler.java`
- Modify: `backend/src/main/java/com/example/chat/chat/ChatController.java`
- Modify: `backend/src/main/java/com/example/chat/chat/ChatService.java` (add `listMessagesEnvelope` helper)
- Modify: `frontend/src/api/chatApi.ts`

- [ ] **Step 1: Create `ApiResponse.java`**

```java
package com.example.chat.dto;

public record ApiResponse<T>(boolean success, T data) {
    public static <T> ApiResponse<T> ok(T data) {
        return new ApiResponse<>(true, data);
    }
}
```

- [ ] **Step 2: Create `ApiError.java`**

```java
package com.example.chat.dto;

public record ApiError(boolean success, String error) {
    public static ApiError of(String message) {
        return new ApiError(false, message);
    }
}
```

- [ ] **Step 3: Create `MessagesResponse.java`**

```java
package com.example.chat.dto;

import java.util.List;

public record MessagesResponse(Long conversationId, List<MessageDto> messages) {}
```

- [ ] **Step 4: Create `ChatExceptionHandler.java`**

```java
package com.example.chat.chat;

import com.example.chat.dto.ApiError;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.server.ResponseStatusException;

@RestControllerAdvice
public class ChatExceptionHandler {

    @ExceptionHandler(ResponseStatusException.class)
    public ResponseEntity<ApiError> handleStatus(ResponseStatusException e) {
        String msg = e.getReason() != null ? e.getReason() : e.getMessage();
        return ResponseEntity.status(e.getStatusCode()).body(ApiError.of(msg));
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ApiError> handleBadArg(IllegalArgumentException e) {
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(ApiError.of(e.getMessage()));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiError> handleOther(Exception e) {
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
            .body(ApiError.of(e.getMessage() != null ? e.getMessage() : "internal error"));
    }
}
```

- [ ] **Step 5: Update `ChatService.listMessages` to return `MessagesResponse`**

In `ChatService.java`, replace the existing `listMessages` method body so it returns the envelope payload (the controller will wrap):

```java
@Transactional(readOnly = true)
public MessagesResponse listMessages(Long conversationId) {
    List<MessageDto> messageDtos = messages.findByConversationIdOrderByCreatedAtAsc(conversationId).stream()
        .map(m -> {
            List<SourceDto> sources = chunkSources.findByMessageId(m.getId()).stream()
                .map(s -> new SourceDto(s.getName(), s.getUrl(), s.getPageNumber(), s.getTotalPages()))
                .toList();
            return new MessageDto(
                m.getId(), m.getRole(), m.getContent(), m.getIntent(),
                m.getDetectedLanguage(), m.isWebSearchUsed(), m.isStopped(),
                m.getCreatedAt(), sources
            );
        }).toList();
    return new MessagesResponse(conversationId, messageDtos);
}
```

- [ ] **Step 6: Update `ChatController` to wrap responses**

Replace the body of `ChatController.java`:

```java
package com.example.chat.chat;

import com.example.chat.dto.*;
import jakarta.validation.Valid;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.List;

@RestController
@RequestMapping("/api")
public class ChatController {

    private final ChatService service;

    public ChatController(ChatService service) {
        this.service = service;
    }

    @PostMapping("/users")
    public ApiResponse<UserDto> createUser(@Valid @RequestBody CreateUserRequest req) {
        return ApiResponse.ok(service.getOrCreateUser(req.username()));
    }

    @GetMapping("/users/{username}/conversations")
    public ApiResponse<List<ConversationDto>> listConversations(@PathVariable String username) {
        return ApiResponse.ok(service.listConversations(username));
    }

    @PostMapping("/conversations")
    public ApiResponse<ConversationDto> createConversation(@Valid @RequestBody CreateConversationRequest req) {
        return ApiResponse.ok(service.createConversation(req.userId(), req.title()));
    }

    @GetMapping("/conversations/{id}/messages")
    public ApiResponse<MessagesResponse> listMessages(@PathVariable Long id) {
        return ApiResponse.ok(service.listMessages(id));
    }

    @DeleteMapping("/conversations/{convId}/messages/from/{messageId}")
    public ResponseEntity<Void> deleteFromMessage(@PathVariable Long convId, @PathVariable Long messageId) {
        service.deleteFromMessage(convId, messageId);
        return ResponseEntity.noContent().build();
    }

    @PostMapping(path = "/conversations/{id}/ask/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter ask(@PathVariable Long id, @Valid @RequestBody AskRequest req) {
        return service.ask(id, req.message());
    }
}
```

(SSE endpoint stays untouched — SSE event format is its own protocol, not JSON envelope.)

- [ ] **Step 7: Update frontend `chatApi.ts` to read `data.*`**

Replace `frontend/src/api/chatApi.ts`:

```ts
import type { Conversation, Message, SseHandler, User } from '../types';

const BASE = '/api';

interface ApiResponse<T> { success: boolean; data: T; error?: string; }

async function unwrap<T>(res: Response): Promise<T> {
  const body = (await res.json()) as ApiResponse<T>;
  if (!body.success) throw new Error(body.error ?? `${res.status}`);
  return body.data;
}

export async function createUser(username: string): Promise<User> {
  const res = await fetch(`${BASE}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username }),
  });
  return unwrap<User>(res);
}

export async function listConversations(username: string): Promise<Conversation[]> {
  const res = await fetch(`${BASE}/users/${encodeURIComponent(username)}/conversations`);
  return unwrap<Conversation[]>(res);
}

export async function createConversation(userId: number, title?: string): Promise<Conversation> {
  const res = await fetch(`${BASE}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userId, title }),
  });
  return unwrap<Conversation>(res);
}

export async function loadMessages(conversationId: number): Promise<Message[]> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/messages`);
  const wrapper = await unwrap<{ conversationId: number; messages: Message[] }>(res);
  return wrapper.messages;
}

export async function deleteFromMessage(conversationId: number, messageId: number): Promise<void> {
  await fetch(`${BASE}/conversations/${conversationId}/messages/from/${messageId}`, { method: 'DELETE' });
}

export async function askStream(
  conversationId: number,
  message: string,
  handlers: SseHandler,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ message }),
    signal,
  });
  if (!res.body) throw new Error('No response body');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent = 'message';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let nl;
      while ((nl = buffer.indexOf('\n')) !== -1) {
        const rawLine = buffer.slice(0, nl).replace(/\r$/, '');
        buffer = buffer.slice(nl + 1);

        if (rawLine === '') { currentEvent = 'message'; continue; }
        if (rawLine.startsWith('event:')) { currentEvent = rawLine.slice(6).trim(); continue; }
        if (rawLine.startsWith('data:')) {
          const data = rawLine.slice(5).trim();
          if (!data) continue;
          try {
            const parsed = JSON.parse(data);
            if (currentEvent === 'meta') handlers.onMeta(parsed);
            else if (currentEvent === 'delta') handlers.onDelta(parsed.content ?? '');
            else if (currentEvent === 'done') handlers.onDone();
            else if (currentEvent === 'error') handlers.onError(parsed.message ?? 'unknown error');
          } catch { /* skip malformed line */ }
        }
      }
    }
  } catch (e: any) {
    if (e.name !== 'AbortError') handlers.onError(String(e.message ?? e));
  }
}
```

- [ ] **Step 8: Build Java + Frontend**

```bash
cd backend && ./mvnw -q -DskipTests compile
cd ../frontend && npm run build
```
Expected: both succeed.

- [ ] **Step 9: Commit**

```bash
git add backend/src/main/java/com/example/chat/dto/ApiResponse.java \
        backend/src/main/java/com/example/chat/dto/ApiError.java \
        backend/src/main/java/com/example/chat/dto/MessagesResponse.java \
        backend/src/main/java/com/example/chat/chat/ChatExceptionHandler.java \
        backend/src/main/java/com/example/chat/chat/ChatController.java \
        backend/src/main/java/com/example/chat/chat/ChatService.java \
        frontend/src/api/chatApi.ts
git commit -m "feat(api): standardize {success,data} envelope across Java + frontend"
```

---

## Task 3: HistoryMessageDto + ChatRedisService (history list ops)

**Files:**
- Create: `backend/src/main/java/com/example/chat/dto/HistoryMessageDto.java`
- Create: `backend/src/main/java/com/example/chat/dto/HistoryForAiResponse.java`
- Create: `backend/src/main/java/com/example/chat/redis/ChatRedisService.java`
- Create: `backend/src/test/java/com/example/chat/redis/ChatRedisServiceTest.java`

- [ ] **Step 1: Create `HistoryMessageDto.java`**

```java
package com.example.chat.dto;

import java.time.Instant;

public record HistoryMessageDto(
    String role,
    String content,
    String intent,
    Instant createdAt
) {}
```

- [ ] **Step 2: Create `HistoryForAiResponse.java`**

```java
package com.example.chat.dto;

import java.util.List;

public record HistoryForAiResponse(Long conversationId, List<HistoryMessageDto> messages) {}
```

- [ ] **Step 3: Write failing test `ChatRedisServiceTest.java`**

```java
package com.example.chat.redis;

import com.example.chat.dto.HistoryMessageDto;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.data.redis.core.ListOperations;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.time.Duration;
import java.time.Instant;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class ChatRedisServiceTest {

    private StringRedisTemplate template;
    private ListOperations<String, String> listOps;
    private ValueOperations<String, String> valueOps;
    private ChatRedisService service;
    private ObjectMapper mapper;

    @BeforeEach
    void setUp() {
        template = mock(StringRedisTemplate.class);
        listOps = mock(ListOperations.class);
        valueOps = mock(ValueOperations.class);
        when(template.opsForList()).thenReturn(listOps);
        when(template.opsForValue()).thenReturn(valueOps);
        mapper = new ObjectMapper().registerModule(new JavaTimeModule());
        service = new ChatRedisService(template, mapper, 1800, 50, 300);
    }

    @Test
    void appendsHistoryAndTrimsAndRefreshesTtl() {
        HistoryMessageDto msg = new HistoryMessageDto("user", "hi", null, Instant.parse("2026-04-26T00:00:00Z"));

        service.appendHistory(123L, msg);

        ArgumentCaptor<String> json = ArgumentCaptor.forClass(String.class);
        verify(listOps).rightPush(eq("chat:history:123"), json.capture());
        assertThat(json.getValue()).contains("\"role\":\"user\"").contains("\"content\":\"hi\"");
        verify(listOps).trim("chat:history:123", -50, -1);
        verify(template).expire("chat:history:123", Duration.ofSeconds(1800));
    }

    @Test
    void readsTailAndRefreshesTtlOnHit() throws Exception {
        HistoryMessageDto m1 = new HistoryMessageDto("user", "hi", null, Instant.parse("2026-04-26T00:00:00Z"));
        when(listOps.range("chat:history:123", -20, -1))
            .thenReturn(List.of(mapper.writeValueAsString(m1)));

        List<HistoryMessageDto> result = service.readHistoryTail(123L, 20);

        assertThat(result).hasSize(1);
        assertThat(result.get(0).role()).isEqualTo("user");
        verify(template).expire("chat:history:123", Duration.ofSeconds(1800));
    }

    @Test
    void emptyOnMissAndDoesNotRefreshTtl() {
        when(listOps.range("chat:history:999", -20, -1)).thenReturn(List.of());

        List<HistoryMessageDto> result = service.readHistoryTail(999L, 20);

        assertThat(result).isEmpty();
        verify(template, never()).expire(eq("chat:history:999"), any());
    }

    @Test
    void invalidateDeletesKey() {
        service.invalidateHistory(123L);
        verify(template).delete("chat:history:123");
    }

    @Test
    void appendStreamTokenCreatesAndRefreshesKey() {
        service.appendStreamToken(123L, "hello");
        verify(valueOps).append("chat:stream:123", "hello");
        verify(template).expire("chat:stream:123", Duration.ofSeconds(300));
    }

    @Test
    void deleteStreamRemovesKey() {
        service.deleteStream(123L);
        verify(template).delete("chat:stream:123");
    }
}
```

- [ ] **Step 4: Run test to verify it fails**

```bash
cd backend && ./mvnw -q -Dtest=ChatRedisServiceTest test
```
Expected: FAIL — `ChatRedisService` not found.

- [ ] **Step 5: Implement `ChatRedisService.java`**

```java
package com.example.chat.redis;

import com.example.chat.dto.HistoryMessageDto;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DataAccessException;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.List;

@Service
public class ChatRedisService {

    private static final Logger log = LoggerFactory.getLogger(ChatRedisService.class);

    private final StringRedisTemplate template;
    private final ObjectMapper mapper;
    private final long historyTtlSeconds;
    private final int historyMaxEntries;
    private final long streamTtlSeconds;

    public ChatRedisService(
        StringRedisTemplate template,
        ObjectMapper mapper,
        @Value("${chat.history.redis-ttl-seconds:1800}") long historyTtlSeconds,
        @Value("${chat.history.redis-max-entries:50}") int historyMaxEntries,
        @Value("${chat.stream.redis-ttl-seconds:300}") long streamTtlSeconds
    ) {
        this.template = template;
        this.mapper = mapper;
        this.historyTtlSeconds = historyTtlSeconds;
        this.historyMaxEntries = historyMaxEntries;
        this.streamTtlSeconds = streamTtlSeconds;
    }

    public void appendHistory(Long conversationId, HistoryMessageDto msg) {
        String key = historyKey(conversationId);
        try {
            String json = mapper.writeValueAsString(msg);
            template.opsForList().rightPush(key, json);
            template.opsForList().trim(key, -historyMaxEntries, -1);
            template.expire(key, Duration.ofSeconds(historyTtlSeconds));
        } catch (JsonProcessingException e) {
            log.warn("Failed to serialize HistoryMessageDto for conv={}: {}", conversationId, e.getMessage());
        } catch (DataAccessException e) {
            log.warn("Redis appendHistory failed for conv={}: {}", conversationId, e.getMessage());
        }
    }

    public List<HistoryMessageDto> readHistoryTail(Long conversationId, int limit) {
        String key = historyKey(conversationId);
        try {
            List<String> jsons = template.opsForList().range(key, -limit, -1);
            if (jsons == null || jsons.isEmpty()) return List.of();
            template.expire(key, Duration.ofSeconds(historyTtlSeconds));
            return jsons.stream().map(this::parseSafe).filter(java.util.Objects::nonNull).toList();
        } catch (DataAccessException e) {
            log.warn("Redis readHistoryTail failed for conv={}: {}", conversationId, e.getMessage());
            return List.of();
        }
    }

    public void invalidateHistory(Long conversationId) {
        try {
            template.delete(historyKey(conversationId));
        } catch (DataAccessException e) {
            log.warn("Redis invalidateHistory failed for conv={}: {}", conversationId, e.getMessage());
        }
    }

    public void appendStreamToken(Long conversationId, String token) {
        String key = streamKey(conversationId);
        try {
            template.opsForValue().append(key, token);
            template.expire(key, Duration.ofSeconds(streamTtlSeconds));
        } catch (DataAccessException e) {
            log.warn("Redis appendStreamToken failed for conv={}: {}", conversationId, e.getMessage());
        }
    }

    public void deleteStream(Long conversationId) {
        try {
            template.delete(streamKey(conversationId));
        } catch (DataAccessException e) {
            log.warn("Redis deleteStream failed for conv={}: {}", conversationId, e.getMessage());
        }
    }

    private HistoryMessageDto parseSafe(String json) {
        try { return mapper.readValue(json, HistoryMessageDto.class); }
        catch (JsonProcessingException e) {
            log.warn("Skipping malformed history JSON: {}", e.getMessage());
            return null;
        }
    }

    private static String historyKey(Long convId) { return "chat:history:" + convId; }
    private static String streamKey(Long convId)  { return "chat:stream:"  + convId; }
}
```

- [ ] **Step 6: Add `JavaTimeModule` to `ObjectMapper`**

`ObjectMapper` injected by Spring auto-config doesn't always include JSR-310. Add a config bean. Append to `RedisConfig.java`:

```java
    @Bean
    public com.fasterxml.jackson.databind.Module javaTimeModule() {
        return new com.fasterxml.jackson.datatype.jsr310.JavaTimeModule();
    }
```

This registers the JSR-310 module globally, so `Instant` serializes as ISO-8601 strings in both Jackson and our Redis service.

- [ ] **Step 7: Run test to verify it passes**

```bash
cd backend && ./mvnw -q -Dtest=ChatRedisServiceTest test
```
Expected: PASS (6 tests).

- [ ] **Step 8: Commit**

```bash
git add backend/src/main/java/com/example/chat/dto/HistoryMessageDto.java \
        backend/src/main/java/com/example/chat/dto/HistoryForAiResponse.java \
        backend/src/main/java/com/example/chat/redis/ChatRedisService.java \
        backend/src/main/java/com/example/chat/config/RedisConfig.java \
        backend/src/test/java/com/example/chat/redis/ChatRedisServiceTest.java
git commit -m "feat(redis): ChatRedisService for history list + stream buffer ops"
```

---

## Task 4: GET /history-for-ai endpoint with cache-aside

**Files:**
- Modify: `backend/src/main/java/com/example/chat/chat/ChatService.java` (add `historyForAi` method)
- Modify: `backend/src/main/java/com/example/chat/chat/ChatController.java` (add new endpoint)
- Create: `backend/src/test/java/com/example/chat/chat/ChatServiceHistoryForAiTest.java`

- [ ] **Step 1: Write failing test `ChatServiceHistoryForAiTest.java`**

```java
package com.example.chat.chat;

import com.example.chat.domain.Conversation;
import com.example.chat.domain.Message;
import com.example.chat.dto.HistoryForAiResponse;
import com.example.chat.dto.HistoryMessageDto;
import com.example.chat.openai.OpenAiChatService;
import com.example.chat.openai.PromptBuilder;
import com.example.chat.messaging.SearchRpcClient;
import com.example.chat.redis.ChatRedisService;
import com.example.chat.redis.RateLimiter;
import com.example.chat.redis.StreamLock;
import com.example.chat.repository.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

class ChatServiceHistoryForAiTest {

    private MessageRepository messages;
    private ConversationRepository conversations;
    private ChatRedisService redis;
    private ChatService service;

    @BeforeEach
    void setUp() {
        UserRepository users = mock(UserRepository.class);
        conversations = mock(ConversationRepository.class);
        messages = mock(MessageRepository.class);
        ChunkSourceRepository chunks = mock(ChunkSourceRepository.class);
        SearchRpcClient rpc = mock(SearchRpcClient.class);
        OpenAiChatService openai = mock(OpenAiChatService.class);
        PromptBuilder pb = mock(PromptBuilder.class);
        redis = mock(ChatRedisService.class);
        RateLimiter rl = mock(RateLimiter.class);
        StreamLock sl = mock(StreamLock.class);

        service = new ChatService(users, conversations, messages, chunks, rpc, openai, pb, redis, rl, sl);
    }

    @Test
    void cacheHitReturnsRedisMessages() {
        HistoryMessageDto cached = new HistoryMessageDto("user", "cached", null, Instant.now());
        when(redis.readHistoryTail(123L, 20)).thenReturn(List.of(cached));

        HistoryForAiResponse resp = service.historyForAi(123L, 20);

        assertThat(resp.conversationId()).isEqualTo(123L);
        assertThat(resp.messages()).hasSize(1);
        assertThat(resp.messages().get(0).content()).isEqualTo("cached");
        verify(messages, never()).findByConversationIdOrderByCreatedAtAsc(any());
    }

    @Test
    void cacheMissWarmsFromDbAndReturns() {
        when(redis.readHistoryTail(123L, 20)).thenReturn(List.of());
        Conversation conv = Conversation.builder().id(123L).build();
        when(conversations.findById(123L)).thenReturn(Optional.of(conv));

        Message m1 = Message.builder().id(1L).conversation(conv).role("user").content("from db").createdAt(Instant.parse("2026-04-26T00:00:00Z")).build();
        when(messages.findByConversationIdOrderByCreatedAtAsc(123L)).thenReturn(List.of(m1));

        HistoryForAiResponse resp = service.historyForAi(123L, 20);

        assertThat(resp.messages()).hasSize(1);
        assertThat(resp.messages().get(0).content()).isEqualTo("from db");
        verify(redis).appendHistory(eq(123L), any(HistoryMessageDto.class));
    }

    @Test
    void cacheMissWithMissingConvReturnsEmpty() {
        when(redis.readHistoryTail(999L, 20)).thenReturn(List.of());
        when(conversations.findById(999L)).thenReturn(Optional.empty());

        HistoryForAiResponse resp = service.historyForAi(999L, 20);

        assertThat(resp.messages()).isEmpty();
        verify(messages, never()).findByConversationIdOrderByCreatedAtAsc(any());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && ./mvnw -q -Dtest=ChatServiceHistoryForAiTest test
```
Expected: FAIL — `ChatService` constructor signature mismatch + `historyForAi` not defined + `RateLimiter`/`StreamLock` not found.

(`RateLimiter` and `StreamLock` will be created in Tasks 5 and 6. For now, create stub classes so this test compiles. The actual implementations come next.)

- [ ] **Step 3: Create stub `RateLimiter.java` and `StreamLock.java`**

`backend/src/main/java/com/example/chat/redis/RateLimiter.java`:
```java
package com.example.chat.redis;

import org.springframework.stereotype.Service;

@Service
public class RateLimiter {
    public boolean tryAcquire(Long userId) { return true; }
}
```

`backend/src/main/java/com/example/chat/redis/StreamLock.java`:
```java
package com.example.chat.redis;

import org.springframework.stereotype.Service;

@Service
public class StreamLock {
    public boolean acquire(Long userId) { return true; }
    public void release(Long userId) {}
}
```

- [ ] **Step 4: Update `ChatService` constructor + add `historyForAi`**

In `ChatService.java`:

```java
// add fields
private final ChatRedisService redis;
private final RateLimiter rateLimiter;
private final StreamLock streamLock;

@Value("${chat.history.default-limit:20}") int defaultHistoryLimit;
@Value("${chat.history.max-limit:50}")     int maxHistoryLimit;
```

Update the constructor to:

```java
public ChatService(UserRepository users, ConversationRepository conversations,
                   MessageRepository messages, ChunkSourceRepository chunkSources,
                   SearchRpcClient rpc, OpenAiChatService openai, PromptBuilder promptBuilder,
                   ChatRedisService redis, RateLimiter rateLimiter, StreamLock streamLock) {
    this.users = users;
    this.conversations = conversations;
    this.messages = messages;
    this.chunkSources = chunkSources;
    this.rpc = rpc;
    this.openai = openai;
    this.promptBuilder = promptBuilder;
    this.redis = redis;
    this.rateLimiter = rateLimiter;
    this.streamLock = streamLock;
}
```

Add the new method (anywhere in the class, before `ask`):

```java
@Transactional(readOnly = true)
public HistoryForAiResponse historyForAi(Long conversationId, Integer limit) {
    int effective = limit == null ? defaultHistoryLimit : Math.min(Math.max(limit, 1), maxHistoryLimit);

    List<HistoryMessageDto> cached = redis.readHistoryTail(conversationId, effective);
    if (!cached.isEmpty()) {
        return new HistoryForAiResponse(conversationId, cached);
    }

    Conversation conv = conversations.findById(conversationId).orElse(null);
    if (conv == null) {
        return new HistoryForAiResponse(conversationId, List.of());
    }

    List<Message> all = messages.findByConversationIdOrderByCreatedAtAsc(conversationId);
    if (all.isEmpty()) {
        return new HistoryForAiResponse(conversationId, List.of());
    }

    List<HistoryMessageDto> warmed = all.stream()
        .map(m -> new HistoryMessageDto(m.getRole(), m.getContent(), m.getIntent(), m.getCreatedAt()))
        .toList();

    // Warm Redis with the entire conversation (capped server-side via LTRIM in appendHistory).
    warmed.forEach(dto -> redis.appendHistory(conversationId, dto));

    // Return only the requested tail length.
    int from = Math.max(0, warmed.size() - effective);
    return new HistoryForAiResponse(conversationId, warmed.subList(from, warmed.size()));
}
```

Required imports added at top of `ChatService.java`:
```java
import com.example.chat.dto.HistoryForAiResponse;
import com.example.chat.dto.HistoryMessageDto;
import com.example.chat.redis.ChatRedisService;
import com.example.chat.redis.RateLimiter;
import com.example.chat.redis.StreamLock;
import org.springframework.beans.factory.annotation.Value;
```

- [ ] **Step 5: Add the controller endpoint**

In `ChatController.java`, add:

```java
@GetMapping("/conversations/{id}/history-for-ai")
public ApiResponse<HistoryForAiResponse> historyForAi(
    @PathVariable Long id,
    @RequestParam(value = "limit", required = false) Integer limit) {
    return ApiResponse.ok(service.historyForAi(id, limit));
}
```

Add import: `import com.example.chat.dto.HistoryForAiResponse;`

- [ ] **Step 6: Run test to verify it passes**

```bash
cd backend && ./mvnw -q -Dtest=ChatServiceHistoryForAiTest test
```
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add backend/src/main/java/com/example/chat/chat/ChatService.java \
        backend/src/main/java/com/example/chat/chat/ChatController.java \
        backend/src/main/java/com/example/chat/redis/RateLimiter.java \
        backend/src/main/java/com/example/chat/redis/StreamLock.java \
        backend/src/test/java/com/example/chat/chat/ChatServiceHistoryForAiTest.java
git commit -m "feat(chat): /history-for-ai endpoint with Redis-then-DB read-through"
```

---

## Task 5: RateLimiter (fixed-window per-minute)

**Files:**
- Modify: `backend/src/main/java/com/example/chat/redis/RateLimiter.java`
- Create: `backend/src/test/java/com/example/chat/redis/RateLimiterTest.java`

- [ ] **Step 1: Write failing test `RateLimiterTest.java`**

```java
package com.example.chat.redis;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.time.Duration;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class RateLimiterTest {

    private StringRedisTemplate template;
    private ValueOperations<String, String> valueOps;
    private RateLimiter limiter;

    @BeforeEach
    void setUp() {
        template = mock(StringRedisTemplate.class);
        valueOps = mock(ValueOperations.class);
        when(template.opsForValue()).thenReturn(valueOps);
        limiter = new RateLimiter(template, 15, java.time.Clock.fixed(
            java.time.Instant.parse("2026-04-26T00:00:30Z"), java.time.ZoneOffset.UTC));
    }

    @Test
    void firstHitInBucketSetsExpiryAndPasses() {
        when(valueOps.increment(anyString())).thenReturn(1L);
        boolean ok = limiter.tryAcquire(42L);
        assertThat(ok).isTrue();
        verify(template).expire("chat:rate:42:30206320", Duration.ofSeconds(60));
    }

    @Test
    void belowLimitPasses() {
        when(valueOps.increment(anyString())).thenReturn(15L);
        assertThat(limiter.tryAcquire(42L)).isTrue();
    }

    @Test
    void atLimitPlusOneRejects() {
        when(valueOps.increment(anyString())).thenReturn(16L);
        assertThat(limiter.tryAcquire(42L)).isFalse();
    }

    @Test
    void redisFailureFailsOpen() {
        when(valueOps.increment(anyString()))
            .thenThrow(new org.springframework.data.redis.RedisConnectionFailureException("down"));
        assertThat(limiter.tryAcquire(42L)).isTrue();  // fail-open per spec §5
    }
}
```

(The bucket value `30206320` = `Instant.parse("2026-04-26T00:00:30Z").getEpochSecond() / 60` = `1809475230 / 60` — recompute and substitute the right number when running. Or compute it inline in the test if you prefer; using a fixed `Clock` keeps it deterministic.)

Note: `1809475230 / 60 = 30157920` — verify the actual integer at test-write time and use that constant. (The test's job is to assert TTL is set with the right key shape; if you find this arithmetic noisy, replace the equality check with `verify(template).expire(startsWith("chat:rate:42:"), eq(Duration.ofSeconds(60)));`.)

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && ./mvnw -q -Dtest=RateLimiterTest test
```
Expected: FAIL — `RateLimiter` constructor mismatch.

- [ ] **Step 3: Implement `RateLimiter.java`**

Replace the stub:

```java
package com.example.chat.redis;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DataAccessException;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Clock;
import java.time.Duration;

@Service
public class RateLimiter {

    private static final Logger log = LoggerFactory.getLogger(RateLimiter.class);

    private final StringRedisTemplate template;
    private final int perUserPerMinute;
    private final Clock clock;

    public RateLimiter(StringRedisTemplate template,
                       @Value("${chat.rate-limit.per-user-per-minute:15}") int perUserPerMinute) {
        this(template, perUserPerMinute, Clock.systemUTC());
    }

    // visible for tests
    RateLimiter(StringRedisTemplate template, int perUserPerMinute, Clock clock) {
        this.template = template;
        this.perUserPerMinute = perUserPerMinute;
        this.clock = clock;
    }

    public boolean tryAcquire(Long userId) {
        long bucket = clock.instant().getEpochSecond() / 60;
        String key = "chat:rate:" + userId + ":" + bucket;
        try {
            Long count = template.opsForValue().increment(key);
            if (count != null && count == 1L) {
                template.expire(key, Duration.ofSeconds(60));
            }
            return count != null && count <= perUserPerMinute;
        } catch (DataAccessException e) {
            log.warn("RateLimiter Redis failure for user={}, failing open: {}", userId, e.getMessage());
            return true;
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && ./mvnw -q -Dtest=RateLimiterTest test
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/src/main/java/com/example/chat/redis/RateLimiter.java \
        backend/src/test/java/com/example/chat/redis/RateLimiterTest.java
git commit -m "feat(redis): RateLimiter — fixed-window per-user-per-minute"
```

---

## Task 6: StreamLock (SETNX active-stream lock)

**Files:**
- Modify: `backend/src/main/java/com/example/chat/redis/StreamLock.java`
- Create: `backend/src/test/java/com/example/chat/redis/StreamLockTest.java`

- [ ] **Step 1: Write failing test `StreamLockTest.java`**

```java
package com.example.chat.redis;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.time.Duration;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

class StreamLockTest {

    private StringRedisTemplate template;
    private ValueOperations<String, String> valueOps;
    private StreamLock lock;

    @BeforeEach
    void setUp() {
        template = mock(StringRedisTemplate.class);
        valueOps = mock(ValueOperations.class);
        when(template.opsForValue()).thenReturn(valueOps);
        lock = new StreamLock(template, 60);
    }

    @Test
    void acquireSucceedsWhenSetIfAbsentReturnsTrue() {
        when(valueOps.setIfAbsent(eq("chat:active:42"), eq("1"), eq(Duration.ofSeconds(60))))
            .thenReturn(Boolean.TRUE);
        assertThat(lock.acquire(42L)).isTrue();
    }

    @Test
    void acquireFailsWhenSetIfAbsentReturnsFalse() {
        when(valueOps.setIfAbsent(any(), any(), any())).thenReturn(Boolean.FALSE);
        assertThat(lock.acquire(42L)).isFalse();
    }

    @Test
    void redisFailureFailsOpen() {
        when(valueOps.setIfAbsent(any(), any(), any()))
            .thenThrow(new org.springframework.data.redis.RedisConnectionFailureException("down"));
        assertThat(lock.acquire(42L)).isTrue();
    }

    @Test
    void releaseDeletesKey() {
        lock.release(42L);
        verify(template).delete("chat:active:42");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && ./mvnw -q -Dtest=StreamLockTest test
```
Expected: FAIL — `StreamLock` constructor mismatch.

- [ ] **Step 3: Implement `StreamLock.java`**

Replace the stub:

```java
package com.example.chat.redis;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DataAccessException;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;

@Service
public class StreamLock {

    private static final Logger log = LoggerFactory.getLogger(StreamLock.class);

    private final StringRedisTemplate template;
    private final long ttlSeconds;

    public StreamLock(StringRedisTemplate template,
                      @Value("${chat.rate-limit.active-stream-lock-seconds:60}") long ttlSeconds) {
        this.template = template;
        this.ttlSeconds = ttlSeconds;
    }

    public boolean acquire(Long userId) {
        try {
            Boolean ok = template.opsForValue()
                .setIfAbsent(key(userId), "1", Duration.ofSeconds(ttlSeconds));
            return Boolean.TRUE.equals(ok);
        } catch (DataAccessException e) {
            log.warn("StreamLock Redis failure for user={}, failing open: {}", userId, e.getMessage());
            return true;
        }
    }

    public void release(Long userId) {
        try {
            template.delete(key(userId));
        } catch (DataAccessException e) {
            log.warn("StreamLock release Redis failure for user={}: {}", userId, e.getMessage());
        }
    }

    private static String key(Long userId) { return "chat:active:" + userId; }
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && ./mvnw -q -Dtest=StreamLockTest test
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/src/main/java/com/example/chat/redis/StreamLock.java \
        backend/src/test/java/com/example/chat/redis/StreamLockTest.java
git commit -m "feat(redis): StreamLock — SETNX-based concurrent-stream guard"
```

---

## Task 7: Integrate rate-limit, lock, history push, stream buffer into ChatService.ask

**Files:**
- Modify: `backend/src/main/java/com/example/chat/chat/ChatService.java`

- [ ] **Step 1: Update the `ask` method body**

Replace the existing `ask(Long conversationId, String userMessage)` method in `ChatService.java`:

```java
public SseEmitter ask(Long conversationId, String userMessage) {
    SseEmitter emitter = new SseEmitter(60_000L);

    Conversation conv = conversations.findById(conversationId)
        .orElseThrow(() -> new IllegalArgumentException("conversation not found"));
    Long userId = conv.getUser().getId();

    // 1) Rate limit
    if (!rateLimiter.tryAcquire(userId)) {
        sendEvent(emitter, "error", Map.of("message", "rate limit exceeded"));
        emitter.complete();
        return emitter;
    }

    // 2) Active-stream lock
    if (!streamLock.acquire(userId)) {
        sendEvent(emitter, "error", Map.of("message", "a response is already streaming"));
        emitter.complete();
        return emitter;
    }

    // 3) Persist user message + push to Redis history
    Message savedUserMsg = messages.save(Message.builder()
        .conversation(conv).role("user").content(userMessage).build());
    conversations.save(conv);
    redis.appendHistory(conversationId, new HistoryMessageDto(
        "user", userMessage, null, savedUserMsg.getCreatedAt()));

    // 4) RPC → Python
    SearchReplyMessage reply;
    try {
        reply = rpc.requestSearch(userMessage, String.valueOf(conversationId));
    } catch (Exception ex) {
        sendEvent(emitter, "error", Map.of("message", nullSafe(ex.getMessage())));
        streamLock.release(userId);
        emitter.complete();
        return emitter;
    }

    // 5) Persist empty assistant message + chunk sources
    Message assistantMsg = messages.save(Message.builder()
        .conversation(conv).role("assistant").content("")
        .intent(reply.intent())
        .detectedLanguage(reply.detectedLanguage())
        .webSearchUsed(reply.webSearchUsed())
        .build());
    if (reply.sources() != null) {
        for (SourceDto s : reply.sources()) {
            chunkSources.save(ChunkSource.builder()
                .message(assistantMsg)
                .name(s.name()).url(s.url())
                .pageNumber(s.pageNumber()).totalPages(s.totalPages())
                .build());
        }
    }

    // 6) Meta event
    sendEvent(emitter, "meta", Map.of(
        "intent", nullSafe(reply.intent()),
        "detectedLanguage", nullSafe(reply.detectedLanguage()),
        "sources", reply.sources() == null ? List.of() : reply.sources(),
        "webSearchUsed", reply.webSearchUsed(),
        "assistantMessageId", assistantMsg.getId()
    ));

    // 7) Static branch
    if ("static".equals(reply.mode())) {
        String text = reply.response() == null ? "" : reply.response();
        sendEvent(emitter, "delta", Map.of("content", text));
        assistantMsg.setContent(text);
        messages.save(assistantMsg);
        redis.appendHistory(conversationId, new HistoryMessageDto(
            "assistant", text, reply.intent(), assistantMsg.getCreatedAt()));
        sendEvent(emitter, "done", Map.of());
        streamLock.release(userId);
        emitter.complete();
        return emitter;
    }

    // 8) Stream branch
    StringBuilder buffer = new StringBuilder();
    String prompt = promptBuilder.build(reply.chunks(), reply.standaloneQuery(), reply.detectedLanguage());

    Disposable sub = openai.streamChat(prompt).subscribe(
        token -> {
            buffer.append(token);
            redis.appendStreamToken(conversationId, token);
            sendEvent(emitter, "delta", Map.of("content", token));
        },
        error -> {
            log.warn("OpenAI stream error", error);
            sendEvent(emitter, "error", Map.of("message", error.getMessage()));
            persistAssistantAndCache(assistantMsg, buffer.toString(), buffer.length() > 0,
                                     conversationId, reply.intent());
            redis.deleteStream(conversationId);
            streamLock.release(userId);
            emitter.complete();
        },
        () -> {
            persistAssistantAndCache(assistantMsg, buffer.toString(), false,
                                     conversationId, reply.intent());
            redis.deleteStream(conversationId);
            sendEvent(emitter, "done", Map.of());
            streamLock.release(userId);
            emitter.complete();
        }
    );

    emitter.onTimeout(() -> {
        sub.dispose();
        persistAssistantAndCache(assistantMsg, buffer.toString(), buffer.length() > 0,
                                 conversationId, reply.intent());
        redis.deleteStream(conversationId);
        streamLock.release(userId);
    });
    emitter.onError(e -> {
        sub.dispose();
        persistAssistantAndCache(assistantMsg, buffer.toString(), buffer.length() > 0,
                                 conversationId, reply.intent());
        redis.deleteStream(conversationId);
        streamLock.release(userId);
    });
    emitter.onCompletion(() -> {
        if (!sub.isDisposed()) sub.dispose();
    });

    return emitter;
}
```

- [ ] **Step 2: Replace `persistAssistant` with `persistAssistantAndCache`**

In `ChatService.java`, replace the existing `persistAssistant` helper with:

```java
void persistAssistantAndCache(Message m, String content, boolean stopped,
                              Long conversationId, String intent) {
    m.setContent(content);
    m.setStopped(stopped);
    Message saved = messages.save(m);
    redis.appendHistory(conversationId, new HistoryMessageDto(
        "assistant", content, intent, saved.getCreatedAt()));
}
```

Remove the old `persistAssistant` method.

- [ ] **Step 3: Build to confirm no compile errors**

```bash
cd backend && ./mvnw -q -DskipTests compile
```
Expected: BUILD SUCCESS.

- [ ] **Step 4: Run all backend tests**

```bash
cd backend && ./mvnw -q test
```
Expected: all existing tests + new tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/main/java/com/example/chat/chat/ChatService.java
git commit -m "feat(chat): integrate Redis rate-limit/lock/history/stream-buffer into ask"
```

---

## Task 8: Cache invalidation on `deleteFromMessage`

**Files:**
- Modify: `backend/src/main/java/com/example/chat/chat/ChatService.java`

- [ ] **Step 1: Update `deleteFromMessage` to invalidate Redis history**

Replace the existing method:

```java
@Transactional
public void deleteFromMessage(Long conversationId, Long messageId) {
    List<Message> toDelete = messages.findByConversationIdAndIdGreaterThanEqual(conversationId, messageId);
    if (toDelete.isEmpty()) return;
    List<Long> ids = toDelete.stream().map(Message::getId).toList();
    chunkSources.deleteByMessageIdIn(ids);
    messages.deleteAll(toDelete);
    redis.invalidateHistory(conversationId);
}
```

- [ ] **Step 2: Compile and run all tests**

```bash
cd backend && ./mvnw -q test
```
Expected: BUILD SUCCESS, all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add backend/src/main/java/com/example/chat/chat/ChatService.java
git commit -m "feat(chat): invalidate Redis history on deleteFromMessage"
```

---

## Task 9: Update Python AI client (URL, path, response shape, parser)

**Files:**
- Modify: `src/external/fetch_history.py`
- Modify: `src/rag/search/entrypoint.py`
- Modify: `.env`
- Create: `tests/test_fetch_history.py`

- [ ] **Step 1: Write failing test `tests/test_fetch_history.py`**

```python
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.external.fetch_history import fetch_raw_chat_history


def _mock_response(json_payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_payload
    resp.raise_for_status.return_value = None
    return resp


def test_returns_messages_from_envelope():
    payload = {
        "success": True,
        "data": {
            "conversationId": 123,
            "messages": [
                {"role": "user",      "content": "hi",     "intent": None,             "createdAt": "2026-04-26T00:00:00Z"},
                {"role": "assistant", "content": "hello",  "intent": "core_knowledge", "createdAt": "2026-04-26T00:00:01Z"},
            ],
        },
    }
    with patch("src.external.fetch_history.httpx.AsyncClient") as client_cls:
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.get.return_value = _mock_response(payload)
        client_cls.return_value = client

        import asyncio
        result = asyncio.run(fetch_raw_chat_history("123"))

    assert result == payload["data"]["messages"]


def test_returns_none_on_unsuccessful_envelope():
    payload = {"success": False, "error": "nope"}
    with patch("src.external.fetch_history.httpx.AsyncClient") as client_cls:
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.get.return_value = _mock_response(payload)
        client_cls.return_value = client

        import asyncio
        result = asyncio.run(fetch_raw_chat_history("123"))

    assert result is None


def test_filter_uses_role_and_flat_content():
    from src.rag.search.entrypoint import _filter_core_knowledge_pairs

    msgs = [
        {"role": "user",      "content": "Q1", "intent": None},
        {"role": "assistant", "content": "A1", "intent": "core_knowledge"},
        {"role": "user",      "content": "Q2", "intent": None},
        {"role": "assistant", "content": "A2", "intent": "off_topic"},
    ]
    out = _filter_core_knowledge_pairs(msgs)
    assert out == [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_fetch_history.py -v
```
Expected: FAIL — assertions on response shape don't match current implementation.

- [ ] **Step 3: Update `src/external/fetch_history.py`**

```python
import httpx

from src.config.app_config import get_app_config
from src.constants.app_constant import CHAT_HISTORY_TIMEOUT
from src.utils.logger_utils import alog_function_call


@alog_function_call
async def fetch_raw_chat_history(
        conversation_id: str,
) -> list[dict] | None:
    config = get_app_config()
    url = f"{config.CHAT_HISTORY_API_BASE}/api/conversations/{conversation_id}/history-for-ai"
    params = {"limit": 20}

    try:
        async with httpx.AsyncClient(timeout=CHAT_HISTORY_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            body = resp.json()
    except httpx.HTTPError:
        return None

    if not body.get("success"):
        return None

    return body.get("data", {}).get("messages", [])
```

(Note: dropped the `X-Api-Key` header per spec §4.3 — Java mới không yêu cầu.)

- [ ] **Step 4: Update `_filter_core_knowledge_pairs` in `src/rag/search/entrypoint.py`**

Replace the function body:

```python
def _filter_core_knowledge_pairs(
        messages: list[dict],
) -> list[dict]:
    history: list[dict] = []
    i = 0

    while i < len(messages) - 1:
        user_msg = messages[i]
        ai_msg = messages[i + 1]

        if (
                user_msg.get("role") == "user"
                and ai_msg.get("role") == "assistant"
                and ai_msg.get("intent") == "core_knowledge"
        ):
            history.append({
                "role": "user",
                "content": user_msg.get("content", ""),
            })
            history.append({
                "role": "assistant",
                "content": ai_msg.get("content", ""),
            })
            i += 2
        else:
            i += 1

    return history
```

- [ ] **Step 5: Update `.env`**

Change `CHAT_HISTORY_API_BASE=http://10.98.36.75:8080` to `CHAT_HISTORY_API_BASE=http://localhost:8085`. Add a comment noting the change is intentional.

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_fetch_history.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 7: Run the full Python test suite to catch regressions**

```bash
pytest tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/external/fetch_history.py src/rag/search/entrypoint.py tests/test_fetch_history.py .env
git commit -m "feat(ai): consume Java /history-for-ai with role/content schema"
```

---

## Task 10: Manual smoke test (end-to-end)

**Files:** none — verifies the live system.

- [ ] **Step 1: Bring up infra**

Ensure Redis (`localhost:6379`, password `admin123`) and RabbitMQ (`localhost:5672`, `admin/admin123`) are running. Use whatever docker-compose setup you already have (the env vars in `.env` indicate they exist).

- [ ] **Step 2: Start Python backend**

```bash
python main.py
```
Expected: FastAPI binds (default port).

- [ ] **Step 3: Start Java backend**

```bash
cd backend && REDIS_PASSWORD=admin123 ./mvnw spring-boot:run
```
Expected: starts on port 8085, no Redis connection errors in logs.

- [ ] **Step 4: Start frontend**

```bash
cd frontend && npm run dev
```

- [ ] **Step 5: Functional check (in browser)**

Open the dev URL. Create user, create a new conversation, send 3 different messages. Verify all three messages stream and render correctly. Reload the page → all 6 messages (3 pairs) reload from `/messages`.

- [ ] **Step 6: Inspect Redis state**

```bash
redis-cli -a admin123 KEYS 'chat:*'
redis-cli -a admin123 LRANGE chat:history:<conversationId> 0 -1
redis-cli -a admin123 TTL chat:history:<conversationId>
```
Expected: 6 entries in the history list (3 user + 3 assistant), TTL > 0 and ≤ 1800. No leftover `chat:stream:*` keys after streams completed.

- [ ] **Step 7: Rate-limit check**

Send 16 messages within 60 seconds (script with `curl` against `/api/conversations/{id}/ask/stream`). The 16th request should emit an SSE `error` event with `"rate limit exceeded"`.

- [ ] **Step 8: Active-stream lock check**

While one stream is in flight, send another `/ask/stream` for the same user. Expected: SSE `error` event with `"a response is already streaming"`.

- [ ] **Step 9: Idle eviction (optional)**

Set `chat.history.redis-ttl-seconds: 30` temporarily, send a message, wait 60 s. `redis-cli KEYS 'chat:history:*'` returns nothing. Send a new message — Java warms from DB; the Redis list rebuilds from the full conversation. Revert the config when done.

- [ ] **Step 10: Crash recovery**

While streaming, kill the Java process. Verify `chat:stream:*` and `chat:active:*` keys exist briefly, then expire (5 min and 60 s respectively). Restart Java; new requests succeed without manual cleanup.

- [ ] **Step 11: No commit** — manual verification only.

---

## Self-review

**Spec coverage check** (against `docs/superpowers/specs/2026-04-26-redis-chat-history-design.md`):

| Spec section | Implemented in task |
|---|---|
| §3.1 Components | Tasks 1, 3, 5, 6 (Java owns Redis namespace; Python via HTTP) |
| §3.2 Redis key schema | Tasks 3 (`chat:history`, `chat:stream`), 5 (`chat:rate`), 6 (`chat:active`) |
| §3.3 Standardized API schema (envelope, both endpoints) | Tasks 2, 4 |
| §3.4 Data flow per chat turn (rate-limit, lock, history push, stream buffer, flush) | Task 7 |
| §3.5 Cache invalidation on deleteFromMessage | Task 8 |
| §4.1 Java new files | Tasks 1, 2, 3, 5, 6 |
| §4.2 Java modified files | Tasks 1, 2, 4, 7, 8 |
| §4.3 Python modified files | Task 9 |
| §4.4 Frontend modified files | Task 2 (single FE-touching task) |
| §4.5 Env / config | Tasks 1, 9 |
| §5 Error handling matrix | Tasks 5, 6 (fail-open), 7 (lock release on all exits), 9 (Python keeps existing None-on-error) |
| §6 Testing | Tasks 3 (ChatRedisServiceTest), 4 (ChatServiceHistoryForAiTest), 5 (RateLimiterTest), 6 (StreamLockTest), 9 (test_fetch_history.py), 10 (manual smoke) |
| §7 Rollout order | Tasks 1–10 follow the spec's 8-step order (split slightly for TDD granularity) |

**Type consistency:**
- `HistoryMessageDto(role, content, intent, createdAt)` used identically in Tasks 3, 4, 7.
- `historyForAi(Long, Integer)` signature consistent: ChatService method (Task 4), called by controller (Task 4).
- `redis.appendHistory(Long, HistoryMessageDto)` consistent across Tasks 3, 4, 7.
- `streamLock.acquire(Long)` / `release(Long)` consistent.
- Redis key prefixes consistent: `chat:history:`, `chat:stream:`, `chat:active:`, `chat:rate:`.

**No placeholders:** every step has either exact code, exact command, or both.

**Open items deferred** (per spec §8): auth on `/history-for-ai`, multi-instance rate limiting, stream resume, Postgres migration. Not in this plan.
