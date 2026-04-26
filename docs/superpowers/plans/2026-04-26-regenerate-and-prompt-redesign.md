# Regenerate flow + Recomp summarize + softened clarity prompt — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Regenerate produce one new assistant answer per click — without duplicating the user message and without re-running retrieval when chunks are cached — and bring the SUMMARIZE / CLARITY prompts in `prompt.py` up to spec.

**Architecture:** New POST `/api/conversations/{c}/messages/{a}/regenerate/stream` endpoint. Java caches chunks in Redis at `chat:chunks:{assistantMsgId}` with a `ChunkContext{chunks, standaloneQuery}` payload (TTL 1h) when the original answer is created; on regenerate it reuses that cache, deletes the old assistant message, persists a new one, and streams via OpenAI with `temperature=0.7` (bumped to bias variation). Frontend opens SSE on the new endpoint without sending the user message again. Python prompt module gets two rewrites: SUMMARIZE becomes Recomp-style (query-focused single-string), CLARITY becomes a friendlier prose prompt that biases toward CLEAR.

**Tech Stack:** Spring Boot 3 (Java 21, JPA, Redis, RabbitMQ, SseEmitter, Reactor) · React 18 + Vite + TypeScript · Python 3.12 (LangChain-style Azure client) · JUnit 5 + AssertJ · Vitest (frontend) · pytest (Python).

---

## File map

**Backend (Java):**
- Modify: `backend/src/main/java/com/example/chat/redis/ChatRedisService.java`
- Modify: `backend/src/main/java/com/example/chat/chat/ChatService.java`
- Modify: `backend/src/main/java/com/example/chat/chat/ChatController.java`
- Modify: `backend/src/main/java/com/example/chat/openai/OpenAiChatService.java`
- Modify: `backend/src/main/java/com/example/chat/repository/MessageRepository.java`
- Modify: `backend/src/main/resources/application.yml`
- Create: `backend/src/main/java/com/example/chat/dto/ChunkContext.java`
- Create: `backend/src/test/java/com/example/chat/redis/ChatRedisServiceChunksTest.java`
- Create: `backend/src/test/java/com/example/chat/chat/ChatServiceRegenerateTest.java`

**Frontend:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/api/chatApi.ts`
- Modify: `frontend/src/hooks/useChatStream.ts`
- Modify: `frontend/src/components/ChatWindow.tsx`
- Modify: `frontend/src/components/MessageBubble.tsx`

**Python:**
- Modify: `src/rag/search/prompt.py`
- Modify: `src/rag/search/entrypoint.py`
- Modify: `src/rag/search/pipeline.py`
- Modify: `src/rag/search/agent/nodes.py`

---

## Phase 1 — Backend: Redis chunk cache foundation

### Task 1: Create `ChunkContext` DTO

**Files:**
- Create: `backend/src/main/java/com/example/chat/dto/ChunkContext.java`

- [ ] **Step 1: Write the file**

```java
package com.example.chat.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ChunkContext(List<ChunkDto> chunks, String standaloneQuery) {}
```

- [ ] **Step 2: Compile**

Run: `cd backend && ./mvnw compile -q`
Expected: BUILD SUCCESS, no errors.

- [ ] **Step 3: Commit**

```bash
git add backend/src/main/java/com/example/chat/dto/ChunkContext.java
git commit -m "feat(chat): add ChunkContext DTO for chunk cache payload"
```

---

### Task 2: Add chunk-cache methods to `ChatRedisService` (TDD)

**Files:**
- Test: `backend/src/test/java/com/example/chat/redis/ChatRedisServiceChunksTest.java`
- Modify: `backend/src/main/java/com/example/chat/redis/ChatRedisService.java`

- [ ] **Step 1: Write the failing test**

Create `backend/src/test/java/com/example/chat/redis/ChatRedisServiceChunksTest.java`:

```java
package com.example.chat.redis;

import com.example.chat.dto.ChunkContext;
import com.example.chat.dto.ChunkDto;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.time.Duration;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ChatRedisServiceChunksTest {

    StringRedisTemplate template;
    ValueOperations<String, String> ops;
    ChatRedisService service;
    ObjectMapper mapper = new ObjectMapper();

    @BeforeEach
    void setup() {
        template = mock(StringRedisTemplate.class);
        ops = mock(ValueOperations.class);
        when(template.opsForValue()).thenReturn(ops);
        service = new ChatRedisService(template, mapper, 1800L, 50, 300L, 3600L);
    }

    @Test
    void cacheChunks_writesJsonWithTtl() throws Exception {
        ChunkDto c = new ChunkDto("hello", Map.of("file_name", "doc.pdf"));
        ChunkContext ctx = new ChunkContext(List.of(c), "what is hello");

        service.cacheChunks(42L, ctx);

        verify(ops).set(eq("chat:chunks:42"), any(String.class), eq(Duration.ofSeconds(3600L)));
    }

    @Test
    void readChunks_returnsNullWhenMiss() {
        when(ops.get("chat:chunks:99")).thenReturn(null);
        assertThat(service.readChunks(99L)).isNull();
    }

    @Test
    void readChunks_roundTripsContext() throws Exception {
        ChunkDto c = new ChunkDto("body", Map.of("page_number", 7));
        ChunkContext ctx = new ChunkContext(List.of(c), "q");
        when(ops.get("chat:chunks:7")).thenReturn(mapper.writeValueAsString(ctx));

        ChunkContext out = service.readChunks(7L);

        assertThat(out).isNotNull();
        assertThat(out.standaloneQuery()).isEqualTo("q");
        assertThat(out.chunks()).hasSize(1);
        assertThat(out.chunks().get(0).text()).isEqualTo("body");
    }

    @Test
    void deleteChunks_callsRedisDelete() {
        service.deleteChunks(5L);
        verify(template).delete("chat:chunks:5");
    }
}
```

- [ ] **Step 2: Run test to verify it fails (compile error)**

Run: `cd backend && ./mvnw test -Dtest=ChatRedisServiceChunksTest -q`
Expected: FAIL with `cannot find symbol method cacheChunks/readChunks/deleteChunks` and constructor arity mismatch (5 args vs new 6).

- [ ] **Step 3: Add a fourth TTL parameter and the three methods**

Edit `backend/src/main/java/com/example/chat/redis/ChatRedisService.java`:

Add new field + constructor param + key constant + three methods. Keep existing methods unchanged.

```java
// Add near other constants/fields:
private final long chunksTtlSeconds;
private static String chunksKey(Long convOrMsgId) { return "chat:chunks:" + convOrMsgId; }

// Replace constructor signature:
public ChatRedisService(
    StringRedisTemplate template,
    ObjectMapper mapper,
    @Value("${chat.history.redis-ttl-seconds:1800}") long historyTtlSeconds,
    @Value("${chat.history.redis-max-entries:50}") int historyMaxEntries,
    @Value("${chat.stream.redis-ttl-seconds:300}") long streamTtlSeconds,
    @Value("${chat.chunks.ttl-seconds:3600}") long chunksTtlSeconds
) {
    this.template = template;
    this.mapper = mapper;
    this.historyTtlSeconds = historyTtlSeconds;
    this.historyMaxEntries = historyMaxEntries;
    this.streamTtlSeconds = streamTtlSeconds;
    this.chunksTtlSeconds = chunksTtlSeconds;
}

// New methods (add near appendStreamToken / deleteStream):
public void cacheChunks(Long assistantMessageId, com.example.chat.dto.ChunkContext ctx) {
    String key = chunksKey(assistantMessageId);
    try {
        String json = mapper.writeValueAsString(ctx);
        template.opsForValue().set(key, json, Duration.ofSeconds(chunksTtlSeconds));
    } catch (JsonProcessingException e) {
        log.warn("Failed to serialize ChunkContext for msg={}: {}", assistantMessageId, e.getMessage());
    } catch (DataAccessException e) {
        log.warn("Redis cacheChunks failed for msg={}: {}", assistantMessageId, e.getMessage());
    }
}

public com.example.chat.dto.ChunkContext readChunks(Long assistantMessageId) {
    String key = chunksKey(assistantMessageId);
    try {
        String json = template.opsForValue().get(key);
        if (json == null) return null;
        return mapper.readValue(json, com.example.chat.dto.ChunkContext.class);
    } catch (JsonProcessingException e) {
        log.warn("Failed to parse ChunkContext for msg={}: {}", assistantMessageId, e.getMessage());
        return null;
    } catch (DataAccessException e) {
        log.warn("Redis readChunks failed for msg={}: {}", assistantMessageId, e.getMessage());
        return null;
    }
}

public void deleteChunks(Long assistantMessageId) {
    try {
        template.delete(chunksKey(assistantMessageId));
    } catch (DataAccessException e) {
        log.warn("Redis deleteChunks failed for msg={}: {}", assistantMessageId, e.getMessage());
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ./mvnw test -Dtest=ChatRedisServiceChunksTest -q`
Expected: PASS, 4 tests.

- [ ] **Step 5: Run the full backend test suite to ensure no regression**

Run: `cd backend && ./mvnw test -q`
Expected: BUILD SUCCESS, all existing tests still pass (the new constructor parameter has a default value, so existing tests using the 5-arg form will fail; **also** update `ChatRedisServiceTest` and any other test that builds `new ChatRedisService(...)` to pass `3600L` as the sixth arg).

If existing tests fail because of the constructor change, fix them by appending `, 3600L` to the constructor call. Re-run `./mvnw test -q` until green.

- [ ] **Step 6: Commit**

```bash
git add backend/src/main/java/com/example/chat/redis/ChatRedisService.java \
        backend/src/test/java/com/example/chat/redis/ChatRedisServiceChunksTest.java \
        backend/src/test/java/com/example/chat/redis/ChatRedisServiceTest.java
git commit -m "feat(redis): cacheChunks/readChunks/deleteChunks for regenerate path"
```

---

### Task 3: Cache chunks during `ChatService.ask()`

**Files:**
- Modify: `backend/src/main/java/com/example/chat/chat/ChatService.java:193-207`

- [ ] **Step 1: Add cache write after the chunk-source persistence loop**

In `ChatService.java`, locate the assistant-message persistence block (around line 193-207 — after `chunkSources.save(...)` loop). Add immediately after:

```java
// Cache full chunk payload + standalone query for potential regenerate.
if (reply.chunks() != null && !reply.chunks().isEmpty()) {
    redis.cacheChunks(
        assistantMsg.getId(),
        new com.example.chat.dto.ChunkContext(reply.chunks(), reply.standaloneQuery())
    );
}
```

- [ ] **Step 2: Compile**

Run: `cd backend && ./mvnw compile -q`
Expected: BUILD SUCCESS.

- [ ] **Step 3: Run full test suite**

Run: `cd backend && ./mvnw test -q`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/src/main/java/com/example/chat/chat/ChatService.java
git commit -m "feat(chat): cache chunks in Redis after assistant msg persisted"
```

---

## Phase 2 — Backend: regenerate endpoint

### Task 4: Add `findLatestUserBefore` to `MessageRepository`

**Files:**
- Modify: `backend/src/main/java/com/example/chat/repository/MessageRepository.java`

- [ ] **Step 1: Add the query method**

```java
package com.example.chat.repository;

import com.example.chat.domain.Message;
import org.springframework.data.domain.Limit;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface MessageRepository extends JpaRepository<Message, Long> {
    List<Message> findByConversationIdOrderByCreatedAtAsc(Long conversationId);
    List<Message> findByConversationIdAndIdGreaterThanEqual(Long conversationId, Long messageId);

    @Query("SELECT m FROM Message m WHERE m.conversation.id = :convId AND m.id < :before AND m.role = 'user' ORDER BY m.id DESC")
    List<Message> findUserMessagesBefore(@Param("convId") Long convId, @Param("before") Long before, Limit limit);

    default Optional<Message> findLatestUserBefore(Long convId, Long before) {
        List<Message> rows = findUserMessagesBefore(convId, before, Limit.of(1));
        return rows.isEmpty() ? Optional.empty() : Optional.of(rows.get(0));
    }
}
```

- [ ] **Step 2: Compile**

Run: `cd backend && ./mvnw compile -q`
Expected: BUILD SUCCESS.

- [ ] **Step 3: Commit**

```bash
git add backend/src/main/java/com/example/chat/repository/MessageRepository.java
git commit -m "feat(chat): MessageRepository.findLatestUserBefore for regenerate"
```

---

### Task 5: Add temperature overload to `OpenAiChatService.streamChat`

**Files:**
- Modify: `backend/src/main/java/com/example/chat/openai/OpenAiChatService.java`

- [ ] **Step 1: Refactor streamChat to take an explicit temperature, then add overload**

Locate the current `streamChat(String prompt)` and the body construction. Refactor as follows:

```java
public reactor.core.publisher.Flux<String> streamChat(String prompt) {
    return streamChat(prompt, this.temperature);
}

public reactor.core.publisher.Flux<String> streamChat(String prompt, double temperatureOverride) {
    java.util.Map<String, Object> body = java.util.Map.of(
        "messages", java.util.List.of(java.util.Map.of("role", "user", "content", prompt)),
        "stream", true,
        "temperature", temperatureOverride,
        "top_p", topP,
        "max_tokens", maxTokens
    );
    // ... rest of the existing WebClient POST + SSE parsing logic, unchanged ...
}
```

The default-flow callers (`ChatService.ask()`) keep using the 1-arg form; regenerate will call the 2-arg form.

- [ ] **Step 2: Compile**

Run: `cd backend && ./mvnw compile -q`
Expected: BUILD SUCCESS.

- [ ] **Step 3: Run existing OpenAiChatServiceTest**

Run: `cd backend && ./mvnw test -Dtest=OpenAiChatServiceTest -q`
Expected: PASS (no behavior change to default flow).

- [ ] **Step 4: Commit**

```bash
git add backend/src/main/java/com/example/chat/openai/OpenAiChatService.java
git commit -m "feat(openai): streamChat(prompt, temperature) overload for regenerate"
```

---

### Task 6: Implement `ChatService.regenerate` (TDD with mocks)

**Files:**
- Test: `backend/src/test/java/com/example/chat/chat/ChatServiceRegenerateTest.java`
- Modify: `backend/src/main/java/com/example/chat/chat/ChatService.java`

- [ ] **Step 1: Write the failing test (cache hit scenario)**

Create `backend/src/test/java/com/example/chat/chat/ChatServiceRegenerateTest.java`:

```java
package com.example.chat.chat;

import com.example.chat.domain.Conversation;
import com.example.chat.domain.Message;
import com.example.chat.domain.User;
import com.example.chat.dto.ChunkContext;
import com.example.chat.dto.ChunkDto;
import com.example.chat.messaging.SearchRpcClient;
import com.example.chat.openai.OpenAiChatService;
import com.example.chat.openai.PromptBuilder;
import com.example.chat.redis.ChatRedisService;
import com.example.chat.redis.RateLimiter;
import com.example.chat.redis.StreamLock;
import com.example.chat.repository.ChunkSourceRepository;
import com.example.chat.repository.ConversationRepository;
import com.example.chat.repository.MessageRepository;
import com.example.chat.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ChatServiceRegenerateTest {

    UserRepository users;
    ConversationRepository conversations;
    MessageRepository messages;
    ChunkSourceRepository chunkSources;
    SearchRpcClient rpc;
    OpenAiChatService openai;
    PromptBuilder promptBuilder;
    ChatRedisService redis;
    RateLimiter rateLimiter;
    StreamLock streamLock;
    ChatService service;

    @BeforeEach
    void setup() {
        users = mock(UserRepository.class);
        conversations = mock(ConversationRepository.class);
        messages = mock(MessageRepository.class);
        chunkSources = mock(ChunkSourceRepository.class);
        rpc = mock(SearchRpcClient.class);
        openai = mock(OpenAiChatService.class);
        promptBuilder = mock(PromptBuilder.class);
        redis = mock(ChatRedisService.class);
        rateLimiter = mock(RateLimiter.class);
        streamLock = mock(StreamLock.class);
        service = new ChatService(users, conversations, messages, chunkSources,
            rpc, openai, promptBuilder, redis, rateLimiter, streamLock);
    }

    private Message buildAssistant(long id, Conversation conv) {
        Message m = Message.builder()
            .role("assistant").content("old answer")
            .intent("core_knowledge").detectedLanguage("English").webSearchUsed(false)
            .conversation(conv).build();
        m.setId(id);
        return m;
    }

    private Conversation buildConv(long convId, long userId) {
        User u = User.builder().username("alice").build();
        u.setId(userId);
        Conversation c = Conversation.builder().user(u).title("t").build();
        c.setId(convId);
        return c;
    }

    @Test
    void regenerate_cacheHit_doesNotCallRpc() {
        Conversation conv = buildConv(1L, 100L);
        Message assistant = buildAssistant(20L, conv);
        Message userMsg = Message.builder().role("user").content("question").conversation(conv).build();
        userMsg.setId(19L);

        when(messages.findById(20L)).thenReturn(Optional.of(assistant));
        when(messages.findLatestUserBefore(1L, 20L)).thenReturn(Optional.of(userMsg));
        when(rateLimiter.tryAcquire(100L)).thenReturn(true);
        when(streamLock.acquire(100L)).thenReturn(true);
        when(redis.readChunks(20L)).thenReturn(
            new ChunkContext(List.of(new ChunkDto("body", Map.of("file_name", "doc"))), "standalone q"));
        when(messages.save(any(Message.class))).thenAnswer(inv -> {
            Message m = inv.getArgument(0);
            if (m.getId() == null) m.setId(21L);
            return m;
        });
        when(promptBuilder.build(any(), anyString(), anyString())).thenReturn("PROMPT");
        when(openai.streamChat(anyString(), anyDouble())).thenReturn(Flux.empty());

        SseEmitter emitter = service.regenerate(1L, 20L);

        verify(rpc, never()).requestSearch(anyString(), anyString());
        verify(redis).deleteChunks(20L);
        verify(redis).cacheChunks(eq(21L), any(ChunkContext.class));
        verify(openai).streamChat(eq("PROMPT"), eq(0.7));
        verify(messages).deleteById(20L);
    }

    @Test
    void regenerate_cacheMiss_fallsBackToRpc() {
        Conversation conv = buildConv(1L, 100L);
        Message assistant = buildAssistant(20L, conv);
        Message userMsg = Message.builder().role("user").content("question").conversation(conv).build();
        userMsg.setId(19L);

        when(messages.findById(20L)).thenReturn(Optional.of(assistant));
        when(messages.findLatestUserBefore(1L, 20L)).thenReturn(Optional.of(userMsg));
        when(rateLimiter.tryAcquire(100L)).thenReturn(true);
        when(streamLock.acquire(100L)).thenReturn(true);
        when(redis.readChunks(20L)).thenReturn(null);
        when(rpc.requestSearch(eq("question"), eq("1"))).thenReturn(
            new com.example.chat.dto.SearchReplyMessage("cid", "chunks", "core_knowledge", "English",
                "rewritten q", false, false, List.of(), List.of(new ChunkDto("body", Map.of())), null));
        when(messages.save(any(Message.class))).thenAnswer(inv -> {
            Message m = inv.getArgument(0);
            if (m.getId() == null) m.setId(21L);
            return m;
        });
        when(promptBuilder.build(any(), anyString(), anyString())).thenReturn("PROMPT");
        when(openai.streamChat(anyString(), anyDouble())).thenReturn(Flux.empty());

        service.regenerate(1L, 20L);

        verify(rpc, times(1)).requestSearch("question", "1");
        verify(openai).streamChat(eq("PROMPT"), eq(0.7));
    }
}
```

- [ ] **Step 2: Run test — should fail because regenerate() doesn't exist**

Run: `cd backend && ./mvnw test -Dtest=ChatServiceRegenerateTest -q`
Expected: FAIL with `cannot find symbol method regenerate`.

- [ ] **Step 3: Implement `regenerate` in `ChatService.java`**

Add the following method to `ChatService` (place after `ask()`):

```java
public SseEmitter regenerate(Long conversationId, Long oldAssistantId) {
    SseEmitter emitter = new SseEmitter(60_000L);

    Message oldA = messages.findById(oldAssistantId)
        .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "assistant message not found"));
    if (!"assistant".equals(oldA.getRole())) {
        throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "messageId is not an assistant message");
    }
    Conversation conv = oldA.getConversation();
    if (!conversationId.equals(conv.getId())) {
        throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "message does not belong to conversation");
    }
    Long userId = conv.getUser().getId();

    Message userMsg = messages.findLatestUserBefore(conversationId, oldAssistantId)
        .orElseThrow(() -> new ResponseStatusException(HttpStatus.BAD_REQUEST, "no user message precedes the assistant"));

    if (!rateLimiter.tryAcquire(userId)) {
        sendEvent(emitter, "error", Map.of("message", "rate limit exceeded"));
        emitter.complete();
        return emitter;
    }
    if (!streamLock.acquire(userId)) {
        sendEvent(emitter, "error", Map.of("message", "a response is already streaming"));
        emitter.complete();
        return emitter;
    }

    com.example.chat.dto.ChunkContext ctx = redis.readChunks(oldAssistantId);
    boolean chunksReused = (ctx != null);
    String detectedLanguage = oldA.getDetectedLanguage();
    String intent = oldA.getIntent();
    boolean webSearchUsed = oldA.isWebSearchUsed();
    if (ctx == null) {
        com.example.chat.dto.SearchReplyMessage reply;
        try {
            reply = rpc.requestSearch(userMsg.getContent(), String.valueOf(conversationId));
        } catch (Exception ex) {
            sendEvent(emitter, "error", Map.of("message", nullSafe(ex.getMessage())));
            streamLock.release(userId);
            emitter.complete();
            return emitter;
        }
        ctx = new com.example.chat.dto.ChunkContext(reply.chunks(), reply.standaloneQuery());
        detectedLanguage = reply.detectedLanguage();
        intent = reply.intent();
        webSearchUsed = reply.webSearchUsed();
    }

    chunkSources.deleteByMessageIdIn(List.of(oldAssistantId));
    messages.deleteById(oldAssistantId);
    redis.deleteChunks(oldAssistantId);
    redis.invalidateHistory(conversationId);

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

    StringBuilder buffer = new StringBuilder();
    String prompt = promptBuilder.build(ctx.chunks(), ctx.standaloneQuery(), detectedLanguage);
    final String intentFinal = intent;

    Disposable sub = openai.streamChat(prompt, regenerateTemperature).subscribe(
        token -> {
            buffer.append(token);
            redis.appendStreamToken(conversationId, token);
            sendEvent(emitter, "delta", Map.of("content", token));
        },
        error -> {
            log.warn("OpenAI stream error during regenerate", error);
            sendEvent(emitter, "error", Map.of("message", error.getMessage()));
            persistAssistantAndCache(newA, buffer.toString(), buffer.length() > 0,
                                     conversationId, intentFinal);
            redis.deleteStream(conversationId);
            streamLock.release(userId);
            emitter.complete();
        },
        () -> {
            persistAssistantAndCache(newA, buffer.toString(), false,
                                     conversationId, intentFinal);
            redis.deleteStream(conversationId);
            sendEvent(emitter, "done", Map.of());
            streamLock.release(userId);
            emitter.complete();
        }
    );

    emitter.onTimeout(() -> {
        sub.dispose();
        persistAssistantAndCache(newA, buffer.toString(), buffer.length() > 0,
                                 conversationId, intentFinal);
        redis.deleteStream(conversationId);
        streamLock.release(userId);
    });
    emitter.onError(e -> {
        sub.dispose();
        persistAssistantAndCache(newA, buffer.toString(), buffer.length() > 0,
                                 conversationId, intentFinal);
        redis.deleteStream(conversationId);
        streamLock.release(userId);
    });
    emitter.onCompletion(() -> { if (!sub.isDisposed()) sub.dispose(); });

    return emitter;
}

private void persistChunkSourcesFromPayload(Message newA, List<com.example.chat.dto.ChunkDto> chunks) {
    if (chunks == null) return;
    for (com.example.chat.dto.ChunkDto c : chunks) {
        Map<String, Object> meta = c.metadata() == null ? Map.of() : c.metadata();
        String name = String.valueOf(meta.getOrDefault("file_name", ""));
        String url  = String.valueOf(meta.getOrDefault("url", ""));
        Integer pageNumber = toInt(meta.get("page_number"));
        Integer totalPages = toInt(meta.get("total_pages"));
        chunkSources.save(com.example.chat.domain.ChunkSource.builder()
            .message(newA)
            .name(name).url(url)
            .pageNumber(pageNumber).totalPages(totalPages)
            .build());
    }
}

private List<com.example.chat.dto.SourceDto> extractSourceDtos(List<com.example.chat.dto.ChunkDto> chunks) {
    if (chunks == null) return List.of();
    return chunks.stream().map(c -> {
        Map<String, Object> meta = c.metadata() == null ? Map.of() : c.metadata();
        return new com.example.chat.dto.SourceDto(
            String.valueOf(meta.getOrDefault("file_name", "")),
            String.valueOf(meta.getOrDefault("url", "")),
            toInt(meta.get("page_number")),
            toInt(meta.get("total_pages"))
        );
    }).toList();
}

private static Integer toInt(Object v) {
    if (v == null) return null;
    if (v instanceof Number n) return n.intValue();
    try { return Integer.parseInt(String.valueOf(v)); } catch (NumberFormatException e) { return null; }
}
```

Also add the field for the configurable temperature near the other `@Value` fields:

```java
@Value("${chat.regenerate.temperature:0.7}")
double regenerateTemperature = 0.7;
```

- [ ] **Step 4: Run regenerate test**

Run: `cd backend && ./mvnw test -Dtest=ChatServiceRegenerateTest -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Run full backend test suite**

Run: `cd backend && ./mvnw test -q`
Expected: BUILD SUCCESS.

- [ ] **Step 6: Commit**

```bash
git add backend/src/main/java/com/example/chat/chat/ChatService.java \
        backend/src/test/java/com/example/chat/chat/ChatServiceRegenerateTest.java
git commit -m "feat(chat): regenerate reuses cached chunks; bumps temperature to 0.7"
```

---

### Task 7: Add the controller endpoint

**Files:**
- Modify: `backend/src/main/java/com/example/chat/chat/ChatController.java`

- [ ] **Step 1: Add the mapping after the existing ask endpoint**

```java
@PostMapping(
    path = "/conversations/{convId}/messages/{assistantMsgId}/regenerate/stream",
    produces = MediaType.TEXT_EVENT_STREAM_VALUE
)
public SseEmitter regenerate(@PathVariable Long convId, @PathVariable Long assistantMsgId) {
    return service.regenerate(convId, assistantMsgId);
}
```

- [ ] **Step 2: Compile**

Run: `cd backend && ./mvnw compile -q`
Expected: BUILD SUCCESS.

- [ ] **Step 3: Commit**

```bash
git add backend/src/main/java/com/example/chat/chat/ChatController.java
git commit -m "feat(chat): POST /messages/{id}/regenerate/stream endpoint"
```

---

### Task 8: Wire config in `application.yml`

**Files:**
- Modify: `backend/src/main/resources/application.yml`

- [ ] **Step 1: Add chunks + regenerate sub-keys under `chat:`**

Locate the existing `chat:` block and add:

```yaml
chat:
  # ... existing keys ...
  chunks:
    ttl-seconds: ${CHAT_CHUNKS_TTL_SECONDS:3600}
  regenerate:
    temperature: ${CHAT_REGENERATE_TEMPERATURE:0.7}
```

- [ ] **Step 2: Boot test**

Run: `cd backend && ./mvnw test -q`
Expected: BUILD SUCCESS — Spring context loads with new keys.

- [ ] **Step 3: Commit**

```bash
git add backend/src/main/resources/application.yml
git commit -m "feat(chat): wire chat.chunks.ttl and chat.regenerate.temperature config"
```

---

## Phase 3 — Frontend

### Task 9: Extend `Message` and `SseHandler` types

**Files:**
- Modify: `frontend/src/types.ts`

- [ ] **Step 1: Update types**

Add `streamingMode` to `Message`, extend `SseHandler.onMeta` payload:

```typescript
export interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  intent?: string;
  detectedLanguage?: string;
  webSearchUsed?: boolean;
  stopped?: boolean;
  createdAt: string;
  sources: Source[];
  streamingMode?: 'ask' | 'regenerate'; // NEW — only meaningful while id < 0 (placeholder)
}

export type SseMeta = {
  intent: string;
  detectedLanguage: string;
  sources: Source[];
  webSearchUsed: boolean;
  assistantMessageId: number;
  regenerated?: boolean;     // NEW
  chunksReused?: boolean;    // NEW
};

export type SseHandler = {
  onMeta: (meta: SseMeta) => void;
  onDelta: (token: string) => void;
  onDone: () => void;
  onError: (err: string) => void;
};
```

If `SseHandler.onMeta` was previously inlined in `chatApi.ts`, update the import there to reference `SseMeta` from `types.ts`.

- [ ] **Step 2: Type check**

Run: `cd frontend && npm run typecheck` (or `npx tsc --noEmit`)
Expected: clean — any type mismatches will be fixed in following tasks.

- [ ] **Step 3: Commit (will likely have downstream type errors; fix in later tasks)**

```bash
git add frontend/src/types.ts
git commit -m "feat(frontend): SseMeta type with regenerated/chunksReused; streamingMode on Message"
```

---

### Task 10: Add `regenerateStream` to `chatApi.ts`

**Files:**
- Modify: `frontend/src/api/chatApi.ts`

- [ ] **Step 1: Add a new function that mirrors `askStream` but POSTs without a body**

Below `askStream` add:

```typescript
export async function regenerateStream(
  conversationId: number,
  assistantMessageId: number,
  handlers: SseHandler,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch(
    `${BASE}/conversations/${conversationId}/messages/${assistantMessageId}/regenerate/stream`,
    {
      method: 'POST',
      headers: { Accept: 'text/event-stream' },
      signal,
    },
  );
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
          } catch { /* skip */ }
        }
      }
    }
  } catch (e: any) {
    if (e.name !== 'AbortError') handlers.onError(String(e.message ?? e));
  }
}
```

(The SSE parsing logic is duplicated intentionally — DRY can come later if a third stream endpoint is added.)

- [ ] **Step 2: Type check**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api/chatApi.ts
git commit -m "feat(frontend): regenerateStream API helper"
```

---

### Task 11: Add `regenerate` + `regenerating` to `useChatStream`

**Files:**
- Modify: `frontend/src/hooks/useChatStream.ts`

- [ ] **Step 1: Update hook**

```typescript
import { useRef, useState } from 'react';
import { askStream, regenerateStream } from '../api/chatApi';
import type { Message, SseMeta, Source } from '../types';

type UpdatePatch = {
  content?: string;
  sources?: Source[];
  intent?: string;
  detectedLanguage?: string;
  webSearchUsed?: boolean;
};

export function useChatStream() {
  const [streaming, setStreaming] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  async function start(
    conversationId: number,
    userMessage: string,
    onAppendUser: (m: Message) => void,
    onPlaceholder: (p: Message) => void,
    onUpdate: (placeholderId: number, patch: UpdatePatch) => void,
    onError: (err: string) => void,
  ) {
    if (streaming || regenerating) return;
    setStreaming(true);
    const tempUserId = -Date.now();
    onAppendUser({ id: tempUserId, role: 'user', content: userMessage,
                   createdAt: new Date().toISOString(), sources: [] });
    const placeholderId = -(Date.now() + 1);
    let buffer = '';
    onPlaceholder({ id: placeholderId, role: 'assistant', content: '',
                    createdAt: new Date().toISOString(), sources: [],
                    streamingMode: 'ask' });

    const controller = new AbortController();
    controllerRef.current = controller;
    try {
      await askStream(conversationId, userMessage, {
        onMeta: (meta: SseMeta) => onUpdate(placeholderId, {
          sources: meta.sources, intent: meta.intent,
          detectedLanguage: meta.detectedLanguage, webSearchUsed: meta.webSearchUsed,
        }),
        onDelta: (tok) => { buffer += tok; onUpdate(placeholderId, { content: buffer }); },
        onDone: () => {},
        onError: (msg) => onError(msg),
      }, controller.signal);
    } finally {
      setStreaming(false);
      controllerRef.current = null;
    }
  }

  async function regenerate(
    conversationId: number,
    assistantMessageId: number,
    onPlaceholder: (p: Message) => void,
    onUpdate: (placeholderId: number, patch: UpdatePatch) => void,
    onError: (err: string) => void,
  ) {
    if (streaming || regenerating) return;
    setRegenerating(true);
    const placeholderId = -Date.now();
    let buffer = '';
    onPlaceholder({ id: placeholderId, role: 'assistant', content: '',
                    createdAt: new Date().toISOString(), sources: [],
                    streamingMode: 'regenerate' });

    const controller = new AbortController();
    controllerRef.current = controller;
    try {
      await regenerateStream(conversationId, assistantMessageId, {
        onMeta: (meta: SseMeta) => onUpdate(placeholderId, {
          sources: meta.sources, intent: meta.intent,
          detectedLanguage: meta.detectedLanguage, webSearchUsed: meta.webSearchUsed,
        }),
        onDelta: (tok) => { buffer += tok; onUpdate(placeholderId, { content: buffer }); },
        onDone: () => {},
        onError: (msg) => onError(msg),
      }, controller.signal);
    } finally {
      setRegenerating(false);
      controllerRef.current = null;
    }
  }

  function stop() { controllerRef.current?.abort(); }

  return { streaming, regenerating, start, regenerate, stop };
}
```

- [ ] **Step 2: Type check**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean (downstream consumer change comes next).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/useChatStream.ts
git commit -m "feat(frontend): useChatStream.regenerate + regenerating flag"
```

---

### Task 12: Rewrite `ChatWindow` regenerate handler & disable composer

**Files:**
- Modify: `frontend/src/components/ChatWindow.tsx`

- [ ] **Step 1: Replace `regenerate` and update composer**

Apply these edits:

1. Destructure new fields from the hook:
   ```tsx
   const { streaming, regenerating, start, regenerate: regenerateStream, stop } = useChatStream();
   ```

2. Replace lines 46-54 with:
   ```tsx
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

3. Update composer (around line 96-108): disable while either stream is active, and show stop while either is active:
   ```tsx
   <textarea
     value={input}
     onChange={(e) => setInput(e.target.value)}
     placeholder="Ask anything…"
     rows={2}
     disabled={streaming || regenerating}
     onKeyDown={(e) => {
       if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (input.trim()) send(input.trim()); }
     }}
   />
   {(streaming || regenerating)
     ? <button type="button" className="stop-btn" onClick={stop}>■ Stop</button>
     : <button type="submit" disabled={!input.trim()}>Send</button>}
   ```

4. Update the `MessageBubble` placeholder check (line 82) — placeholders for either flow should be treated the same:
   ```tsx
   const isPlaceholder = m.id < 0 && m.role === 'assistant';
   ```
   And pass through `streamingMode`:
   ```tsx
   <MessageBubble
     key={m.id}
     message={m}
     streaming={(streaming || regenerating) && isPlaceholder}
     streamingMode={m.streamingMode}
     isLastAssistant={m.id === lastAssistantId && !streaming && !regenerating}
     onEdit={m.role === 'user' ? () => { setEditingId(m.id); setEditValue(m.content); } : undefined}
     onRegenerate={m.role === 'assistant' && !streaming && !regenerating ? () => regenerate(m) : undefined}
   />
   ```

- [ ] **Step 2: Type check**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChatWindow.tsx
git commit -m "fix(frontend): regenerate no longer duplicates user msg; composer disabled during regen"
```

---

### Task 13: `MessageBubble` — show "regenerating…" cursor & accept new prop

**Files:**
- Modify: `frontend/src/components/MessageBubble.tsx`

- [ ] **Step 1: Add prop and conditional placeholder text**

```tsx
interface Props {
  message: Message;
  streaming: boolean;
  streamingMode?: 'ask' | 'regenerate';
  isLastAssistant: boolean;
  onEdit?: () => void;
  onRegenerate?: () => void;
}

export function MessageBubble({ message, streaming, streamingMode, isLastAssistant, onEdit, onRegenerate }: Props) {
  const isUser = message.role === 'user';
  const placeholderLabel = streamingMode === 'regenerate' ? 'regenerating…' : 'thinking…';
  const showPlaceholderLabel = streaming && !isUser && !message.content;

  return (
    <div className={`bubble ${isUser ? 'user' : 'assistant'}`}>
      <div className="bubble-body">
        {showPlaceholderLabel
          ? <span className="placeholder-label">{placeholderLabel}</span>
          : <ReactMarkdown remarkPlugins={[remarkGfm]} /* ... existing props ... */>
              {message.content || ''}
            </ReactMarkdown>}
        {streaming && !isUser && message.content && <span className="cursor">▋</span>}
      </div>
      {/* ... rest unchanged: sources block, stopped marker, actions ... */}
    </div>
  );
}
```

- [ ] **Step 2: Type check**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 3: Visual smoke (manual)**

Start dev: `cd frontend && npm run dev`
Send a question, click Regenerate on the answer. The placeholder bubble should briefly say "regenerating…" then stream tokens. The user bubble must NOT duplicate.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/MessageBubble.tsx
git commit -m "feat(frontend): MessageBubble shows regenerating… label during regen stream"
```

---

## Phase 4 — Python prompts

### Task 14: Replace `SUMMARIZE_PROMPT_TEMPLATE` (Recomp-style)

**Files:**
- Modify: `src/rag/search/prompt.py:23-61`

- [ ] **Step 1: Replace the template**

Edit `src/rag/search/prompt.py`. Replace the existing `SUMMARIZE_PROMPT_TEMPLATE` definition (lines 23-61) with:

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

- [ ] **Step 2: Lint / sanity check**

Run: `python -c "from src.rag.search.prompt import SUMMARIZE_PROMPT_TEMPLATE; print(SUMMARIZE_PROMPT_TEMPLATE.format(query='q', conversation_history='User: hi'))"`
Expected: prints the formatted prompt without `KeyError`. The two `{query}` and `{conversation_history}` placeholders resolve.

- [ ] **Step 3: Commit**

```bash
git add src/rag/search/prompt.py
git commit -m "feat(prompt): Recomp-style SUMMARIZE_PROMPT_TEMPLATE (query-focused, single-string)"
```

---

### Task 15: Update `_summarize_history` and `afetch_chat_history` signatures

**Files:**
- Modify: `src/rag/search/entrypoint.py:53-91`

- [ ] **Step 1: Add `query` parameter through the chain**

Replace the two function definitions:

```python
async def _summarize_history(llm: AzureChatClient, history_string: str, query: str) -> str:
    """Recomp-style query-focused summary. Returns plain string (possibly empty)."""
    prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
        conversation_history=history_string,
        query=query,
    )
    raw = (await llm.ainvoke(prompt)).strip()
    try:
        result = parse_json_response(raw).get("summary", "")
        return result if isinstance(result, str) else ""
    except Exception:
        logger.warning("Failed to parse summary response, returning empty. Raw: %s", raw)
        return ""


@alog_function_call
async def afetch_chat_history(
    llm: AzureChatClient,
    conversation_id: str | None = None,
    query: str = "",
) -> str:
    if not conversation_id or not conversation_id.strip():
        return ""
    if not query or not query.strip():
        return ""

    messages = await fetch_raw_chat_history(conversation_id)
    if not messages:
        return ""

    filtered = _filter_core_knowledge_pairs(messages)
    if not filtered:
        return ""

    history_string = _format_history(filtered)
    summary = await _summarize_history(llm, history_string, query)
    return summary
```

- [ ] **Step 2: Compile-check by import**

Run: `python -c "from src.rag.search.entrypoint import afetch_chat_history; import inspect; print(inspect.signature(afetch_chat_history))"`
Expected: prints `(llm: src.rag.llm.chat_llm.AzureChatClient, conversation_id: str | None = None, query: str = '') -> str`.

- [ ] **Step 3: Commit (do not run tests yet — call sites still broken)**

```bash
git add src/rag/search/entrypoint.py
git commit -m "feat(rag): afetch_chat_history takes query for Recomp summarize"
```

---

### Task 16: Update `afetch_chat_history` call sites

**Files:**
- Modify: `src/rag/search/pipeline.py:152` (and any other call sites)
- Modify: `src/rag/search/agent/nodes.py:76`

- [ ] **Step 1: Pass `query=user_input` at every call site**

In `src/rag/search/pipeline.py` near line 152:

```python
if has_history:
    chat_history = await afetch_chat_history(llm, conversation_id, query=user_input)
    standalone_query = await Reflection(llm).areflect(chat_history, user_input)
else:
    standalone_query = await Reflection(llm).areflect("", user_input)
```

In `src/rag/search/agent/nodes.py` near line 76:

```python
if has_history:
    chat_history = await afetch_chat_history(llm, conversation_id, query=user_input)
    standalone_query = await Reflection(llm).areflect(chat_history, user_input)
else:
    standalone_query = await Reflection(llm).areflect("", user_input)
```

- [ ] **Step 2: Grep for any other call sites**

Run: `grep -rn "afetch_chat_history(" src/`
Expected: only the two call sites above plus the definition. If any other call site exists, update it analogously.

- [ ] **Step 3: Run Python tests**

Run: `pytest -q` (from repo root)
Expected: PASS — or, if there is no test for entrypoint, at least no import errors.

- [ ] **Step 4: Commit**

```bash
git add src/rag/search/pipeline.py src/rag/search/agent/nodes.py
git commit -m "feat(rag): pass user_input as query into afetch_chat_history"
```

---

### Task 17: Replace `CLARITY_CHECK_PROMPT` (softened)

**Files:**
- Modify: `src/rag/search/prompt.py:62-131`

- [ ] **Step 1: Replace the template**

Replace `CLARITY_CHECK_PROMPT` with:

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

- [ ] **Step 2: Format-check**

Run: `python -c "from src.rag.search.prompt import CLARITY_CHECK_PROMPT; print(CLARITY_CHECK_PROMPT.format(standalone_query='hi', response_language='Vietnamese')[:200])"`
Expected: prints first 200 chars without `KeyError`.

- [ ] **Step 3: Smoke test the pipeline against a known clear query**

If the project has a manual integration test runner, hit `pipeline._acheck_input_clarity` or run the Python service end-to-end with a clear question (e.g., "What is adverse selection?") and verify the response is CLEAR. Otherwise spot-check with the Java backend.

- [ ] **Step 4: Commit**

```bash
git add src/rag/search/prompt.py
git commit -m "feat(prompt): friendlier CLARITY_CHECK_PROMPT, biases toward CLEAR"
```

---

## Phase 5 — End-to-end verification

### Task 18: Manual smoke test

**Files:** none (manual verification)

- [ ] **Step 1: Start backend, frontend, Python worker**

```bash
# Terminal A
cd backend && ./mvnw spring-boot:run
# Terminal B
cd frontend && npm run dev
# Terminal C (Python worker — repo's existing entrypoint)
python -m src.apis.app_controller   # adjust to actual entrypoint
```

- [ ] **Step 2: In the browser, exercise these scenarios**

1. **Happy path:** ask "What is the Law of Large Numbers?" → answer streams. Click ↻ Regenerate on that answer. New answer streams in place; **only one user bubble**, **only one assistant bubble** at the end. Composer disabled and shows ■ Stop while regenerating; placeholder shows "regenerating…".
2. **Different answer:** the regenerated text should differ from the original (temperature 0.7 vs 0.7 default + LLM stochasticity — at minimum word choice should vary).
3. **Cache-miss fallback:** flush the Redis chunk key (`redis-cli DEL chat:chunks:<id>`) before clicking Regenerate. Confirm a new answer still streams; check Java logs for an RPC search call.
4. **Edit user message:** edit a user message and submit. Confirm assistant chain after that user msg is rebuilt as before — no regression in the edit flow (which uses `commitEdit` → `send`).
5. **CLARITY softening:** ask a borderline-fuzzy query like "tell me risk thing" — should now lean toward CLEAR or a single rephrase, not gibberish-suggestions.
6. **Recomp summarize:** in a multi-turn chat, watch logs for the new summarize prompt. Verify the LLM returns `{"summary": "..."}` (string, not list). Verify downstream Reflection still produces a sensible standalone query.

- [ ] **Step 3: Document any deviations**

If a scenario fails, file a follow-up task referencing this plan; do NOT mark the plan complete.

- [ ] **Step 4: Commit nothing, but mark complete in your own task tracker**

---

## Self-review checklist

- [x] Spec §1.1 (FE duplicate user msg) → covered by Task 12.
- [x] Spec §1.2 (re-search) → covered by Tasks 1-8 (chunk cache + regenerate endpoint).
- [x] Spec §1.3 (Recomp summarize) → covered by Tasks 14-16.
- [x] Spec §1.4 (clarity softening) → covered by Task 17.
- [x] Spec §3 architecture (delete A, persist A', cache new chunks) → Task 6.
- [x] Spec §4 API contract (path, events, payload fields) → Tasks 6, 7.
- [x] Spec §5.1 ChunkContext + Redis methods → Tasks 1, 2.
- [x] Spec §5.2 ask() caches chunks → Task 3.
- [x] Spec §5.4 streamChat temperature overload → Task 5.
- [x] Spec §5.5 findLatestUserBefore → Task 4.
- [x] Spec §5.7 application.yml keys → Task 8.
- [x] Spec §6 frontend changes (types, api, hook, ChatWindow, MessageBubble) → Tasks 9-13.
- [x] Spec §7 SUMMARIZE rewrite + caller signature update + call sites → Tasks 14-16.
- [x] Spec §8 CLARITY rewrite (schema preserved) → Task 17.
- [x] Spec §12 acceptance criteria → exercised in Task 18.
- [x] No "TBD"/"TODO"/"add appropriate" placeholders.
- [x] Type consistency: `ChunkContext` field names (`chunks`, `standaloneQuery`) used consistently across DTO, Redis methods, regenerate logic, and tests. `streamingMode: 'ask' | 'regenerate'` used consistently in `types.ts`, `useChatStream`, `MessageBubble`. Method `findLatestUserBefore` used in both repository and service.
