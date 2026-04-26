package com.example.chat.chat;

import com.example.chat.domain.*;
import com.example.chat.dto.*;
import com.example.chat.messaging.SearchRpcClient;
import com.example.chat.openai.OpenAiChatService;
import com.example.chat.openai.PromptBuilder;
import com.example.chat.redis.ChatRedisService;
import com.example.chat.redis.RateLimiter;
import com.example.chat.redis.StreamLock;
import com.example.chat.repository.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import reactor.core.Disposable;

import java.io.IOException;
import java.util.List;
import java.util.Map;

@Service
public class ChatService {

    private static final Logger log = LoggerFactory.getLogger(ChatService.class);
    private static final ObjectMapper JSON = new ObjectMapper();

    private final UserRepository users;
    private final ConversationRepository conversations;
    private final MessageRepository messages;
    private final ChunkSourceRepository chunkSources;

    private final SearchRpcClient rpc;
    private final OpenAiChatService openai;
    private final PromptBuilder promptBuilder;
    private final ChatRedisService redis;
    private final RateLimiter rateLimiter;
    private final StreamLock streamLock;

    @Value("${chat.history.default-limit:20}") int defaultHistoryLimit = 20;
    @Value("${chat.history.max-limit:50}")     int maxHistoryLimit = 50;

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

    @Transactional
    public UserDto getOrCreateUser(String username) {
        User u = users.findByUsername(username).orElseGet(() -> {
            try {
                return users.saveAndFlush(User.builder().username(username).build());
            } catch (DataIntegrityViolationException e) {
                return users.findByUsername(username).orElseThrow();
            }
        });
        return new UserDto(u.getId(), u.getUsername());
    }

    @Transactional(readOnly = true)
    public List<ConversationDto> listConversations(String username) {
        return conversations.findByUserUsernameOrderByUpdatedAtDesc(username).stream()
            .map(c -> new ConversationDto(c.getId(), c.getTitle(), c.getCreatedAt(), c.getUpdatedAt()))
            .toList();
    }

    @Transactional
    public ConversationDto createConversation(Long userId, String title) {
        User user = users.findById(userId).orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "user not found"));
        Conversation c = conversations.save(Conversation.builder()
            .user(user)
            .title(title == null || title.isBlank() ? "New chat" : title)
            .build());
        return new ConversationDto(c.getId(), c.getTitle(), c.getCreatedAt(), c.getUpdatedAt());
    }

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

    @Transactional
    public void deleteFromMessage(Long conversationId, Long messageId) {
        List<Message> toDelete = messages.findByConversationIdAndIdGreaterThanEqual(conversationId, messageId);
        if (toDelete.isEmpty()) return;
        List<Long> ids = toDelete.stream().map(Message::getId).toList();
        chunkSources.deleteByMessageIdIn(ids);
        messages.deleteAll(toDelete);
        redis.invalidateHistory(conversationId);
    }

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

    // Note: not @Transactional — called from Reactor scheduler threads where
    // Spring's ThreadLocal-based tx propagation does not apply. messages.save()
    // wraps each call in its own tx via SimpleJpaRepository.
    void persistAssistantAndCache(Message m, String content, boolean stopped,
                                  Long conversationId, String intent) {
        m.setContent(content);
        m.setStopped(stopped);
        Message saved = messages.save(m);
        redis.appendHistory(conversationId, new HistoryMessageDto(
            "assistant", content, intent, saved.getCreatedAt()));
    }

    private static String nullSafe(String s) { return s == null ? "" : s; }

    private void sendEvent(SseEmitter emitter, String name, Object data) {
        try {
            emitter.send(SseEmitter.event().name(name).data(JSON.writeValueAsString(data)));
        } catch (IOException e) {
            log.debug("SSE send failed (client disconnected): {}", e.getMessage());
        }
    }
}
