package com.example.chat.chat;

import com.example.chat.domain.*;
import com.example.chat.dto.*;
import com.example.chat.messaging.SearchRpcClient;
import com.example.chat.openai.OpenAiChatService;
import com.example.chat.openai.PromptBuilder;
import com.example.chat.repository.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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

    public ChatService(UserRepository users, ConversationRepository conversations,
                       MessageRepository messages, ChunkSourceRepository chunkSources,
                       SearchRpcClient rpc, OpenAiChatService openai, PromptBuilder promptBuilder) {
        this.users = users;
        this.conversations = conversations;
        this.messages = messages;
        this.chunkSources = chunkSources;
        this.rpc = rpc;
        this.openai = openai;
        this.promptBuilder = promptBuilder;
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
    public List<MessageDto> listMessages(Long conversationId) {
        return messages.findByConversationIdOrderByCreatedAtAsc(conversationId).stream()
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
    }

    @Transactional
    public void deleteFromMessage(Long conversationId, Long messageId) {
        List<Message> toDelete = messages.findByConversationIdAndIdGreaterThanEqual(conversationId, messageId);
        if (toDelete.isEmpty()) return;
        List<Long> ids = toDelete.stream().map(Message::getId).toList();
        chunkSources.deleteByMessageIdIn(ids);
        messages.deleteAll(toDelete);
    }

    public SseEmitter ask(Long conversationId, String userMessage) {
        SseEmitter emitter = new SseEmitter(60_000L);

        Conversation conv = conversations.findById(conversationId)
            .orElseThrow(() -> new IllegalArgumentException("conversation not found"));

        // 1) Persist the user message (committed before RPC — on RPC failure the user
        // message remains; assistant shell is not created. Frontend reload will show
        // an orphan user msg in that branch — accepted tradeoff for demo scope.)
        messages.save(Message.builder()
            .conversation(conv).role("user").content(userMessage).build());
        conversations.save(conv); // touches updatedAt

        // 2) RPC → Python
        SearchReplyMessage reply;
        try {
            reply = rpc.requestSearch(userMessage, String.valueOf(conversationId));
        } catch (Exception ex) {
            sendEvent(emitter, "error", Map.of("message", nullSafe(ex.getMessage())));
            emitter.complete();
            return emitter;
        }

        // 3) Persist empty assistant message
        Message assistantMsg = messages.save(Message.builder()
            .conversation(conv).role("assistant").content("")
            .intent(reply.intent())
            .detectedLanguage(reply.detectedLanguage())
            .webSearchUsed(reply.webSearchUsed())
            .build());

        // 4) Persist chunk sources (if any)
        if (reply.sources() != null) {
            for (SourceDto s : reply.sources()) {
                chunkSources.save(ChunkSource.builder()
                    .message(assistantMsg)
                    .name(s.name()).url(s.url())
                    .pageNumber(s.pageNumber()).totalPages(s.totalPages())
                    .build());
            }
        }

        // 5) Emit meta event
        sendEvent(emitter, "meta", Map.of(
            "intent", nullSafe(reply.intent()),
            "detectedLanguage", nullSafe(reply.detectedLanguage()),
            "sources", reply.sources() == null ? java.util.List.of() : reply.sources(),
            "webSearchUsed", reply.webSearchUsed(),
            "assistantMessageId", assistantMsg.getId()
        ));

        // 6) Branch on mode
        if ("static".equals(reply.mode())) {
            String text = reply.response() == null ? "" : reply.response();
            sendEvent(emitter, "delta", Map.of("content", text));
            assistantMsg.setContent(text);
            messages.save(assistantMsg);
            sendEvent(emitter, "done", Map.of());
            emitter.complete();
            return emitter;
        }

        // mode == "chunks": stream from OpenAI
        StringBuilder buffer = new StringBuilder();
        String prompt = promptBuilder.build(reply.chunks(), reply.standaloneQuery(), reply.detectedLanguage());

        Disposable sub = openai.streamChat(prompt).subscribe(
            token -> {
                buffer.append(token);
                sendEvent(emitter, "delta", Map.of("content", token));
            },
            error -> {
                log.warn("OpenAI stream error", error);
                sendEvent(emitter, "error", Map.of("message", error.getMessage()));
                persistAssistant(assistantMsg, buffer.toString(), buffer.length() > 0);
                emitter.complete();
            },
            () -> {
                persistAssistant(assistantMsg, buffer.toString(), false);
                sendEvent(emitter, "done", Map.of());
                emitter.complete();
            }
        );

        emitter.onTimeout(() -> { sub.dispose(); persistAssistant(assistantMsg, buffer.toString(), buffer.length() > 0); });
        emitter.onError(e -> {
            sub.dispose();
            persistAssistant(assistantMsg, buffer.toString(), buffer.length() > 0);
        });
        emitter.onCompletion(() -> {
            if (!sub.isDisposed()) sub.dispose();
        });

        return emitter;
    }

    // Note: not @Transactional — called from Reactor scheduler threads where
    // Spring's ThreadLocal-based tx propagation does not apply. messages.save()
    // wraps each call in its own tx via SimpleJpaRepository.
    void persistAssistant(Message m, String content, boolean stopped) {
        m.setContent(content);
        m.setStopped(stopped);
        messages.save(m);
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
