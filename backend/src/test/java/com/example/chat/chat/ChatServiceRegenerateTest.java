package com.example.chat.chat;

import com.example.chat.domain.Conversation;
import com.example.chat.domain.Message;
import com.example.chat.domain.User;
import com.example.chat.dto.ChunkContext;
import com.example.chat.dto.ChunkDto;
import com.example.chat.dto.SearchReplyMessage;
import com.example.chat.dto.SourceDto;
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

import java.time.Instant;
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

    private User buildUser(long userId) {
        User u = User.builder().username("alice").createdAt(Instant.now()).build();
        u.setId(userId);
        return u;
    }

    private Conversation buildConv(long convId, User u) {
        Conversation c = Conversation.builder().user(u).title("t").build();
        c.setId(convId);
        return c;
    }

    private Message buildAssistant(long id, Conversation conv) {
        Message m = Message.builder()
            .role("assistant").content("old answer")
            .intent("core_knowledge").detectedLanguage("English").webSearchUsed(false)
            .conversation(conv).build();
        m.setId(id);
        return m;
    }

    @Test
    void regenerate_cacheHit_doesNotCallRpc() {
        User u = buildUser(100L);
        Conversation conv = buildConv(1L, u);
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
        User u = buildUser(100L);
        Conversation conv = buildConv(1L, u);
        Message assistant = buildAssistant(20L, conv);
        Message userMsg = Message.builder().role("user").content("question").conversation(conv).build();
        userMsg.setId(19L);

        when(messages.findById(20L)).thenReturn(Optional.of(assistant));
        when(messages.findLatestUserBefore(1L, 20L)).thenReturn(Optional.of(userMsg));
        when(rateLimiter.tryAcquire(100L)).thenReturn(true);
        when(streamLock.acquire(100L)).thenReturn(true);
        when(redis.readChunks(20L)).thenReturn(null);
        when(rpc.requestSearch(eq("question"), eq("1"))).thenReturn(
            new SearchReplyMessage("cid", "chunks", "core_knowledge", "English",
                "rewritten q", false, false, List.of(),
                List.of(new ChunkDto("body", Map.of())), null));
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
