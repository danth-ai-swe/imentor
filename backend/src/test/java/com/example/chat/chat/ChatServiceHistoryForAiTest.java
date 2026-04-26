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
