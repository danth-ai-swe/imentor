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
        service = new ChatRedisService(template, mapper, 1800, 50, 300, 3600);
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
