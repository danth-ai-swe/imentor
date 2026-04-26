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
