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
