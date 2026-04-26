package com.example.chat.redis;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ValueOperations;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;

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
        Clock fixed = Clock.fixed(Instant.parse("2026-04-26T00:00:30Z"), ZoneOffset.UTC);
        limiter = new RateLimiter(template, 15, fixed);
    }

    @Test
    void firstHitInBucketSetsExpiryAndPasses() {
        when(valueOps.increment(anyString())).thenReturn(1L);
        boolean ok = limiter.tryAcquire(42L);
        assertThat(ok).isTrue();
        verify(template).expire(startsWith("chat:rate:42:"), eq(Duration.ofSeconds(60)));
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
