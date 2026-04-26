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
