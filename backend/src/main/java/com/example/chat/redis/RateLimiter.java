package com.example.chat.redis;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DataAccessException;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Clock;
import java.time.Duration;

@Service
public class RateLimiter {

    private static final Logger log = LoggerFactory.getLogger(RateLimiter.class);

    private final StringRedisTemplate template;
    private final int perUserPerMinute;
    private final Clock clock;

    @Autowired
    public RateLimiter(StringRedisTemplate template,
                       @Value("${chat.rate-limit.per-user-per-minute:15}") int perUserPerMinute) {
        this(template, perUserPerMinute, Clock.systemUTC());
    }

    // visible for tests
    RateLimiter(StringRedisTemplate template, int perUserPerMinute, Clock clock) {
        this.template = template;
        this.perUserPerMinute = perUserPerMinute;
        this.clock = clock;
    }

    public boolean tryAcquire(Long userId) {
        long bucket = clock.instant().getEpochSecond() / 60;
        String key = "chat:rate:" + userId + ":" + bucket;
        try {
            Long count = template.opsForValue().increment(key);
            if (count != null && count == 1L) {
                template.expire(key, Duration.ofSeconds(60));
            }
            return count != null && count <= perUserPerMinute;
        } catch (DataAccessException e) {
            log.warn("RateLimiter Redis failure for user={}, failing open: {}", userId, e.getMessage());
            return true;
        }
    }
}
