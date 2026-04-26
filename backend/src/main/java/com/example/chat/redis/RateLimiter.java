package com.example.chat.redis;

import org.springframework.stereotype.Service;

@Service
public class RateLimiter {
    public boolean tryAcquire(Long userId) { return true; }
}
