package com.example.chat.redis;

import org.springframework.stereotype.Service;

@Service
public class StreamLock {
    public boolean acquire(Long userId) { return true; }
    public void release(Long userId) {}
}
