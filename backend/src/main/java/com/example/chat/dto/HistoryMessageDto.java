package com.example.chat.dto;

import java.time.Instant;

public record HistoryMessageDto(
    String role,
    String content,
    String intent,
    Instant createdAt
) {}
