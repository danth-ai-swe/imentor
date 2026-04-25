package com.example.chat.dto;

import java.time.Instant;
import java.util.List;

public record MessageDto(
    Long id,
    String role,
    String content,
    String intent,
    String detectedLanguage,
    boolean webSearchUsed,
    boolean stopped,
    Instant createdAt,
    List<SourceDto> sources
) {}
