package com.example.chat.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record SearchReplyMessage(
    String correlationId,
    String mode,                // "chunks" | "static"
    String intent,
    String detectedLanguage,
    String standaloneQuery,
    boolean answerSatisfied,
    boolean webSearchUsed,
    List<SourceDto> sources,
    List<ChunkDto> chunks,
    String response
) {}
