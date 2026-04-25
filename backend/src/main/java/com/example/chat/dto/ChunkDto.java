package com.example.chat.dto;

import java.util.Map;

public record ChunkDto(String text, Map<String, Object> metadata) {}
