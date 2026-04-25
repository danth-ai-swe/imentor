package com.example.chat.dto;

import jakarta.validation.constraints.NotNull;

public record CreateConversationRequest(@NotNull Long userId, String title) {}
