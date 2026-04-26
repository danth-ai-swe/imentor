package com.example.chat.dto;

import java.util.List;

public record HistoryForAiResponse(Long conversationId, List<HistoryMessageDto> messages) {}
