package com.example.chat.dto;

import java.util.List;

public record MessagesResponse(Long conversationId, List<MessageDto> messages) {}
