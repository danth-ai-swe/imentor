package com.example.chat.dto;

public record SearchRequestMessage(String correlationId, String userMessage, String conversationId) {}
