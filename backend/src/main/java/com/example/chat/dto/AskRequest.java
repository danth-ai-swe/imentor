package com.example.chat.dto;

import jakarta.validation.constraints.NotBlank;

public record AskRequest(@NotBlank String message) {}
