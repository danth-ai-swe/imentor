package com.example.chat.dto;

import jakarta.validation.constraints.NotBlank;

public record CreateUserRequest(@NotBlank String username) {}
