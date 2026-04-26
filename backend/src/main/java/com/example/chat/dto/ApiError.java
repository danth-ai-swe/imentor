package com.example.chat.dto;

public record ApiError(boolean success, String error) {
    public static ApiError of(String message) {
        return new ApiError(false, message);
    }
}
