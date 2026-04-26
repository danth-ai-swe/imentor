package com.example.chat.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ChunkContext(List<ChunkDto> chunks, String standaloneQuery) {}
