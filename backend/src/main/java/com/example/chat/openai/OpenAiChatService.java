package com.example.chat.openai;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.Map;

@Service
public class OpenAiChatService {

    private static final Logger log = LoggerFactory.getLogger(OpenAiChatService.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final WebClient webClient;
    private final String model;

    public OpenAiChatService(
        @Qualifier("openAiWebClient") WebClient webClient,
        @Value("${openai.model}") String model
    ) {
        this.webClient = webClient;
        this.model = model;
    }

    public Flux<String> streamChat(String prompt) {
        Map<String, Object> body = Map.of(
            "model", model,
            "stream", true,
            "messages", List.of(Map.of("role", "user", "content", prompt))
        );

        return webClient.post()
            .uri("/v1/chat/completions")
            .contentType(MediaType.APPLICATION_JSON)
            .bodyValue(body)
            .retrieve()
            .bodyToFlux(String.class) // each "data: {...}" SSE event arrives as a String
            .filter(line -> line != null && !line.isBlank())
            .takeWhile(line -> !"[DONE]".equals(line.trim()))
            .mapNotNull(this::extractContent)
            .filter(s -> !s.isEmpty());
    }

    private String extractContent(String dataLine) {
        try {
            JsonNode root = MAPPER.readTree(dataLine);
            JsonNode delta = root.path("choices").path(0).path("delta").path("content");
            return delta.isMissingNode() || delta.isNull() ? "" : delta.asText();
        } catch (Exception e) {
            log.warn("Failed to parse OpenAI SSE chunk: {}", dataLine);
            return "";
        }
    }
}
