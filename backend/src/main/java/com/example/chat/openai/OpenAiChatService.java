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
    private final String deployment;
    private final String apiVersion;
    private final double temperature;
    private final double topP;
    private final int maxTokens;

    public OpenAiChatService(
        @Qualifier("openAiWebClient") WebClient webClient,
        @Value("${openai.chat-model}") String deployment,
        @Value("${openai.api-version}") String apiVersion,
        @Value("${openai.temperature:0.7}") double temperature,
        @Value("${openai.top-p:0.95}") double topP,
        @Value("${openai.max-tokens:3500}") int maxTokens
    ) {
        this.webClient = webClient;
        this.deployment = deployment;
        this.apiVersion = apiVersion;
        this.temperature = temperature;
        this.topP = topP;
        this.maxTokens = maxTokens;
    }

    public Flux<String> streamChat(String prompt) {
        Map<String, Object> body = Map.of(
            "messages", List.of(Map.of("role", "user", "content", prompt)),
            "stream", true,
            "temperature", temperature,
            "top_p", topP,
            "max_tokens", maxTokens
        );

        return webClient.post()
            .uri(uri -> uri
                .path("/openai/deployments/{deployment}/chat/completions")
                .queryParam("api-version", apiVersion)
                .build(deployment))
            .contentType(MediaType.APPLICATION_JSON)
            .bodyValue(body)
            .retrieve()
            .bodyToFlux(String.class)
            .filter(line -> line != null && !line.isBlank())
            .takeWhile(line -> !"data: [DONE]".equals(line.trim()) && !"[DONE]".equals(line.trim()))
            .mapNotNull(this::extractContent)
            .filter(s -> !s.isEmpty());
    }

    private String extractContent(String dataLine) {
        String json = dataLine.startsWith("data:") ? dataLine.substring(5).trim() : dataLine.trim();
        if (json.isEmpty()) return "";
        try {
            JsonNode root = MAPPER.readTree(json);
            JsonNode choices = root.path("choices");
            if (!choices.isArray() || choices.isEmpty()) return "";
            JsonNode delta = choices.path(0).path("delta").path("content");
            return delta.isMissingNode() || delta.isNull() ? "" : delta.asText();
        } catch (Exception e) {
            log.warn("Failed to parse OpenAI SSE chunk: {}", dataLine);
            return "";
        }
    }
}
