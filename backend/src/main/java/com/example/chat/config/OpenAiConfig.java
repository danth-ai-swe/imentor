package com.example.chat.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;

import java.time.Duration;

@Configuration
public class OpenAiConfig {

    @Bean(name = "openAiWebClient")
    public WebClient openAiWebClient(
        @Value("${openai.api-base}") String apiBase,
        @Value("${openai.api-key}") String apiKey
    ) {
        HttpClient httpClient = HttpClient.create().responseTimeout(Duration.ofSeconds(120));
        return WebClient.builder()
            .baseUrl(apiBase)
            .defaultHeader("api-key", apiKey)
            .defaultHeader(HttpHeaders.ACCEPT, MediaType.TEXT_EVENT_STREAM_VALUE)
            .clientConnector(new ReactorClientHttpConnector(httpClient))
            .build();
    }
}
