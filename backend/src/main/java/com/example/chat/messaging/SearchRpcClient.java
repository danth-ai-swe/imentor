package com.example.chat.messaging;

import com.example.chat.config.RabbitConfig;
import com.example.chat.dto.SearchReplyMessage;
import com.example.chat.dto.SearchRequestMessage;
import org.springframework.amqp.core.MessageProperties;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.UUID;
import java.util.concurrent.*;

@Component
public class SearchRpcClient {

    private final RabbitTemplate rabbit;
    private final long timeoutSeconds;
    private final ConcurrentHashMap<String, CompletableFuture<SearchReplyMessage>> pending = new ConcurrentHashMap<>();

    public SearchRpcClient(
        RabbitTemplate rabbit,
        @Value("${chat.search-timeout-seconds:30}") long timeoutSeconds
    ) {
        this.rabbit = rabbit;
        this.timeoutSeconds = timeoutSeconds;
    }

    public SearchReplyMessage requestSearch(String userMessage, String conversationId) {
        String correlationId = UUID.randomUUID().toString();
        CompletableFuture<SearchReplyMessage> future = new CompletableFuture<>();
        pending.put(correlationId, future);

        SearchRequestMessage payload = new SearchRequestMessage(correlationId, userMessage, conversationId);
        rabbit.convertAndSend(RabbitConfig.EXCHANGE, RabbitConfig.REQUEST_RK, payload, m -> {
            m.getMessageProperties().setCorrelationId(correlationId);
            m.getMessageProperties().setContentType(MessageProperties.CONTENT_TYPE_JSON);
            return m;
        });

        try {
            return future.get(timeoutSeconds, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            pending.remove(correlationId);
            throw new RuntimeException("Search request timed out after " + timeoutSeconds + "s", e);
        } catch (InterruptedException | ExecutionException e) {
            pending.remove(correlationId);
            throw new RuntimeException("Search request failed", e);
        }
    }

    void complete(String correlationId, SearchReplyMessage reply) {
        CompletableFuture<SearchReplyMessage> f = pending.remove(correlationId);
        if (f != null) f.complete(reply);
    }
}
