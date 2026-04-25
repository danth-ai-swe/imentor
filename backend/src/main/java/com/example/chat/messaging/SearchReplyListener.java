package com.example.chat.messaging;

import com.example.chat.dto.SearchReplyMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

@Component
public class SearchReplyListener {
    private static final Logger log = LoggerFactory.getLogger(SearchReplyListener.class);
    private final SearchRpcClient client;

    public SearchReplyListener(SearchRpcClient client) {
        this.client = client;
    }

    @RabbitListener(queues = "chat.search.reply")
    public void onReply(SearchReplyMessage reply) {
        if (reply == null || reply.correlationId() == null) {
            log.warn("Reply with no correlationId, dropping");
            return;
        }
        client.complete(reply.correlationId(), reply);
    }
}
