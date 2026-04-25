package com.example.chat.config;

import org.springframework.amqp.core.*;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.support.converter.Jackson2JsonMessageConverter;
import org.springframework.amqp.support.converter.MessageConverter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class RabbitConfig {

    public static final String EXCHANGE = "chat";
    public static final String REQUEST_QUEUE = "chat.search.request";
    public static final String REPLY_QUEUE = "chat.search.reply";
    public static final String REQUEST_RK = "search.request";
    public static final String REPLY_RK = "search.reply";

    @Bean public TopicExchange chatExchange()       { return ExchangeBuilder.topicExchange(EXCHANGE).durable(true).build(); }
    @Bean public Queue searchRequestQueue()         { return QueueBuilder.durable(REQUEST_QUEUE).build(); }
    @Bean public Queue searchReplyQueue()           { return QueueBuilder.durable(REPLY_QUEUE).build(); }
    @Bean public Binding requestBinding()           { return BindingBuilder.bind(searchRequestQueue()).to(chatExchange()).with(REQUEST_RK); }
    @Bean public Binding replyBinding()             { return BindingBuilder.bind(searchReplyQueue()).to(chatExchange()).with(REPLY_RK); }

    @Bean
    public MessageConverter jsonMessageConverter() {
        return new Jackson2JsonMessageConverter();
    }

    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory cf, MessageConverter conv) {
        RabbitTemplate t = new RabbitTemplate(cf);
        t.setMessageConverter(conv);
        t.setExchange(EXCHANGE);
        return t;
    }
}
