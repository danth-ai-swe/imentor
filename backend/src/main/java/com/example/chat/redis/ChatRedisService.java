package com.example.chat.redis;

import com.example.chat.dto.HistoryMessageDto;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DataAccessException;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.List;

@Service
public class ChatRedisService {

    private static final Logger log = LoggerFactory.getLogger(ChatRedisService.class);

    private final StringRedisTemplate template;
    private final ObjectMapper mapper;
    private final long historyTtlSeconds;
    private final int historyMaxEntries;
    private final long streamTtlSeconds;
    private final long chunksTtlSeconds;

    public ChatRedisService(
        StringRedisTemplate template,
        ObjectMapper mapper,
        @Value("${chat.history.redis-ttl-seconds:1800}") long historyTtlSeconds,
        @Value("${chat.history.redis-max-entries:50}") int historyMaxEntries,
        @Value("${chat.stream.redis-ttl-seconds:300}") long streamTtlSeconds,
        @Value("${chat.chunks.ttl-seconds:3600}") long chunksTtlSeconds
    ) {
        this.template = template;
        this.mapper = mapper;
        this.historyTtlSeconds = historyTtlSeconds;
        this.historyMaxEntries = historyMaxEntries;
        this.streamTtlSeconds = streamTtlSeconds;
        this.chunksTtlSeconds = chunksTtlSeconds;
    }

    public void appendHistory(Long conversationId, HistoryMessageDto msg) {
        String key = historyKey(conversationId);
        try {
            String json = mapper.writeValueAsString(msg);
            template.opsForList().rightPush(key, json);
            template.opsForList().trim(key, -historyMaxEntries, -1);
            template.expire(key, Duration.ofSeconds(historyTtlSeconds));
        } catch (JsonProcessingException e) {
            log.warn("Failed to serialize HistoryMessageDto for conv={}: {}", conversationId, e.getMessage());
        } catch (DataAccessException e) {
            log.warn("Redis appendHistory failed for conv={}: {}", conversationId, e.getMessage());
        }
    }

    public List<HistoryMessageDto> readHistoryTail(Long conversationId, int limit) {
        String key = historyKey(conversationId);
        try {
            List<String> jsons = template.opsForList().range(key, -limit, -1);
            if (jsons == null || jsons.isEmpty()) return List.of();
            template.expire(key, Duration.ofSeconds(historyTtlSeconds));
            return jsons.stream().map(this::parseSafe).filter(java.util.Objects::nonNull).toList();
        } catch (DataAccessException e) {
            log.warn("Redis readHistoryTail failed for conv={}: {}", conversationId, e.getMessage());
            return List.of();
        }
    }

    public void invalidateHistory(Long conversationId) {
        try {
            template.delete(historyKey(conversationId));
        } catch (DataAccessException e) {
            log.warn("Redis invalidateHistory failed for conv={}: {}", conversationId, e.getMessage());
        }
    }

    public void appendStreamToken(Long conversationId, String token) {
        String key = streamKey(conversationId);
        try {
            template.opsForValue().append(key, token);
            template.expire(key, Duration.ofSeconds(streamTtlSeconds));
        } catch (DataAccessException e) {
            log.warn("Redis appendStreamToken failed for conv={}: {}", conversationId, e.getMessage());
        }
    }

    public void deleteStream(Long conversationId) {
        try {
            template.delete(streamKey(conversationId));
        } catch (DataAccessException e) {
            log.warn("Redis deleteStream failed for conv={}: {}", conversationId, e.getMessage());
        }
    }

    public void cacheChunks(Long assistantMessageId, com.example.chat.dto.ChunkContext ctx) {
        String key = chunksKey(assistantMessageId);
        try {
            String json = mapper.writeValueAsString(ctx);
            template.opsForValue().set(key, json, Duration.ofSeconds(chunksTtlSeconds));
        } catch (JsonProcessingException e) {
            log.warn("Failed to serialize ChunkContext for msg={}: {}", assistantMessageId, e.getMessage());
        } catch (DataAccessException e) {
            log.warn("Redis cacheChunks failed for msg={}: {}", assistantMessageId, e.getMessage());
        }
    }

    public com.example.chat.dto.ChunkContext readChunks(Long assistantMessageId) {
        String key = chunksKey(assistantMessageId);
        try {
            String json = template.opsForValue().get(key);
            if (json == null) return null;
            return mapper.readValue(json, com.example.chat.dto.ChunkContext.class);
        } catch (JsonProcessingException e) {
            log.warn("Failed to parse ChunkContext for msg={}: {}", assistantMessageId, e.getMessage());
            return null;
        } catch (DataAccessException e) {
            log.warn("Redis readChunks failed for msg={}: {}", assistantMessageId, e.getMessage());
            return null;
        }
    }

    public void deleteChunks(Long assistantMessageId) {
        try {
            template.delete(chunksKey(assistantMessageId));
        } catch (DataAccessException e) {
            log.warn("Redis deleteChunks failed for msg={}: {}", assistantMessageId, e.getMessage());
        }
    }

    private HistoryMessageDto parseSafe(String json) {
        try { return mapper.readValue(json, HistoryMessageDto.class); }
        catch (JsonProcessingException e) {
            log.warn("Skipping malformed history JSON: {}", e.getMessage());
            return null;
        }
    }

    private static String historyKey(Long convId) { return "chat:history:" + convId; }
    private static String streamKey(Long convId)  { return "chat:stream:"  + convId; }
    private static String chunksKey(Long convOrMsgId) { return "chat:chunks:" + convOrMsgId; }
}
