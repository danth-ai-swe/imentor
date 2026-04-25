package com.example.chat.chat;

import com.example.chat.domain.*;
import com.example.chat.dto.*;
import com.example.chat.repository.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public class ChatService {

    private final UserRepository users;
    private final ConversationRepository conversations;
    private final MessageRepository messages;
    private final ChunkSourceRepository chunkSources;

    public ChatService(UserRepository users, ConversationRepository conversations,
                       MessageRepository messages, ChunkSourceRepository chunkSources) {
        this.users = users;
        this.conversations = conversations;
        this.messages = messages;
        this.chunkSources = chunkSources;
    }

    @Transactional
    public UserDto getOrCreateUser(String username) {
        User u = users.findByUsername(username)
            .orElseGet(() -> users.save(User.builder().username(username).build()));
        return new UserDto(u.getId(), u.getUsername());
    }

    @Transactional(readOnly = true)
    public List<ConversationDto> listConversations(String username) {
        return conversations.findByUserUsernameOrderByUpdatedAtDesc(username).stream()
            .map(c -> new ConversationDto(c.getId(), c.getTitle(), c.getCreatedAt(), c.getUpdatedAt()))
            .toList();
    }

    @Transactional
    public ConversationDto createConversation(Long userId, String title) {
        User user = users.findById(userId).orElseThrow(() -> new IllegalArgumentException("user not found"));
        Conversation c = conversations.save(Conversation.builder()
            .user(user)
            .title(title == null || title.isBlank() ? "New chat" : title)
            .build());
        return new ConversationDto(c.getId(), c.getTitle(), c.getCreatedAt(), c.getUpdatedAt());
    }

    @Transactional(readOnly = true)
    public List<MessageDto> listMessages(Long conversationId) {
        return messages.findByConversationIdOrderByCreatedAtAsc(conversationId).stream()
            .map(m -> {
                List<SourceDto> sources = chunkSources.findByMessageId(m.getId()).stream()
                    .map(s -> new SourceDto(s.getName(), s.getUrl(), s.getPageNumber(), s.getTotalPages()))
                    .toList();
                return new MessageDto(
                    m.getId(), m.getRole(), m.getContent(), m.getIntent(),
                    m.getDetectedLanguage(), m.isWebSearchUsed(), m.isStopped(),
                    m.getCreatedAt(), sources
                );
            }).toList();
    }

    @Transactional
    public void deleteFromMessage(Long conversationId, Long messageId) {
        List<Message> toDelete = messages.findByConversationIdAndIdGreaterThanEqual(conversationId, messageId);
        if (toDelete.isEmpty()) return;
        List<Long> ids = toDelete.stream().map(Message::getId).toList();
        chunkSources.deleteByMessageIdIn(ids);
        messages.deleteAll(toDelete);
    }
}
