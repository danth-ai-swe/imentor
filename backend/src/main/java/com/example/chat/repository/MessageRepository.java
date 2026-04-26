package com.example.chat.repository;

import com.example.chat.domain.Message;
import org.springframework.data.domain.Limit;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface MessageRepository extends JpaRepository<Message, Long> {
    List<Message> findByConversationIdOrderByCreatedAtAsc(Long conversationId);
    List<Message> findByConversationIdAndIdGreaterThanEqual(Long conversationId, Long messageId);

    @Query("SELECT m FROM Message m WHERE m.conversation.id = :convId AND m.id < :before AND m.role = 'user' ORDER BY m.id DESC")
    List<Message> findUserMessagesBefore(@Param("convId") Long convId, @Param("before") Long before, Limit limit);

    default Optional<Message> findLatestUserBefore(Long convId, Long before) {
        List<Message> rows = findUserMessagesBefore(convId, before, Limit.of(1));
        return rows.isEmpty() ? Optional.empty() : Optional.of(rows.get(0));
    }
}
