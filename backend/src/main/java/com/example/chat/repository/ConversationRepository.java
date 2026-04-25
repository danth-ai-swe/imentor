package com.example.chat.repository;

import com.example.chat.domain.Conversation;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ConversationRepository extends JpaRepository<Conversation, Long> {
    List<Conversation> findByUserUsernameOrderByUpdatedAtDesc(String username);
}
