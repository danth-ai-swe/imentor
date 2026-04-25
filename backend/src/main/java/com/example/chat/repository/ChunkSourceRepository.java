package com.example.chat.repository;

import com.example.chat.domain.ChunkSource;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ChunkSourceRepository extends JpaRepository<ChunkSource, Long> {
    List<ChunkSource> findByMessageId(Long messageId);
    void deleteByMessageIdIn(List<Long> messageIds);
}
