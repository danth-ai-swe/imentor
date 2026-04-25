package com.example.chat.domain;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "chunk_sources")
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class ChunkSource {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "message_id")
    private Message message;

    private String name;

    @Column(length = 1024)
    private String url;

    private Integer pageNumber;
    private Integer totalPages;
}
