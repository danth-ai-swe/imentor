package com.example.chat.openai;

import com.example.chat.dto.ChunkDto;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class PromptBuilderTest {

    private final PromptBuilder builder = new PromptBuilder();

    @Test
    void buildsPromptWithLanguageContextAndQuery() {
        ChunkDto chunk = new ChunkDto(
            "Adverse selection occurs when...",
            Map.of("file_name", "loma281.pdf", "page_number", 12)
        );
        String prompt = builder.build(List.of(chunk), "What is adverse selection?", "Vietnamese");
        assertThat(prompt)
            .contains("Reply in **Vietnamese**")
            .contains("Source: loma281.pdf | Page: 12")
            .contains("Adverse selection occurs when...")
            .contains("## User Question")
            .contains("What is adverse selection?");
    }

    @Test
    void usesPlaceholderWhenNoChunks() {
        String prompt = builder.build(List.of(), "Hi", null);
        assertThat(prompt).contains("(no context)").contains("Reply in **English**");
    }
}
