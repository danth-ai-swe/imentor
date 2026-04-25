package com.example.chat.openai;

import com.example.chat.dto.ChunkDto;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@Component
public class PromptBuilder {

    private static final String TEMPLATE = """
        You are **Insuripedia** — a warm, witty LOMA281/LOMA291 study buddy.
        Answer using ONLY the Retrieved Context. Reply in **%s**.
        Be concise. Open with one warm sentence + a fitting emoji. **Bold** key terms.
        Use bullets for 3+ items. Never invent facts. Never end with a question.

        ## Retrieved Context
        %s

        ## User Question
        %s
        """;

    public String build(List<ChunkDto> chunks, String standaloneQuery, String detectedLanguage) {
        String language = (detectedLanguage == null || detectedLanguage.isBlank()) ? "English" : detectedLanguage;
        String contextStr = chunks == null || chunks.isEmpty()
            ? "(no context)"
            : chunks.stream().map(this::renderChunk).reduce((a, b) -> a + "\n\n---\n\n" + b).orElse("");
        return TEMPLATE.formatted(language, contextStr, standaloneQuery == null ? "" : standaloneQuery);
    }

    private String renderChunk(ChunkDto c) {
        Map<String, Object> meta = c.metadata() == null ? Map.of() : c.metadata();
        String fileName = String.valueOf(meta.getOrDefault("file_name", "unknown"));
        StringBuilder header = new StringBuilder("Source: ").append(fileName);
        for (String key : new String[]{"course", "module_number", "lesson_number", "page_number"}) {
            Object v = meta.get(key);
            if (v != null && !String.valueOf(v).isEmpty()) {
                header.append(" | ").append(prettyKey(key)).append(": ").append(v);
            }
        }
        return "[" + header + "]\n" + (c.text() == null ? "" : c.text());
    }

    private String prettyKey(String k) {
        return switch (k) {
            case "module_number" -> "Module";
            case "lesson_number" -> "Lesson";
            case "page_number"   -> "Page";
            default              -> Character.toUpperCase(k.charAt(0)) + k.substring(1);
        };
    }
}
