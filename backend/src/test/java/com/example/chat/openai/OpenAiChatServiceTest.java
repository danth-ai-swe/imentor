package com.example.chat.openai;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;

import static org.assertj.core.api.Assertions.assertThat;

class OpenAiChatServiceTest {

    private String invokeExtractContent(String line) throws Exception {
        OpenAiChatService svc = new OpenAiChatService(null, "gpt-4o", "2025-01-01-preview", 0.7, 0.95, 3500);
        Method m = OpenAiChatService.class.getDeclaredMethod("extractContent", String.class);
        m.setAccessible(true);
        return (String) m.invoke(svc, line);
    }

    @Test
    void stripsDataPrefixAndExtractsDelta() throws Exception {
        String line = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}";
        assertThat(invokeExtractContent(line)).isEqualTo("Hello");
    }

    @Test
    void handlesLineWithoutPrefix() throws Exception {
        String line = "{\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}";
        assertThat(invokeExtractContent(line)).isEqualTo("Hi");
    }

    @Test
    void returnsEmptyForBlankInput() throws Exception {
        assertThat(invokeExtractContent("data: ")).isEmpty();
        assertThat(invokeExtractContent("")).isEmpty();
    }

    @Test
    void returnsEmptyWhenNoContent() throws Exception {
        String line = "data: {\"choices\":[{\"delta\":{}}]}";
        assertThat(invokeExtractContent(line)).isEmpty();
    }
}
