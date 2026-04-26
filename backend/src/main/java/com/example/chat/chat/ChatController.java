package com.example.chat.chat;

import com.example.chat.dto.*;
import jakarta.validation.Valid;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.List;

@RestController
@RequestMapping("/api")
public class ChatController {

    private final ChatService service;

    public ChatController(ChatService service) {
        this.service = service;
    }

    @PostMapping("/users")
    public ApiResponse<UserDto> createUser(@Valid @RequestBody CreateUserRequest req) {
        return ApiResponse.ok(service.getOrCreateUser(req.username()));
    }

    @GetMapping("/users/{username}/conversations")
    public ApiResponse<List<ConversationDto>> listConversations(@PathVariable String username) {
        return ApiResponse.ok(service.listConversations(username));
    }

    @PostMapping("/conversations")
    public ApiResponse<ConversationDto> createConversation(@Valid @RequestBody CreateConversationRequest req) {
        return ApiResponse.ok(service.createConversation(req.userId(), req.title()));
    }

    @GetMapping("/conversations/{id}/messages")
    public ApiResponse<MessagesResponse> listMessages(@PathVariable Long id) {
        return ApiResponse.ok(service.listMessages(id));
    }

    @DeleteMapping("/conversations/{convId}/messages/from/{messageId}")
    public ResponseEntity<Void> deleteFromMessage(@PathVariable Long convId, @PathVariable Long messageId) {
        service.deleteFromMessage(convId, messageId);
        return ResponseEntity.noContent().build();
    }

    @PostMapping(path = "/conversations/{id}/ask/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter ask(@PathVariable Long id, @Valid @RequestBody AskRequest req) {
        return service.ask(id, req.message());
    }
}
