package com.example.chat.chat;

import com.example.chat.dto.*;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api")
public class ChatController {

    private final ChatService service;

    public ChatController(ChatService service) {
        this.service = service;
    }

    @PostMapping("/users")
    public UserDto createUser(@Valid @RequestBody CreateUserRequest req) {
        return service.getOrCreateUser(req.username());
    }

    @GetMapping("/users/{username}/conversations")
    public List<ConversationDto> listConversations(@PathVariable String username) {
        return service.listConversations(username);
    }

    @PostMapping("/conversations")
    public ConversationDto createConversation(@Valid @RequestBody CreateConversationRequest req) {
        return service.createConversation(req.userId(), req.title());
    }

    @GetMapping("/conversations/{id}/messages")
    public List<MessageDto> listMessages(@PathVariable Long id) {
        return service.listMessages(id);
    }

    @DeleteMapping("/conversations/{convId}/messages/from/{messageId}")
    public ResponseEntity<Void> deleteFromMessage(@PathVariable Long convId, @PathVariable Long messageId) {
        service.deleteFromMessage(convId, messageId);
        return ResponseEntity.noContent().build();
    }
}
