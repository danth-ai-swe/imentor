import { useEffect, useRef, useState } from 'react';
import { deleteFromMessage, loadMessages } from '../api/chatApi';
import { useChatStream } from '../hooks/useChatStream';
import type { Conversation, Message, Source } from '../types';
import { MessageBubble } from './MessageBubble';
import { SuggestedPrompts } from './SuggestedPrompts';

interface Props { conversation: Conversation; }

export function ChatWindow({ conversation }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editValue, setEditValue] = useState('');
  const [error, setError] = useState<string | null>(null);
  const { streaming, start, stop } = useChatStream();
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadMessages(conversation.id).then(setMessages);
  }, [conversation.id]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [messages]);

  function appendUser(msg: Message) { setMessages((m) => [...m, msg]); }
  function appendPlaceholder(p: Message) { setMessages((m) => [...m, p]); }
  function patchPlaceholder(
    placeholderId: number,
    patch: { content?: string; sources?: Source[]; intent?: string;
             detectedLanguage?: string; webSearchUsed?: boolean },
  ) {
    setMessages((m) => m.map((x) => (x.id === placeholderId ? { ...x, ...patch } as Message : x)));
  }

  async function send(text: string) {
    setError(null);
    setInput('');
    await start(conversation.id, text, appendUser, appendPlaceholder, patchPlaceholder, setError);
    // After streaming completes, reload to pull real DB ids.
    const fresh = await loadMessages(conversation.id);
    setMessages(fresh);
  }

  async function regenerate(assistantMsg: Message) {
    const idx = messages.findIndex((m) => m.id === assistantMsg.id);
    if (idx < 1) return;
    const previousUser = messages[idx - 1];
    if (previousUser.role !== 'user') return;
    await deleteFromMessage(conversation.id, assistantMsg.id);
    setMessages((m) => m.filter((x) => x.id !== assistantMsg.id));
    await send(previousUser.content);
  }

  async function commitEdit(userMsg: Message, newContent: string) {
    setEditingId(null);
    await deleteFromMessage(conversation.id, userMsg.id);
    setMessages((m) => m.filter((x) => x.id < userMsg.id));
    await send(newContent);
  }

  const lastAssistantId = [...messages].reverse().find((m) => m.role === 'assistant')?.id ?? -1;

  return (
    <div className="chat-window">
      <header className="chat-header">{conversation.title}</header>
      <div className="messages" ref={listRef}>
        {messages.length === 0 && !streaming && <SuggestedPrompts onPick={send} />}
        {messages.map((m) => {
          if (editingId === m.id) {
            return (
              <div key={m.id} className="bubble user editing">
                <textarea value={editValue} onChange={(e) => setEditValue(e.target.value)} />
                <div className="edit-actions">
                  <button onClick={() => commitEdit(m, editValue)}>Save</button>
                  <button onClick={() => setEditingId(null)}>Cancel</button>
                </div>
              </div>
            );
          }
          const isPlaceholder = m.id < 0 && m.role === 'assistant';
          return (
            <MessageBubble
              key={m.id}
              message={m}
              streaming={streaming && isPlaceholder}
              isLastAssistant={m.id === lastAssistantId && !streaming}
              onEdit={m.role === 'user' ? () => { setEditingId(m.id); setEditValue(m.content); } : undefined}
              onRegenerate={m.role === 'assistant' ? () => regenerate(m) : undefined}
            />
          );
        })}
      </div>
      {error && <div className="error">{error}</div>}
      <form className="composer" onSubmit={(e) => { e.preventDefault(); if (input.trim()) send(input.trim()); }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask anything…"
          rows={2}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (input.trim()) send(input.trim()); }
          }}
        />
        {streaming
          ? <button type="button" className="stop-btn" onClick={stop}>■ Stop</button>
          : <button type="submit" disabled={!input.trim()}>Send</button>}
      </form>
    </div>
  );
}
