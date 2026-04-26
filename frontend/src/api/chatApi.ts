import type { Conversation, Message, SseHandler, User } from '../types';

const BASE = '/api';

interface ApiResponse<T> { success: boolean; data: T; error?: string; }

async function unwrap<T>(res: Response): Promise<T> {
  const body = (await res.json()) as ApiResponse<T>;
  if (!body.success) throw new Error(body.error ?? `${res.status}`);
  return body.data;
}

export async function createUser(username: string): Promise<User> {
  const res = await fetch(`${BASE}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username }),
  });
  return unwrap<User>(res);
}

export async function listConversations(username: string): Promise<Conversation[]> {
  const res = await fetch(`${BASE}/users/${encodeURIComponent(username)}/conversations`);
  return unwrap<Conversation[]>(res);
}

export async function createConversation(userId: number, title?: string): Promise<Conversation> {
  const res = await fetch(`${BASE}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userId, title }),
  });
  return unwrap<Conversation>(res);
}

export async function loadMessages(conversationId: number): Promise<Message[]> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/messages`);
  const wrapper = await unwrap<{ conversationId: number; messages: Message[] }>(res);
  return wrapper.messages;
}

export async function deleteFromMessage(conversationId: number, messageId: number): Promise<void> {
  await fetch(`${BASE}/conversations/${conversationId}/messages/from/${messageId}`, { method: 'DELETE' });
}

export async function askStream(
  conversationId: number,
  message: string,
  handlers: SseHandler,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ message }),
    signal,
  });
  if (!res.body) throw new Error('No response body');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent = 'message';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let nl;
      while ((nl = buffer.indexOf('\n')) !== -1) {
        const rawLine = buffer.slice(0, nl).replace(/\r$/, '');
        buffer = buffer.slice(nl + 1);

        if (rawLine === '') { currentEvent = 'message'; continue; }
        if (rawLine.startsWith('event:')) { currentEvent = rawLine.slice(6).trim(); continue; }
        if (rawLine.startsWith('data:')) {
          const data = rawLine.slice(5).trim();
          if (!data) continue;
          try {
            const parsed = JSON.parse(data);
            if (currentEvent === 'meta') handlers.onMeta(parsed);
            else if (currentEvent === 'delta') handlers.onDelta(parsed.content ?? '');
            else if (currentEvent === 'done') handlers.onDone();
            else if (currentEvent === 'error') handlers.onError(parsed.message ?? 'unknown error');
          } catch { /* skip malformed line */ }
        }
      }
    }
  } catch (e: any) {
    if (e.name !== 'AbortError') handlers.onError(String(e.message ?? e));
  }
}
