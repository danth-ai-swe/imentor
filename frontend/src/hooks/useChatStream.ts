import { useRef, useState } from 'react';
import { askStream } from '../api/chatApi';
import type { Message, Source } from '../types';

export function useChatStream() {
  const [streaming, setStreaming] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  async function start(
    conversationId: number,
    userMessage: string,
    onAppendUser: (msg: Message) => void,
    onPlaceholder: (placeholder: Message) => void,
    onUpdatePlaceholder: (
      assistantMessageId: number,
      patch: { content?: string; sources?: Source[]; intent?: string;
               detectedLanguage?: string; webSearchUsed?: boolean },
    ) => void,
    onError: (err: string) => void,
  ) {
    if (streaming) return;
    setStreaming(true);

    const tempUserId = -Date.now();
    onAppendUser({
      id: tempUserId, role: 'user', content: userMessage,
      createdAt: new Date().toISOString(), sources: [],
    });

    const placeholderId = -(Date.now() + 1);
    let buffer = '';
    onPlaceholder({
      id: placeholderId, role: 'assistant', content: '',
      createdAt: new Date().toISOString(), sources: [],
    });

    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      await askStream(conversationId, userMessage, {
        onMeta: (meta) => {
          onUpdatePlaceholder(placeholderId, {
            sources: meta.sources, intent: meta.intent,
            detectedLanguage: meta.detectedLanguage, webSearchUsed: meta.webSearchUsed,
          });
        },
        onDelta: (tok) => {
          buffer += tok;
          onUpdatePlaceholder(placeholderId, { content: buffer });
        },
        onDone: () => { /* nothing extra — caller will reload from server */ },
        onError: (msg) => onError(msg),
      }, controller.signal);
    } finally {
      setStreaming(false);
      controllerRef.current = null;
    }
  }

  function stop() {
    controllerRef.current?.abort();
  }

  return { streaming, start, stop };
}
