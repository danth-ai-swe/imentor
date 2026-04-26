import { useRef, useState } from 'react';
import { askStream, regenerateStream } from '../api/chatApi';
import type { Message, SseMeta, Source } from '../types';

type UpdatePatch = {
  content?: string;
  sources?: Source[];
  intent?: string;
  detectedLanguage?: string;
  webSearchUsed?: boolean;
};

export function useChatStream() {
  const [streaming, setStreaming] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  async function start(
    conversationId: number,
    userMessage: string,
    onAppendUser: (m: Message) => void,
    onPlaceholder: (p: Message) => void,
    onUpdate: (placeholderId: number, patch: UpdatePatch) => void,
    onError: (err: string) => void,
  ) {
    if (streaming || regenerating) return;
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
      streamingMode: 'ask',
    });

    const controller = new AbortController();
    controllerRef.current = controller;
    try {
      await askStream(conversationId, userMessage, {
        onMeta: (meta: SseMeta) => onUpdate(placeholderId, {
          sources: meta.sources, intent: meta.intent,
          detectedLanguage: meta.detectedLanguage, webSearchUsed: meta.webSearchUsed,
        }),
        onDelta: (tok) => { buffer += tok; onUpdate(placeholderId, { content: buffer }); },
        onDone: () => { /* nothing extra — caller will reload */ },
        onError: (msg) => onError(msg),
      }, controller.signal);
    } finally {
      setStreaming(false);
      controllerRef.current = null;
    }
  }

  async function regenerate(
    conversationId: number,
    assistantMessageId: number,
    onPlaceholder: (p: Message) => void,
    onUpdate: (placeholderId: number, patch: UpdatePatch) => void,
    onError: (err: string) => void,
  ) {
    if (streaming || regenerating) return;
    setRegenerating(true);
    const placeholderId = -Date.now();
    let buffer = '';
    onPlaceholder({
      id: placeholderId, role: 'assistant', content: '',
      createdAt: new Date().toISOString(), sources: [],
      streamingMode: 'regenerate',
    });

    const controller = new AbortController();
    controllerRef.current = controller;
    try {
      await regenerateStream(conversationId, assistantMessageId, {
        onMeta: (meta: SseMeta) => onUpdate(placeholderId, {
          sources: meta.sources, intent: meta.intent,
          detectedLanguage: meta.detectedLanguage, webSearchUsed: meta.webSearchUsed,
        }),
        onDelta: (tok) => { buffer += tok; onUpdate(placeholderId, { content: buffer }); },
        onDone: () => {},
        onError: (msg) => onError(msg),
      }, controller.signal);
    } finally {
      setRegenerating(false);
      controllerRef.current = null;
    }
  }

  function stop() { controllerRef.current?.abort(); }

  return { streaming, regenerating, start, regenerate, stop };
}
