export interface User { id: number; username: string; }
export interface Conversation { id: number; title: string; createdAt: string; updatedAt: string; }
export interface Source { name: string; url: string; pageNumber: number; totalPages: number; }
export interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  intent?: string;
  detectedLanguage?: string;
  webSearchUsed?: boolean;
  stopped?: boolean;
  createdAt: string;
  sources: Source[];
}

export type SseHandler = {
  onMeta: (meta: { intent: string; detectedLanguage: string; sources: Source[];
                   webSearchUsed: boolean; assistantMessageId: number }) => void;
  onDelta: (token: string) => void;
  onDone: () => void;
  onError: (err: string) => void;
};
