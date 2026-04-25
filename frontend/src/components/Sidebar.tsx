import { useEffect, useState } from 'react';
import { createConversation, listConversations } from '../api/chatApi';
import type { Conversation, User } from '../types';

interface Props {
  user: User;
  selectedId: number | null;
  onSelect: (c: Conversation) => void;
  refreshKey: number;
}

export function Sidebar({ user, selectedId, onSelect, refreshKey }: Props) {
  const [items, setItems] = useState<Conversation[]>([]);

  useEffect(() => {
    listConversations(user.username).then(setItems);
  }, [user.username, refreshKey]);

  async function newChat() {
    const c = await createConversation(user.id, 'New chat');
    setItems((s) => [c, ...s]);
    onSelect(c);
  }

  return (
    <aside className="sidebar">
      <button className="new-chat-btn" onClick={newChat}>+ New Chat</button>
      <div className="conversations">
        {items.map((c) => (
          <button
            key={c.id}
            className={`conv-item ${c.id === selectedId ? 'selected' : ''}`}
            onClick={() => onSelect(c)}
          >
            <div className="conv-title">{c.title || 'Untitled'}</div>
            <div className="conv-date">{new Date(c.updatedAt).toLocaleString()}</div>
          </button>
        ))}
      </div>
      <div className="user-footer">Signed in as <strong>{user.username}</strong></div>
    </aside>
  );
}
